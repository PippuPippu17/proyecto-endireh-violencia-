import polars as pl
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.rutas import RUTA_DATA_RAW, RUTA_DATA_PROCESSED

# Diccionario de corrección directa para cadenas alteradas por encoding
CORRECCION_ENTIDADES = {
    "QUERÃÂTARO": "QUERÉTARO",
    "YUCATÃÂN": "YUCATÁN",
    "SAN LUIS POTOSÃÂ": "SAN LUIS POTOSÍ",
    "MICHOACÃÂN": "MICHOACÁN",
    "MÃÂXICO": "MÉXICO",
    "NUEVO LEÃÂN": "NUEVO LEÓN"
}

DICCIONARIO_ENTIDADES = {
    "01": "AGUASCALIENTES", "02": "BAJA CALIFORNIA", "03": "BAJA CALIFORNIA SUR",
    "04": "CAMPECHE", "05": "COAHUILA", "06": "COLIMA", "07": "CHIAPAS",
    "08": "CHIHUAHUA", "09": "CIUDAD DE MÉXICO", "10": "DURANGO",
    "11": "GUANAJUATO", "12": "GUERRERO", "13": "HIDALGO", "14": "JALISCO",
    "15": "MÉXICO", "16": "MICHOACÁN", "17": "MORELOS", "18": "NAYARIT",
    "19": "NUEVO LEÓN", "20": "OAXACA", "21": "PUEBLA", "22": "QUERÉTARO",
    "23": "QUINTANA ROO", "24": "SAN LUIS POTOSÍ", "25": "SINALOA",
    "26": "SONORA", "27": "TABASCO", "28": "TAMAULIPAS", "29": "TLAXCALA",
    "30": "VERACRUZ", "31": "YUCATÁN", "32": "ZACATECAS"
}

def procesar_endireh():
    archivos_csv = list(RUTA_DATA_RAW.glob("*.csv"))
    if not archivos_csv:
        print(f"[-] No se encontró ningún archivo .csv en '{RUTA_DATA_RAW}'.")
        return

    ruta_csv = archivos_csv[0]
    print(f"[+] Archivo detectado: {ruta_csv.name}")
    print("[+] Cargando dataset con Polars...")

    df = pl.read_csv(ruta_csv, encoding="latin1", infer_schema_length=10000)
    df_limpio = df.unique()

    exprs = []
    cols = df_limpio.columns
    
    if "EDAD" in cols:
        exprs.append(pl.col("EDAD").alias("edad"))
    elif "edad" in cols:
        exprs.append(pl.col("edad"))
        
    if "edad_primer_union" in cols:
        exprs.append(pl.col("edad_primer_union").cast(pl.Utf8))
    elif "P4_2" in cols:
        exprs.append(pl.col("P4_2").cast(pl.Utf8).alias("edad_primer_union"))

    if "num_hijos" in cols:
        exprs.append(pl.col("num_hijos").cast(pl.Utf8))
    elif "P3_7" in cols:
        exprs.append(pl.col("P3_7").cast(pl.Utf8).alias("num_hijos"))

    if "nivel_escolaridad" in cols:
        exprs.append(pl.col("nivel_escolaridad"))
    elif "NIVEL" in cols:
        exprs.append(pl.col("NIVEL").alias("nivel_escolaridad"))

    if "estado_civil_desc" in cols:
        exprs.append(pl.col("estado_civil_desc"))
    elif "P3_1" in cols:
        exprs.append(pl.col("P3_1").alias("estado_civil_desc"))

    if "factor_expansion" in cols:
        exprs.append(pl.col("factor_expansion").cast(pl.Float64))
    elif "FAC_TRI" in cols:
        exprs.append(pl.col("FAC_TRI").cast(pl.Float64).alias("factor_expansion"))
    elif "FAC_MUL" in cols:
        exprs.append(pl.col("FAC_MUL").cast(pl.Float64).alias("factor_expansion"))

    if "CVE_ENT" in cols:
        exprs.append(
            pl.col("CVE_ENT")
            .cast(pl.Utf8)
            .str.zfill(2)
            .replace(DICCIONARIO_ENTIDADES)
            .alias("nom_entidad")
        )
    elif "nom_entidad" in cols:
        exprs.append(
            pl.col("nom_entidad")
            .replace(CORRECCION_ENTIDADES)
            .alias("nom_entidad")
        )

    if "sufrio_violencia_pareja" in cols:
        exprs.append(pl.col("sufrio_violencia_pareja").cast(pl.Int32))

    df_procesado = df_limpio.select(exprs) if exprs else df_limpio

    if "edad" in df_procesado.columns:
        df_procesado = df_procesado.filter(pl.col("edad") >= 15)

    out_parquet = RUTA_DATA_PROCESSED / "endireh_limpio.parquet"
    out_csv = RUTA_DATA_PROCESSED / "endireh_limpio.csv"
    
    df_procesado.write_parquet(out_parquet)
    df_procesado.write_csv(out_csv)
    print(f"[+] Dataset procesado exitosamente: {df_procesado.shape[0]} filas, {df_procesado.shape[1]} columnas.")

if __name__ == "__main__":
    procesar_endireh()

import polars as pl
import numpy as np
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.rutas import RUTA_DATA_PROCESSED

def responder_preguntas_eda():
    ruta_data = RUTA_DATA_PROCESSED / "endireh_limpio.parquet"
    if not ruta_data.exists():
        print("[-] Primero debes ejecutar el script de limpieza (python3 src/cleaning/limpieza.py).")
        return

    df = pl.read_parquet(ruta_data)
    print("="*60)
    print(" ANÁLISIS EXPLORATORIO DE DATOS (ENDIREH 2021) ")
    print("="*60)

    # 1. Edad promedio a la primera unión
    if "edad_primer_union" in df.columns and "factor_expansion" in df.columns:
        # Filtrar solo edades válidas de unión (ej. entre 10 y 95 años)
        df_union = df.with_columns(
            pl.col("edad_primer_union").cast(pl.Float64, strict=False).alias("edad_u_num")
        ).filter(
            (pl.col("edad_u_num") >= 10) & 
            (pl.col("edad_u_num") < 98) & 
            pl.col("edad_u_num").is_not_null()
        )
        
        edades_u = df_union["edad_u_num"].to_numpy()
        factores_u = df_union["factor_expansion"].to_numpy()
        
        if len(edades_u) > 0:
            media_u_ponderada = np.average(edades_u, weights=factores_u)
            print(f"\n1. Edad promedio a la primera unión (Ponderada): {media_u_ponderada:.2f} años")

    # 2. Promedio de hijos por mujer
    if "num_hijos" in df.columns and "factor_expansion" in df.columns:
        df_hijos = df.with_columns(
            pl.col("num_hijos").cast(pl.Float64, strict=False).alias("hijos_num")
        ).filter(
            (pl.col("hijos_num") >= 0) & 
            (pl.col("hijos_num") < 98) & 
            pl.col("hijos_num").is_not_null()
        )
        
        hijos = df_hijos["hijos_num"].to_numpy()
        factores_h = df_hijos["factor_expansion"].to_numpy()
        
        if len(hijos) > 0:
            media_h_ponderada = np.average(hijos, weights=factores_h)
            print(f"2. Número promedio de hijos por mujer (Ponderado): {media_h_ponderada:.2f}")

    # 3. Prevalencia de violencia por entidad federativa
    if "nom_entidad" in df.columns and "sufrio_violencia_pareja" in df.columns:
        print("\n3. Porcentaje de violencia de pareja por Entidad Federativa (Top 10):")
        
        resumen_ent = (
            df.group_by("nom_entidad")
            .agg([
                (pl.col("sufrio_violencia_pareja") * pl.col("factor_expansion")).sum().alias("casos_exp"),
                pl.col("factor_expansion").sum().alias("total_exp")
            ])
            .with_columns(
                ((pl.col("casos_exp") / pl.col("total_exp")) * 100).alias("porcentaje_violencia")
            )
            .sort("porcentaje_violencia", descending=True)
        )
        
        # Corregir codificación visual de nombres de entidades si viene alterada
        resumen_limpio = resumen_ent.with_columns(
            pl.col("nom_entidad")
            .str.replace_all("QUERÃÂTARO", "QUERÉTARO")
            .str.replace_all("YUCATÃÂN", "YUCATÁN")
            .str.replace_all("SAN LUIS POTOSÃÂ", "SAN LUIS POTOSÍ")
            .str.replace_all("MICHOACÃÂN", "MICHOACÁN")
            .str.replace_all("MÃÂXICO", "MÉXICO")
        )
        
        print(resumen_limpio.select(["nom_entidad", "porcentaje_violencia"]).head(10))

if __name__ == "__main__":
    responder_preguntas_eda()

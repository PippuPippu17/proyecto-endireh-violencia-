"""
Preprocesamiento de la ENDIREH 2021 (archivo consolidado).

Pipeline, en este orden:
    1. Carga del CSV crudo en UTF-8.
    2. Validacion de que existan todas las columnas esperadas.
    3. Reparacion del encoding de las columnas de texto.
    4. Conversion de codigos de no respuesta a nulo.
    5. Conversion de tipos de dato.
    6. Eliminacion de duplicados.
    7. Imputacion (solo donde procede; ver NOTA sobre tipos de nulo).
    8. Escritura a data/data-processed/.

El orden importa: los duplicados se eliminan DESPUES de dejar las columnas
en su forma final, y la imputacion va al final para no imputar sobre filas
que luego se descartan.

NOTA sobre los tres tipos de nulo del dataset:
    a) Nulo estructural ("no aplica"): la pregunta no se le hizo a esa mujer.
       Ejemplo: ingreso_pareja es nulo si y solo si pareja_trabaja = No.
       NO se imputa. Inventar un ingreso para una pareja que no trabaja
       introduce informacion que la encuesta nunca levanto.
    b) Nulo por no respuesta: la pregunta si aplicaba, pero la persona no
       supo o no quiso contestar. Vienen codificados (999997/98/99).
       Estos SI se imputan.
    c) Nulo por diseno del consolidado: edad_primer_union solo la tienen las
       mujeres cuya union termino. Se deja como nulo y se documenta.

Uso:
    python3 src/cleaning/limpieza.py
"""

import sys
from pathlib import Path

import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.rutas import RUTA_DATA_PROCESSED, RUTA_DATA_RAW

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------

COLUMNAS_ESPERADAS = [
    "cve_entidad",
    "nom_entidad",
    "cve_municipio",
    "nom_municipio",
    "edad_primer_union",
    "num_hijos",
    "nivel_escolaridad",
    "estado_civil_id",
    "estado_civil_desc",
    "estrato_socioeconomico",
    "pareja_trabaja_id",
    "pareja_trabaja_desc",
    "ingreso_pareja",
    "dinero_propio_id",
    "dinero_propio_desc",
    "apoyo_gobierno_id",
    "apoyo_gobierno_desc",
    "tiene_ahorros_id",
    "tiene_ahorros_desc",
    "propietaria_vivienda_id",
    "propietaria_vivienda_desc",
    "sufrio_violencia_pareja",
    "factor_expansion",
    "anio_encuesta",
]

# Columnas de texto que traen el encoding danado desde el origen.
COLUMNAS_TEXTO = [
    "nom_entidad",
    "nom_municipio",
    "nivel_escolaridad",
    "estado_civil_desc",
    "pareja_trabaja_desc",
    "dinero_propio_desc",
    "apoyo_gobierno_desc",
    "tiene_ahorros_desc",
    "propietaria_vivienda_desc",
]

# Codigos de no respuesta observados en el diagnostico del crudo.
CODIGOS_NO_RESPUESTA = {
    "edad_primer_union": [98.0, 99.0],
    "ingreso_pareja": [999997.0, 999998.0, 999999.0],
}

# Cualitativas que conviene guardar como categoricas.
COLUMNAS_CATEGORICAS = [
    "nom_entidad",
    "nom_municipio",
    "nivel_escolaridad",
    "estado_civil_desc",
    "pareja_trabaja_desc",
    "dinero_propio_desc",
    "apoyo_gobierno_desc",
    "tiene_ahorros_desc",
    "propietaria_vivienda_desc",
]


# --------------------------------------------------------------------------
# Utilidades
# --------------------------------------------------------------------------


def log(mensaje: str) -> None:
    print(f"[limpieza] {mensaje}")


def reparar_mojibake(texto: str) -> str:
    """
    El archivo viene en UTF-8, pero su CONTENIDO ya trae una ronda de
    corruccion de origen: adentro dice literalmente 'MAX(...)ICO' en vez de
    'MEXICO'. Una sola pasada de latin1 -> utf8 lo devuelve a su forma
    correcta. Si el texto ya esta bien, la conversion falla y se devuelve
    intacto, asi que la funcion es segura de aplicar dos veces.
    """
    try:
        return texto.encode("latin1").decode("utf8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


def reparar_columna(df: pl.DataFrame, columna: str) -> pl.DataFrame:
    """
    Aplica la reparacion sobre los valores UNICOS de la columna y luego los
    sustituye. Es mucho mas rapido que recorrer las 110 mil filas: hay 32
    entidades y 1,206 municipios distintos, no 110,127.
    """
    unicos = df[columna].unique().drop_nulls().to_list()
    mapa = {v: reparar_mojibake(v) for v in unicos}
    mapa = {k: v for k, v in mapa.items() if k != v}
    if mapa:
        df = df.with_columns(pl.col(columna).replace(mapa))
    return df, len(mapa)


def resumen_nulos(df: pl.DataFrame, columnas: list[str]) -> str:
    partes = []
    for c in columnas:
        n = df[c].null_count()
        partes.append(f"{c}={n:,} ({100 * n / df.height:.1f}%)")
    return " | ".join(partes)


# --------------------------------------------------------------------------
# Pasos del pipeline
# --------------------------------------------------------------------------


def cargar(ruta: Path) -> pl.DataFrame:
    log(f"Cargando {ruta.name} ...")
    df = pl.read_csv(ruta, encoding="utf8", infer_schema_length=10_000)
    log(f"Crudo: {df.height:,} filas x {df.width} columnas")
    return df


def validar(df: pl.DataFrame) -> None:
    """Falla ruidosamente. Una columna ausente no debe desaparecer en silencio."""
    faltantes = [c for c in COLUMNAS_ESPERADAS if c not in df.columns]
    if faltantes:
        raise KeyError(
            f"El CSV crudo no trae estas columnas esperadas: {faltantes}. "
            f"Columnas encontradas: {df.columns}"
        )
    extra = [c for c in df.columns if c not in COLUMNAS_ESPERADAS]
    if extra:
        log(f"AVISO: el crudo trae columnas no contempladas: {extra}")
    log(f"Validacion: las {len(COLUMNAS_ESPERADAS)} columnas esperadas estan presentes")


def reparar_texto(df: pl.DataFrame) -> pl.DataFrame:
    total = 0
    for c in COLUMNAS_TEXTO:
        df, n = reparar_columna(df, c)
        total += n
    log(
        f"Encoding: {total} valores distintos reparados en {len(COLUMNAS_TEXTO)} columnas"
    )
    return df


def codigos_a_nulo(df: pl.DataFrame) -> pl.DataFrame:
    for columna, codigos in CODIGOS_NO_RESPUESTA.items():
        antes = df[columna].null_count()
        df = df.with_columns(
            pl.when(pl.col(columna).is_in(codigos))
            .then(None)
            .otherwise(pl.col(columna))
            .alias(columna)
        )
        nuevos = df[columna].null_count() - antes
        log(
            f"No respuesta: {columna} -> {nuevos:,} valores {codigos} convertidos a nulo"
        )
    return df


def convertir_tipos(df: pl.DataFrame) -> pl.DataFrame:
    df = df.with_columns(
        [pl.col(c).cast(pl.Categorical) for c in COLUMNAS_CATEGORICAS]
        + [
            pl.col("cve_entidad").cast(pl.Int8),
            pl.col("cve_municipio").cast(pl.Int16),
            pl.col("estado_civil_id").cast(pl.Int8),
            pl.col("estrato_socioeconomico").cast(pl.Int8),
            pl.col("pareja_trabaja_id").cast(pl.Int8),
            pl.col("dinero_propio_id").cast(pl.Int8),
            pl.col("apoyo_gobierno_id").cast(pl.Int8),
            pl.col("tiene_ahorros_id").cast(pl.Int8),
            pl.col("propietaria_vivienda_id").cast(pl.Int8),
            pl.col("sufrio_violencia_pareja").cast(pl.Int8),
            pl.col("factor_expansion").cast(pl.Float64),
            pl.col("edad_primer_union").cast(pl.Float64),
            pl.col("num_hijos").cast(pl.Float64),
            pl.col("ingreso_pareja").cast(pl.Float64),
            pl.col("anio_encuesta").cast(pl.Int16),
        ]
    )
    log(
        "Tipos: cualitativas a Categorical, identificadores a enteros, cuantitativas a Float64"
    )
    return df


def quitar_duplicados(df: pl.DataFrame) -> pl.DataFrame:
    antes = df.height
    # maintain_order=True hace el resultado reproducible entre corridas.
    df = df.unique(maintain_order=True)
    log(
        f"Duplicados: {antes - df.height:,} filas identicas eliminadas ({df.height:,} restantes)"
    )
    return df


def imputar(df: pl.DataFrame) -> pl.DataFrame:
    """
    Solo se imputa ingreso_pareja, y solo en los casos de NO RESPUESTA.

    Criterio: mediana por estrato_socioeconomico.
      - Mediana y no media, porque la distribucion del ingreso esta muy
        sesgada a la derecha (mediana 1,500 contra media 3,817).
      - Por estrato y no global, porque el ingreso guarda relacion con el
        nivel socioeconomico del hogar; imputar con la mediana nacional
        aplanaria esa diferencia.

    Se agrega la bandera ingreso_pareja_imputado para que el EDA pueda
    excluir estos casos y verificar que la imputacion no mueva los
    resultados.
    """
    es_no_respuesta = (pl.col("pareja_trabaja_id") == 1) & pl.col(
        "ingreso_pareja"
    ).is_null()
    mediana_estrato = pl.col("ingreso_pareja").median().over("estrato_socioeconomico")

    n_imputar = df.select(es_no_respuesta.sum()).item()
    df = df.with_columns(
        pl.when(es_no_respuesta)
        .then(mediana_estrato)
        .otherwise(pl.col("ingreso_pareja"))
        .alias("ingreso_pareja"),
        es_no_respuesta.alias("ingreso_pareja_imputado"),
    )
    log(f"Imputacion: {n_imputar:,} valores de ingreso_pareja (mediana por estrato)")

    estructurales = df.select((pl.col("ingreso_pareja").is_null()).sum()).item()
    log(
        f"Imputacion: {estructurales:,} nulos estructurales de ingreso_pareja NO se imputan (pareja no trabaja)"
    )
    return df


def guardar(df: pl.DataFrame) -> None:
    RUTA_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.write_parquet(RUTA_DATA_PROCESSED / "endireh_limpio.parquet")
    df.write_csv(RUTA_DATA_PROCESSED / "endireh_limpio.csv")
    log(f"Guardado en {RUTA_DATA_PROCESSED}: {df.height:,} filas x {df.width} columnas")


# --------------------------------------------------------------------------


def procesar_endireh() -> pl.DataFrame:
    archivos = sorted(RUTA_DATA_RAW.glob("*.csv"))
    if not archivos:
        raise FileNotFoundError(
            f"No hay ningun .csv en {RUTA_DATA_RAW}. "
            "Coloca ahi el archivo consolidado que entrego la ayudantia."
        )
    if len(archivos) > 1:
        log(f"AVISO: hay {len(archivos)} archivos .csv; se usara {archivos[0].name}")

    print("=" * 70)
    df = cargar(archivos[0])
    validar(df)

    cuantitativas = ["edad_primer_union", "num_hijos", "ingreso_pareja"]
    log(f"Nulos ANTES: {resumen_nulos(df, cuantitativas)}")
    print("-" * 70)

    df = reparar_texto(df)
    df = codigos_a_nulo(df)
    df = convertir_tipos(df)
    df = quitar_duplicados(df)
    df = imputar(df)

    print("-" * 70)
    log(f"Nulos DESPUES: {resumen_nulos(df, cuantitativas)}")
    guardar(df)
    print("=" * 70)
    return df


if __name__ == "__main__":
    procesar_endireh()

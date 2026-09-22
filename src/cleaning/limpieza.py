"""
Preprocesamiento de la ENDIREH 2021.

Lee el archivo consolidado de data/data-raw/, lo deja consistente y escribe el
resultado en data/data-processed/ como Parquet y CSV.

Pasos: carga, validacion de columnas, reparacion de encoding, conversion de
codigos de no respuesta a nulo, conversion de tipos, eliminacion de duplicados
e imputacion.

Uso:
    python3 src/cleaning/limpieza.py
"""

import sys
from pathlib import Path

import polars as pl

# .parent.parent.parent sube de src/cleaning/ a la raiz del proyecto.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.rutas import RUTA_DATA_PROCESSED, RUTA_DATA_RAW

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------

# Columnas que debe traer el archivo consolidado.
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

# Columnas de texto cuyos acentos vienen danados desde el origen.
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

# Valores con que el INEGI marca la no respuesta en cada variable.
CODIGOS_NO_RESPUESTA = {
    "edad_primer_union": [98.0, 99.0],
    "ingreso_pareja": [999997.0, 999998.0, 999999.0],
}

# Cualitativas que se almacenan como Categorical.
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
    """Imprime un mensaje del pipeline con un prefijo que lo identifica."""
    print(f"[limpieza] {mensaje}")


def reparar_mojibake(texto: str) -> str:
    """
    Devuelve el texto con los acentos corregidos.

    El archivo viene en UTF-8 pero su contenido ya estaba mal interpretado
    desde el origen: dentro dice 'MÃ‰XICO' en lugar de 'MÉXICO'. Volver a
    codificar en latin1 y releer en utf8 deshace esa interpretacion. Si el
    texto ya esta bien, la conversion falla y se devuelve intacto.
    """
    try:
        return texto.encode("latin1").decode("utf8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


def reparar_columna(df: pl.DataFrame, columna: str) -> tuple[pl.DataFrame, int]:
    """
    Repara una columna de texto y devuelve cuantos valores distintos cambiaron.

    La correccion se calcula sobre los valores unicos y se sustituye en bloque.
    """
    unicos = df[columna].unique().drop_nulls().to_list()
    mapa = {v: reparar_mojibake(v) for v in unicos}
    mapa = {k: v for k, v in mapa.items() if k != v}

    if mapa:
        df = df.with_columns(pl.col(columna).replace(mapa))
    return df, len(mapa)


def resumen_nulos(df: pl.DataFrame, columnas: list[str]) -> str:
    """Arma una linea con el conteo y el porcentaje de nulos de cada columna."""
    partes = []
    for c in columnas:
        n = df[c].null_count()
        partes.append(f"{c}={n:,} ({100 * n / df.height:.1f}%)")
    return " | ".join(partes)


# --------------------------------------------------------------------------
# Pasos del pipeline
# --------------------------------------------------------------------------


def cargar(ruta: Path) -> pl.DataFrame:
    """Lee el CSV crudo en UTF-8 y reporta sus dimensiones."""
    log(f"Cargando {ruta.name} ...")

    # Muestra de inferencia ampliada: tipa bien las columnas con nulos al inicio.
    df = pl.read_csv(ruta, encoding="utf8", infer_schema_length=10_000)

    log(f"Crudo: {df.height:,} filas x {df.width} columnas")
    return df


def validar(df: pl.DataFrame) -> None:
    """
    Verifica que el crudo traiga las columnas esperadas.

    Lanza KeyError si falta alguna. Las columnas no contempladas solo se
    reportan en el log.
    """
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
    """Corrige el encoding de todas las columnas de texto del dataset."""
    total = 0
    for c in COLUMNAS_TEXTO:
        df, n = reparar_columna(df, c)
        total += n
    log(
        f"Encoding: {total} valores distintos reparados en {len(COLUMNAS_TEXTO)} columnas"
    )
    return df


def codigos_a_nulo(df: pl.DataFrame) -> pl.DataFrame:
    """
    Convierte a nulo los codigos de no respuesta.

    Son valores centinela que caen dentro del rango numerico de la variable:
    98 y 99 en edad_primer_union, 999997 a 999999 en ingreso_pareja.
    """
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
    """Asigna a cada columna su tipo definitivo."""
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
    """
    Elimina las filas identicas en todas sus columnas.

    El criterio es la identidad de fila completa; el dataset no trae
    identificador de registro. maintain_order=True fija el orden de salida.
    """
    antes = df.height
    df = df.unique(maintain_order=True)
    log(
        f"Duplicados: {antes - df.height:,} filas identicas eliminadas ({df.height:,} restantes)"
    )
    return df


def imputar(df: pl.DataFrame) -> pl.DataFrame:
    """
    Rellena los nulos de ingreso_pareja que corresponden a no respuesta.

    Alcanza solo a los registros cuya pareja trabaja; cuando no trabaja el nulo
    es estructural y se conserva. El valor imputado es la mediana del ingreso
    dentro del estrato socioeconomico del registro.

    Agrega la columna ingreso_pareja_imputado, que marca los registros
    afectados.
    """
    es_no_respuesta = (pl.col("pareja_trabaja_id") == 1) & pl.col(
        "ingreso_pareja"
    ).is_null()

    # .over() calcula la mediana por estrato sin colapsar las filas.
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
    """
    Escribe el resultado en data/data-processed/, en Parquet y en CSV.

    El Parquet conserva los tipos de dato y es el que leen el EDA y las
    graficas.
    """
    RUTA_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.write_parquet(RUTA_DATA_PROCESSED / "endireh_limpio.parquet")
    df.write_csv(RUTA_DATA_PROCESSED / "endireh_limpio.csv")
    log(f"Guardado en {RUTA_DATA_PROCESSED}: {df.height:,} filas x {df.width} columnas")


# --------------------------------------------------------------------------


def procesar_endireh() -> pl.DataFrame:
    """
    Ejecuta el pipeline completo y devuelve el DataFrame resultante.

    El orden de los pasos condiciona el resultado: la deduplicacion opera sobre
    los tipos ya fijados y los codigos ya convertidos a nulo, y la imputacion
    se calcula sobre las filas que sobreviven.

    El archivo de entrada se localiza por su extension dentro de data-raw/.
    """
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

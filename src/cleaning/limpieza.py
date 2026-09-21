"""
Preprocesamiento de la ENDIREH 2021.

Lee el archivo consolidado desde data/data-raw/, lo deja consistente y escribe
el resultado en data/data-processed/ en formato Parquet y CSV.

El pipeline ejecuta ocho pasos en este orden:

    1. Carga        Lee el CSV en UTF-8.
    2. Validacion   Verifica que esten las 24 columnas esperadas.
    3. Encoding     Repara los acentos danados en las columnas de texto.
    4. No respuesta Convierte los codigos centinela a nulo.
    5. Tipos        Asigna a cada columna su tipo definitivo.
    6. Duplicados   Elimina las filas identicas.
    7. Imputacion   Rellena los nulos que corresponden a no respuesta.
    8. Escritura    Guarda el resultado.

El orden de los pasos 4 a 7 no es intercambiable:

    - Los codigos de no respuesta se convierten a nulo ANTES de deduplicar,
      porque dos filas que solo se distinguen por el codigo empleado (98
      frente a 99) describen el mismo registro.
    - La deduplicacion ocurre DESPUES de fijar los tipos, de modo que se
      comparen valores y no representaciones de texto.
    - La imputacion va al final, para no calcular medianas sobre filas que
      despues se eliminan.

Los nulos del dataset son de tres naturalezas distintas y cada una recibe un
tratamiento propio:

    a) Estructural. La pregunta no aplicaba. ingreso_pareja es nulo si y solo
       si pareja_trabaja = No. No se imputa: un ingreso inventado para una
       pareja que no trabaja agrega informacion que la encuesta no levanto.
    b) No respuesta. La pregunta aplicaba pero no se contesto, y viene con un
       codigo centinela. Este si se imputa.
    c) De diseno. edad_primer_union solo tiene valor en las mujeres cuya union
       termino. Se conserva como nulo.

Uso:
    python3 src/cleaning/limpieza.py
"""

import sys
from pathlib import Path

import polars as pl

# Agregamos la raiz del proyecto al path para poder importar config/.
# __file__: ruta de este archivo (src/cleaning/limpieza.py)
# .parent.parent.parent: sube tres niveles hasta la raiz del proyecto.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.rutas import RUTA_DATA_PROCESSED, RUTA_DATA_RAW

# --------------------------------------------------------------------------
# Configuracion
# --------------------------------------------------------------------------

# Las 24 columnas que debe traer el archivo consolidado. Si falta alguna, el
# pipeline se detiene en validar().
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

# Valores que el INEGI usa para marcar no respuesta dentro del rango de cada
# variable. Se identificaron revisando la distribucion del archivo crudo.
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

    El CSV esta codificado en UTF-8, pero su contenido ya venia mal
    interpretado desde el origen: dentro del archivo dice literalmente
    'MÃ‰XICO' en lugar de 'MÉXICO'. Deshacer esa interpretacion requiere
    recorrer el camino inverso:

        .encode("latin1")  recupera los bytes originales.
        .decode("utf8")    los vuelve a leer con la codificacion correcta.

    Cuando el texto ya esta bien, alguno de los dos pasos lanza excepcion y la
    funcion lo devuelve sin cambios, de modo que aplicarla dos veces sobre el
    mismo texto es inofensivo.
    """
    try:
        return texto.encode("latin1").decode("utf8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return texto


def reparar_columna(df: pl.DataFrame, columna: str) -> tuple[pl.DataFrame, int]:
    """
    Repara una columna de texto.

    Devuelve el DataFrame ya corregido y cuantos valores distintos cambiaron.

    La correccion se calcula sobre el catalogo de valores UNICOS y despues se
    sustituye en bloque. nom_municipio, por ejemplo, tiene 1,206 valores
    distintos repartidos en 110,127 filas: reparar el catalogo y reemplazar es
    dos ordenes de magnitud mas barato que recorrer fila por fila.
    """
    # Catalogo de valores distintos de la columna, sin nulos.
    unicos = df[columna].unique().drop_nulls().to_list()

    # Mapa valor_original -> valor_reparado, conservando solo los que cambian.
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

    # infer_schema_length=10_000: polars deduce el tipo de cada columna a
    # partir de las primeras 10 mil filas en vez de las 100 por omision, para
    # que una columna con muchos nulos al inicio no termine leida como texto.
    df = pl.read_csv(ruta, encoding="utf8", infer_schema_length=10_000)

    log(f"Crudo: {df.height:,} filas x {df.width} columnas")
    return df


def validar(df: pl.DataFrame) -> None:
    """
    Verifica que el crudo traiga las columnas esperadas.

    Lanza KeyError si falta alguna. Continuar sin ella haria que la columna
    desapareciera del resultado sin que nadie se entere, asi que el pipeline se
    detiene. Las columnas de mas solo se avisan, porque no rompen nada.
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
    Convierte a nulo los codigos de no respuesta de cada columna.

    En las encuestas del INEGI la no respuesta se codifica con valores
    centinela que caen dentro del rango numerico de la variable. Tratarlos como
    numeros reales distorsiona cualquier medida: la media de ingreso_pareja
    pasa de $3,817 a $86,351 si los codigos 999997 a 999999 no se convierten.
    """
    for columna, codigos in CODIGOS_NO_RESPUESTA.items():
        antes = df[columna].null_count()

        # when/then/otherwise funciona como un if por fila: donde el valor este
        # en la lista de codigos se escribe nulo, en el resto se deja igual.
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
    """
    Asigna a cada columna su tipo definitivo.

    Las cualitativas pasan a Categorical, que guarda una sola copia de cada
    etiqueta y en las filas solo un indice. Las claves y banderas se guardan
    como enteros pequenos, y las cuantitativas como Float64 para que admitan
    nulos y decimales.
    """
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

    maintain_order=True conserva el orden de aparicion. Sin ese argumento
    polars puede devolver las filas en un orden distinto en cada ejecucion, y
    el archivo de salida dejaria de ser reproducible.

    El dataset no trae identificador de registro, asi que el unico criterio
    disponible es la identidad de fila completa.
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

    La imputacion se aplica unicamente cuando la pareja si trabaja: ahi el
    ingreso existe y no fue declarado. Cuando la pareja no trabaja el nulo es
    estructural y se conserva.

    El valor imputado es la mediana del estrato socioeconomico de la mujer:

      - Mediana y no media, porque la distribucion del ingreso esta muy sesgada
        a la derecha (mediana $1,500 frente a media $3,820).
      - Por estrato y no global, porque el ingreso varia con el nivel
        socioeconomico del hogar: las medianas van de $900 en el estrato 1 a
        $6,000 en el estrato 4, y una mediana nacional aplanaria esa diferencia.

    La columna ingreso_pareja_imputado marca los registros afectados, de modo
    que el EDA pueda excluirlos y comprobar que la imputacion no desplaza los
    resultados.
    """
    # Condicion de no respuesta: la pareja trabaja, pero el ingreso esta vacio.
    es_no_respuesta = (pl.col("pareja_trabaja_id") == 1) & pl.col(
        "ingreso_pareja"
    ).is_null()

    # Mediana del ingreso dentro de cada estrato. .over() la calcula por grupo
    # sin colapsar las filas, e ignora los nulos al calcularla.
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

    # Los nulos que sobreviven son los estructurales.
    estructurales = df.select((pl.col("ingreso_pareja").is_null()).sum()).item()
    log(
        f"Imputacion: {estructurales:,} nulos estructurales de ingreso_pareja NO se imputan (pareja no trabaja)"
    )
    return df


def guardar(df: pl.DataFrame) -> None:
    """
    Escribe el resultado en data/data-processed/.

    Se guardan dos formatos: Parquet conserva los tipos de dato y ocupa una
    fraccion del espacio, y es el que leen el EDA y las graficas; el CSV queda
    disponible para inspeccionar el resultado sin necesidad de polars.
    """
    RUTA_DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    df.write_parquet(RUTA_DATA_PROCESSED / "endireh_limpio.parquet")
    df.write_csv(RUTA_DATA_PROCESSED / "endireh_limpio.csv")
    log(f"Guardado en {RUTA_DATA_PROCESSED}: {df.height:,} filas x {df.width} columnas")


# --------------------------------------------------------------------------


def procesar_endireh() -> pl.DataFrame:
    """
    Ejecuta el pipeline completo y devuelve el DataFrame resultante.

    Busca el archivo en data/data-raw/ sin depender de su nombre, porque el
    consolidado se entrega con el nombre de cada estudiante al inicio.
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

    # Fotografia de los nulos antes de tocar nada, para contrastar al final.
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

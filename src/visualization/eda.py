"""
Analisis exploratorio de la ENDIREH 2021.

Calcula e imprime las medidas descriptivas de los Pasos 5, 6 y 7 sobre el
dataset ya preprocesado: moda y frecuencias de las cualitativas, medidas de
localizacion en version simple y ponderada, y medidas de variabilidad
comparando el grupo que reporto violencia de pareja contra el que no.

Cada variable se filtra por separado. Los valores se toman completos, y las
incompatibilidades entre una variable y su nombre se reportan como avisos junto
a la medida correspondiente.

Las funciones seccion_* son el punto de entrada que usa el notebook.

Uso:
    python3 src/visualization/eda.py
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.rutas import RUTA_DATA_PROCESSED

# Variables cuantitativas del dataset; ingreso_pareja es la unica continua.
VARIABLES_CUANTITATIVAS = ["edad_primer_union", "num_hijos", "ingreso_pareja"]

VARIABLES_CUALITATIVAS = [
    "nivel_escolaridad",
    "estado_civil_desc",
    "nom_entidad",
    "estrato_socioeconomico",
]

ANCHO = 78


# --------------------------------------------------------------------------
# Funciones estadisticas
# --------------------------------------------------------------------------


def cuantil_ponderado(valores: np.ndarray, pesos: np.ndarray, q: float) -> float:
    """
    Devuelve el cuantil q de los valores, ponderado por los pesos.

    Ordena los valores, acumula los pesos y devuelve el primero cuyo peso
    acumulado alcanza la fraccion q del total: la definicion de CDF inversa.
    numpy y polars no ofrecen cuantiles con pesos.
    """
    orden = np.argsort(valores)
    v = valores[orden]
    w = pesos[orden]
    acumulado = np.cumsum(w)
    corte = q * acumulado[-1]
    idx = int(np.searchsorted(acumulado, corte))
    return float(v[min(idx, len(v) - 1)])


def moda_ponderada(valores: np.ndarray, pesos: np.ndarray):
    """
    Devuelve el valor con mayor peso acumulado.

    Suma los factores de expansion de cada valor distinto y devuelve el de
    mayor total.
    """
    unicos = np.unique(valores)
    pesos_por_valor = [pesos[valores == u].sum() for u in unicos]
    return float(unicos[int(np.argmax(pesos_por_valor))])


def medidas_localizacion(valores: np.ndarray, pesos: np.ndarray) -> dict:
    """
    Calcula las medidas de localizacion de una variable.

    Cada medida se devuelve en dos versiones: la simple, sobre la muestra
    encuestada, y la ponderada, sobre la poblacion estimada. n_modas indica
    cuantos valores empatan en frecuencia.
    """
    unicos, cuentas = np.unique(valores, return_counts=True)
    moda_simple = float(unicos[int(np.argmax(cuentas))])
    n_modas = int((cuentas == cuentas.max()).sum())

    return {
        "n": len(valores),
        "media": float(np.mean(valores)),
        "mediana": float(np.median(valores)),
        "moda": moda_simple,
        "n_modas": n_modas,
        "Q1": float(np.percentile(valores, 25)),
        "Q2": float(np.percentile(valores, 50)),
        "Q3": float(np.percentile(valores, 75)),
        "P10": float(np.percentile(valores, 10)),
        "P90": float(np.percentile(valores, 90)),
        "media_pond": float(np.average(valores, weights=pesos)),
        "mediana_pond": cuantil_ponderado(valores, pesos, 0.50),
        "moda_pond": moda_ponderada(valores, pesos),
        "Q1_pond": cuantil_ponderado(valores, pesos, 0.25),
        "Q3_pond": cuantil_ponderado(valores, pesos, 0.75),
        "P10_pond": cuantil_ponderado(valores, pesos, 0.10),
        "P90_pond": cuantil_ponderado(valores, pesos, 0.90),
        "poblacion": float(pesos.sum()),
    }


def medidas_variabilidad(valores: np.ndarray) -> dict:
    """
    Calcula las medidas de dispersion de una variable.

    ddof=1 produce la varianza muestral, que divide entre n-1.
    """
    media = float(np.mean(valores))
    desv = float(np.std(valores, ddof=1))
    q1 = float(np.percentile(valores, 25))
    q3 = float(np.percentile(valores, 75))
    return {
        "n": len(valores),
        "media": media,
        "rango": float(np.max(valores) - np.min(valores)),
        "minimo": float(np.min(valores)),
        "maximo": float(np.max(valores)),
        "varianza": desv**2,
        "desv_est": desv,
        # El CV no esta definido con media cero.
        "CV": (desv / media * 100) if media != 0 else float("nan"),
        "IQR": q3 - q1,
    }


# --------------------------------------------------------------------------
# Presentacion
# --------------------------------------------------------------------------


def titulo(texto: str, caracter: str = "=") -> None:
    """Imprime un encabezado de seccion."""
    print("\n" + caracter * ANCHO)
    print(f" {texto}")
    print(caracter * ANCHO)


def serie_valida(df: pl.DataFrame, columna: str) -> pl.DataFrame:
    """
    Devuelve las filas en que la columna indicada tiene valor.

    Filtra unicamente por esa columna; los nulos de las demas no intervienen.
    """
    return df.filter(pl.col(columna).is_not_null())


def alertas_de_coherencia(columna: str, valores: np.ndarray) -> list[str]:
    """
    Devuelve los avisos aplicables a una variable.

    Contrasta los valores observados contra lo que su nombre afirma.
    """
    avisos = []
    if columna == "edad_primer_union":
        imposibles = int((valores < 10).sum())
        if imposibles:
            avisos.append(
                f"{imposibles:,} valores ({100 * imposibles / len(valores):.1f}%) "
                f"son menores a 10. Una edad a la primera union no puede serlo."
            )
    if columna == "num_hijos":
        distintos = np.unique(valores)
        if len(distintos) <= 6:
            avisos.append(
                f"solo toma {len(distintos)} valores distintos {distintos.tolist()}. "
                f"Se comporta como un codigo, no como un conteo."
            )
    return avisos


def imprimir_localizacion(columna: str, m: dict, avisos: list[str]) -> None:
    """Imprime la tabla de medidas de localizacion de una variable."""
    print(f"\n--- {columna}")
    print(
        f"    n = {m['n']:,} observaciones | poblacion estimada = {m['poblacion']:,.0f} mujeres"
    )
    if avisos:
        for a in avisos:
            print(f"    [!] AVISO: {a}")
    print()
    print(f"    {'Medida':<12} {'Sin ponderar':>14} {'Ponderada':>14}")
    print(f"    {'-' * 42}")
    filas = [
        ("Media", m["media"], m["media_pond"]),
        ("Mediana (Q2)", m["mediana"], m["mediana_pond"]),
        ("Moda", m["moda"], m["moda_pond"]),
        ("Q1", m["Q1"], m["Q1_pond"]),
        ("Q3", m["Q3"], m["Q3_pond"]),
        ("P10", m["P10"], m["P10_pond"]),
        ("P90", m["P90"], m["P90_pond"]),
    ]
    for nombre, simple, pond in filas:
        print(f"    {nombre:<12} {simple:>14,.2f} {pond:>14,.2f}")
    if m["n_modas"] > 1:
        print(
            f"    (la distribucion es multimodal: {m['n_modas']} valores empatan en frecuencia)"
        )
    dif = m["media_pond"] - m["media"]
    print(f"\n    Diferencia media ponderada - media simple: {dif:+.2f}")


def imprimir_variabilidad(columna: str, con: dict, sin: dict, total: dict) -> None:
    """Imprime la tabla comparativa de dispersion entre los dos grupos."""
    print(f"\n--- {columna}")
    etiquetas = ["Medida", "Total", "Con violencia", "Sin violencia"]
    print(
        f"    {etiquetas[0]:<16}{etiquetas[1]:>14}{etiquetas[2]:>16}{etiquetas[3]:>16}"
    )
    print(f"    {'-' * 60}")
    for clave, nombre in [
        ("n", "n"),
        ("media", "Media"),
        ("rango", "Rango"),
        ("varianza", "Varianza"),
        ("desv_est", "Desv. estandar"),
        ("CV", "CV (%)"),
        ("IQR", "IQR"),
    ]:
        if clave == "n":
            print(
                f"    {nombre:<16}{total[clave]:>14,}{con[clave]:>16,}{sin[clave]:>16,}"
            )
        else:
            print(
                f"    {nombre:<16}{total[clave]:>14,.2f}{con[clave]:>16,.2f}{sin[clave]:>16,.2f}"
            )

    if not (np.isnan(con["CV"]) or np.isnan(sin["CV"])):
        razon = con["CV"] / sin["CV"] if sin["CV"] else float("nan")
        print(f"\n    Razon de CV (con/sin): {razon:.2f}")


# --------------------------------------------------------------------------
# Secciones del analisis
# --------------------------------------------------------------------------


def seccion_localizacion(df: pl.DataFrame) -> None:
    """Imprime las medidas de localizacion de las variables cuantitativas."""
    titulo("PASO 6: MEDIDAS DE LOCALIZACION")
    print("\nLa media ponderada usa factor_expansion. La mediana, los cuartiles y")
    print("los percentiles ponderados se calculan con una funcion propia, porque")
    print("ni polars ni numpy ofrecen cuantiles con pesos.")

    for columna in VARIABLES_CUANTITATIVAS:
        sub = serie_valida(df, columna)
        valores = sub[columna].to_numpy()
        pesos = sub["factor_expansion"].to_numpy()
        m = medidas_localizacion(valores, pesos)
        imprimir_localizacion(columna, m, alertas_de_coherencia(columna, valores))


def seccion_variabilidad(df: pl.DataFrame) -> None:
    """Imprime la comparacion de dispersion entre los grupos con y sin violencia."""
    titulo("PASO 7: MEDIDAS DE VARIABILIDAD")
    print("\nComparacion entre el grupo que reporto violencia de pareja y el que no.")
    print("Las medidas van sin ponderar: el CV compara dispersion relativa dentro")
    print("de cada grupo, no estima un total poblacional. Es una limitacion a")
    print("declarar en el reporte.")

    for columna in VARIABLES_CUANTITATIVAS:
        sub = serie_valida(df, columna)
        total = medidas_variabilidad(sub[columna].to_numpy())
        con = medidas_variabilidad(
            sub.filter(pl.col("sufrio_violencia_pareja") == 1)[columna].to_numpy()
        )
        sin = medidas_variabilidad(
            sub.filter(pl.col("sufrio_violencia_pareja") == 0)[columna].to_numpy()
        )
        imprimir_variabilidad(columna, con, sin, total)


def seccion_cualitativas(df: pl.DataFrame) -> None:
    """Imprime moda y tabla de frecuencias de las variables cualitativas."""
    titulo("PASO 5: VARIABLES CUALITATIVAS (moda y frecuencias)")
    print("\nEn una variable cualitativa la unica medida de tendencia central")
    print("valida es la moda. En una ordinal ademas tiene sentido la mediana,")
    print("pero nunca la media.")

    for columna in VARIABLES_CUALITATIVAS:
        resumen = (
            df.group_by(columna)
            .agg(
                pl.len().alias("frecuencia"),
                pl.col("factor_expansion").sum().alias("poblacion"),
            )
            .with_columns(
                (100 * pl.col("frecuencia") / df.height).alias("pct_muestra"),
                (100 * pl.col("poblacion") / df["factor_expansion"].sum()).alias(
                    "pct_poblacion"
                ),
            )
            .sort("poblacion", descending=True)
        )
        moda = resumen[columna][0]
        print(f"\n--- {columna}   (moda ponderada: {moda})")
        print(f"    {'Categoria':<34}{'Frec.':>10}{'% muestra':>12}{'% poblacion':>14}")
        print(f"    {'-' * 70}")
        for fila in resumen.head(10).iter_rows(named=True):
            print(
                f"    {str(fila[columna]):<34}{fila['frecuencia']:>10,}"
                f"{fila['pct_muestra']:>11.1f}%{fila['pct_poblacion']:>13.1f}%"
            )
        if resumen.height > 10:
            print(f"    ... y {resumen.height - 10} categorias mas")


def seccion_prevalencia(df: pl.DataFrame) -> None:
    """Imprime la prevalencia ponderada de violencia, nacional y por grupo."""
    titulo("PREVALENCIA DE VIOLENCIA DE PAREJA (ponderada)")

    total = 100 * (
        (df["sufrio_violencia_pareja"] * df["factor_expansion"]).sum()
        / df["factor_expansion"].sum()
    )
    print(f"\nPrevalencia nacional ponderada: {total:.1f}%")
    print("Interpretacion: es la proporcion de mujeres que REPORTARON violencia")
    print("de pareja en la encuesta, no una medicion directa de su ocurrencia.")

    for columna in ["nom_entidad", "estado_civil_desc"]:
        resumen = (
            df.group_by(columna)
            .agg(
                (pl.col("sufrio_violencia_pareja") * pl.col("factor_expansion"))
                .sum()
                .alias("casos"),
                pl.col("factor_expansion").sum().alias("total"),
            )
            .with_columns(
                (100 * pl.col("casos") / pl.col("total")).alias("prevalencia")
            )
            .sort("prevalencia", descending=True)
        )
        print(f"\n--- Por {columna}")
        for fila in resumen.head(10).iter_rows(named=True):
            print(f"    {str(fila[columna]):<38}{fila['prevalencia']:>8.1f}%")
        if resumen.height > 10:
            print(f"    ... ({resumen.height - 10} categorias mas)")


# --------------------------------------------------------------------------


def ejecutar_eda() -> None:
    """Carga el dataset procesado e imprime el analisis completo."""
    ruta = RUTA_DATA_PROCESSED / "endireh_limpio.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta antes: python3 src/cleaning/limpieza.py"
        )

    df = pl.read_parquet(ruta)
    titulo("ANALISIS EXPLORATORIO DE DATOS - ENDIREH 2021")
    print(f"\nRegistros: {df.height:,}   Variables: {df.width}")
    print(
        f"Poblacion representada: {df['factor_expansion'].sum():,.0f} mujeres de 15 anios y mas"
    )

    seccion_cualitativas(df)
    seccion_localizacion(df)
    seccion_variabilidad(df)
    seccion_prevalencia(df)
    print("\n" + "=" * ANCHO)


if __name__ == "__main__":
    ejecutar_eda()

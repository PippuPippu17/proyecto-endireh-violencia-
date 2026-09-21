"""
Graficas del analisis exploratorio de la ENDIREH 2021.

Genera cuatro figuras en reports/figuras/ listas para insertar en el reporte:

    01_distribucion_edad_primer_union.png
    02_ingreso_por_violencia.png
    03_prevalencia_por_entidad.png
    04_prevalencia_por_estado_civil.png

Criterios de diseno aplicados:
  - Una sola familia de color. Azul y naranja para las dos categorias
    comparadas; el par esta verificado para daltonismo (delta E 24.7 en
    protanopia, muy por encima del minimo de 8).
  - En las graficas de una sola serie no se usa color para codificar el
    orden: todas las barras son del mismo color y el rango lo da la
    posicion. Colorear por ranking sugiere una diferencia de categoria que
    no existe.
  - Ejes y retícula en gris tenue para que no compitan con los datos.
  - Toda prevalencia esta PONDERADA por factor_expansion; el pie de cada
    figura lo declara junto con el tamano de muestra.

Uso:
    python3 src/visualization/graficas.py
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: necesario para guardar a archivo
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import polars as pl

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.rutas import BASE_DIR, RUTA_DATA_PROCESSED

RUTA_FIGURAS = BASE_DIR / "reports" / "figuras"

# Paleta
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"
AZUL = "#2a78d6"  # categoria 1 / serie unica
NARANJA = "#eb6834"  # categoria 2
GRIS_RETICULA = "#d8d7d2"

plt.rcParams.update(
    {
        "figure.facecolor": SUPERFICIE,
        "axes.facecolor": SUPERFICIE,
        "savefig.facecolor": SUPERFICIE,
        "text.color": TINTA,
        "axes.labelcolor": TINTA_SUAVE,
        "xtick.color": TINTA_SUAVE,
        "ytick.color": TINTA_SUAVE,
        "axes.edgecolor": GRIS_RETICULA,
        "font.size": 10,
        "figure.dpi": 130,
    }
)


def preparar_ejes(ax, eje_valor: str = "y") -> None:
    """Retícula tenue en el eje de valores y sin marcos superfluos."""
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(GRIS_RETICULA)
    ax.spines["bottom"].set_color(GRIS_RETICULA)
    ax.grid(axis=eje_valor, color=GRIS_RETICULA, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def titular(ax, titulo: str, subtitulo: str) -> None:
    """Titulo y subtitulo separados en PUNTOS, no en fraccion de ejes: una
    fraccion fija se lee distinto en una figura de 5 pulgadas que en una de
    11, y el subtitulo terminaba encima del titulo."""
    ax.set_title(
        titulo, fontsize=13, fontweight="bold", color=TINTA, loc="left", pad=26
    )
    ax.annotate(
        subtitulo,
        xy=(0, 1),
        xycoords="axes fraction",
        xytext=(0, 7),
        textcoords="offset points",
        fontsize=9.5,
        color=TINTA_SUAVE,
        va="bottom",
        ha="left",
    )


def marcar_nacional(ax, valor: float) -> None:
    """Linea de referencia nacional, etiquetada arriba dentro del area de
    graficado para no invadir el eje."""
    ax.axvline(valor, color=NARANJA, linewidth=1.6, linestyle="--")
    # Aire arriba para que la etiqueta no caiga encima de la barra mas alta.
    y0, y1 = ax.get_ylim()
    ax.set_ylim(y0, y1 + (y1 - y0) * 0.07)
    trans = blended_transform_factory(ax.transData, ax.transAxes)
    ax.annotate(
        f"Nacional {valor:.1f}%",
        xy=(valor, 1),
        xycoords=trans,
        xytext=(5, -10),
        textcoords="offset points",
        color=NARANJA,
        fontsize=9,
        va="top",
        ha="left",
    )


def pie_de_figura(ax, texto: str) -> None:
    """Nota al pie anclada debajo del eje, medida en puntos para que no
    colisione con la etiqueta del eje en ninguna figura."""
    ax.annotate(
        texto,
        xy=(0, 0),
        xycoords="axes fraction",
        xytext=(0, -52),
        textcoords="offset points",
        fontsize=8,
        color=TINTA_SUAVE,
        va="top",
        ha="left",
    )


def guardar(fig, nombre: str) -> None:
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = RUTA_FIGURAS / nombre
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"[graficas] {ruta.relative_to(BASE_DIR)}")


def prevalencia_ponderada(df: pl.DataFrame, columna: str) -> pl.DataFrame:
    return (
        df.group_by(columna)
        .agg(
            (pl.col("sufrio_violencia_pareja") * pl.col("factor_expansion"))
            .sum()
            .alias("casos"),
            pl.col("factor_expansion").sum().alias("total"),
            pl.len().alias("n"),
        )
        .with_columns((100 * pl.col("casos") / pl.col("total")).alias("prevalencia"))
        .sort("prevalencia")
    )


# --------------------------------------------------------------------------


def figura_01_distribucion_edad(df: pl.DataFrame) -> None:
    """
    Histograma de edad_primer_union. La figura documenta la anomalia en vez
    de esconderla: la zona imposible para una edad va sombreada y anotada.
    """
    valores = df.filter(pl.col("edad_primer_union").is_not_null())[
        "edad_primer_union"
    ].to_numpy()
    imposibles = int((valores < 10).sum())

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(
        valores,
        bins=np.arange(0, valores.max() + 2, 1),
        color=AZUL,
        edgecolor=SUPERFICIE,
        linewidth=0.4,
    )

    ax.axvspan(-0.5, 9.5, color=NARANJA, alpha=0.10, zorder=0)
    ax.axvline(9.5, color=NARANJA, linewidth=1.6, linestyle="--")
    ax.annotate(
        f"{imposibles:,} valores ({100 * imposibles / len(valores):.1f}%)\n"
        "por debajo de 10: imposibles\ncomo edad a la primera union",
        xy=(9.5, ax.get_ylim()[1] * 0.78),
        xytext=(22, ax.get_ylim()[1] * 0.80),
        fontsize=9,
        color=NARANJA,
        arrowprops=dict(arrowstyle="->", color=NARANJA, linewidth=1.2),
    )

    preparar_ejes(ax)
    titular(
        ax,
        "La variable edad_primer_union no se comporta como una edad",
        "Distribucion de frecuencias de los valores observados",
    )
    ax.set_xlabel("Valor declarado")
    ax.set_ylabel("Numero de mujeres encuestadas")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {len(valores):,} registros con valor. "
        "Los codigos 98 y 99 ya fueron convertidos a nulo en el preprocesamiento.",
    )
    guardar(fig, "01_distribucion_edad_primer_union.png")


def figura_02_ingreso_por_violencia(df: pl.DataFrame) -> None:
    """
    Comparacion del ingreso de la pareja entre grupos, por percentiles.

    Se descarto el boxplot: en escala logaritmica las dos cajas se veian
    casi iguales y en escala lineal la cola de 800,000 aplastaba todo. El
    hallazgo real es DONDE se separan las dos distribuciones, y para eso
    hay que comparar percentil contra percentil.
    """
    sub = df.filter(
        pl.col("ingreso_pareja").is_not_null() & (pl.col("ingreso_pareja") > 0)
    )
    con = sub.filter(pl.col("sufrio_violencia_pareja") == 1)[
        "ingreso_pareja"
    ].to_numpy()
    sin = sub.filter(pl.col("sufrio_violencia_pareja") == 0)[
        "ingreso_pareja"
    ].to_numpy()

    etiquetas = ["P10", "Q1 (P25)", "Mediana", "Q3 (P75)", "P90"]
    cortes = [10, 25, 50, 75, 90]
    v_con = np.percentile(con, cortes)
    v_sin = np.percentile(sin, cortes)

    y = np.arange(len(etiquetas))
    alto = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y + alto / 2, v_sin, height=alto, color=AZUL, label="No reporto violencia")
    ax.barh(y - alto / 2, v_con, height=alto, color=NARANJA, label="Reporto violencia")

    for yi, (vs, vc) in enumerate(zip(v_sin, v_con)):
        ax.text(
            vs + 150,
            yi + alto / 2,
            f"${vs:,.0f}",
            va="center",
            fontsize=9,
            color=TINTA_SUAVE,
        )
        ax.text(
            vc + 150,
            yi - alto / 2,
            f"${vc:,.0f}",
            va="center",
            fontsize=9,
            color=TINTA_SUAVE,
        )

    ax.set_yticks(y, etiquetas)
    ax.invert_yaxis()
    ax.set_xlim(0, max(v_sin.max(), v_con.max()) * 1.18)
    preparar_ejes(ax, eje_valor="x")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    titular(
        ax,
        "Los dos grupos solo se separan en la mitad alta de la distribucion",
        "Ingreso mensual de la pareja por percentil, segun si se reporto violencia",
    )
    ax.set_xlabel("Ingreso de la pareja (pesos)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {len(con):,} con violencia y {len(sin):,} sin violencia; "
        "se excluyen ingresos en cero.\n"
        "Hasta la mediana las distribuciones son casi identicas; la brecha aparece en Q3 y P90. "
        "La asociacion no implica causalidad:\ntambien puede reflejar diferencias en la "
        "disposicion a reportar, no solo en la ocurrencia.",
    )
    guardar(fig, "02_ingreso_por_violencia.png")


def figura_03_prevalencia_entidad(df: pl.DataFrame) -> None:
    resumen = prevalencia_ponderada(df, "nom_entidad")
    nombres = resumen["nom_entidad"].to_list()
    valores = resumen["prevalencia"].to_numpy()
    nacional = 100 * (
        (df["sufrio_violencia_pareja"] * df["factor_expansion"]).sum()
        / df["factor_expansion"].sum()
    )

    fig, ax = plt.subplots(figsize=(9, 11))
    ax.barh(nombres, valores, color=AZUL, height=0.72)
    marcar_nacional(ax, nacional)

    for y, v in enumerate(valores):
        ax.text(v + 0.2, y, f"{v:.1f}%", va="center", fontsize=8.5, color=TINTA_SUAVE)

    ax.set_xlim(0, valores.max() * 1.15)
    preparar_ejes(ax, eje_valor="x")
    titular(
        ax,
        "Prevalencia de violencia de pareja por entidad federativa",
        "Porcentaje de mujeres de 15 anios y mas que la reportaron, ponderado",
    )
    ax.set_xlabel("Prevalencia (%)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {df.height:,} registros, ponderados por "
        "factor_expansion. Mide lo REPORTADO en la encuesta, no la ocurrencia real.",
    )
    guardar(fig, "03_prevalencia_por_entidad.png")


def figura_04_prevalencia_estado_civil(df: pl.DataFrame) -> None:
    resumen = prevalencia_ponderada(df, "estado_civil_desc")
    nombres = [str(x) for x in resumen["estado_civil_desc"].to_list()]
    valores = resumen["prevalencia"].to_numpy()
    nacional = 100 * (
        (df["sufrio_violencia_pareja"] * df["factor_expansion"]).sum()
        / df["factor_expansion"].sum()
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(nombres, valores, color=AZUL, height=0.62)
    marcar_nacional(ax, nacional)

    for y, v in enumerate(valores):
        ax.text(v + 0.5, y, f"{v:.1f}%", va="center", fontsize=9.5, color=TINTA_SUAVE)

    ax.set_xlim(0, valores.max() * 1.18)
    preparar_ejes(ax, eje_valor="x")
    titular(
        ax,
        "Las mujeres separadas y divorciadas reportan el triple que las casadas",
        "Prevalencia ponderada de violencia de pareja por estado conyugal",
    )
    ax.set_xlabel("Prevalencia (%)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {df.height:,} registros ponderados.\n"
        "Lectura: lo mas probable es que la violencia haya precedido a la separacion, "
        "no que separarse la provoque. El dato no permite establecer la direccion causal.",
    )
    guardar(fig, "04_prevalencia_por_estado_civil.png")


# --------------------------------------------------------------------------


def generar_graficas() -> None:
    ruta = RUTA_DATA_PROCESSED / "endireh_limpio.parquet"
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta}. Ejecuta antes: python3 src/cleaning/limpieza.py"
        )
    df = pl.read_parquet(ruta)
    print(f"[graficas] Dataset: {df.height:,} filas. Guardando en {RUTA_FIGURAS}")

    figura_01_distribucion_edad(df)
    figura_02_ingreso_por_violencia(df)
    figura_03_prevalencia_entidad(df)
    figura_04_prevalencia_estado_civil(df)
    print("[graficas] Listo: 4 figuras generadas.")


if __name__ == "__main__":
    generar_graficas()

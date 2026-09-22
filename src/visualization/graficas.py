"""
Graficas del analisis exploratorio de la ENDIREH 2021.

Construye cuatro figuras y las guarda en reports/figuras/, listas para
insertarse en el reporte:

    01_distribucion_edad_primer_union.png
    02_ingreso_por_violencia.png
    03_prevalencia_por_entidad.png
    04_prevalencia_por_estado_civil.png
    05_dispersion_edad_por_violencia.png
    06_escolaridad_muestra_vs_poblacion.png

Cuatro criterios de diseno se aplican en todas ellas:

  - Color. Azul y naranja identifican a los dos grupos comparados. El par esta
    verificado para daltonismo: separacion de 24.7 en protanopia, frente al
    minimo recomendado de 8.
  - Las graficas de una sola serie usan un unico color. El color no codifica
    el lugar en el ranking, porque sugiere diferencias de categoria que no
    existen; el orden ya lo comunica la posicion de cada barra.
  - Ejes y reticula en gris tenue, para que la tinta del encuadre no compita
    con la de los datos.
  - Toda prevalencia esta ponderada por factor_expansion, y el pie de cada
    figura lo declara junto con el tamano de muestra.

Uso:
    python3 src/visualization/graficas.py
"""

import sys
from pathlib import Path

import matplotlib

# Agg dibuja en memoria en lugar de abrir una ventana. Debe elegirse antes de
# importar pyplot, y permite generar las figuras en una terminal sin entorno
# grafico.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np
import polars as pl

# Agregamos la raiz del proyecto al path para poder importar config/.
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from config.rutas import BASE_DIR, RUTA_DATA_PROCESSED

RUTA_FIGURAS = BASE_DIR / "reports" / "figuras"

# Paleta comun a las cuatro figuras.
SUPERFICIE = "#fcfcfb"
TINTA = "#0b0b0b"
TINTA_SUAVE = "#52514e"
AZUL = "#2a78d6"  # categoria 1 / serie unica
NARANJA = "#eb6834"  # categoria 2
GRIS_RETICULA = "#d8d7d2"

# rcParams fija los valores por omision de matplotlib para todo el modulo, de
# modo que cada figura no tenga que repetir colores ni tamanos.
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
    """
    Deja los ejes con reticula tenue y sin marcos superfluos.

    eje_valor indica cual de los dos ejes lleva la magnitud: "y" en las
    graficas de barras verticales e histogramas, "x" en las horizontales. Solo
    ese eje recibe reticula; el de categorias no la necesita.
    """
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    ax.spines["left"].set_color(GRIS_RETICULA)
    ax.spines["bottom"].set_color(GRIS_RETICULA)
    ax.grid(axis=eje_valor, color=GRIS_RETICULA, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def titular(ax, titulo: str, subtitulo: str) -> None:
    """
    Escribe el titulo y el subtitulo por encima del area de graficado.

    La separacion se mide en puntos y no en fraccion de los ejes, porque una
    fraccion fija equivale a distinta distancia segun el alto de la figura: lo
    que separa bien en una de 5 pulgadas superpone los textos en una de 11.
    """
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
    """
    Dibuja la linea de referencia nacional y la etiqueta.

    La etiqueta se coloca dentro del area de graficado, en la parte superior,
    para no invadir el eje. blended_transform_factory combina dos sistemas de
    coordenadas: la posicion horizontal se expresa en unidades de los datos,
    para que siga a la linea, y la vertical en fraccion de los ejes, para que
    quede arriba sea cual sea la escala.
    """
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
    """Escribe la nota al pie con la fuente de los datos y el tamano de muestra."""
    # El desplazamiento va en puntos respecto de la esquina inferior izquierda
    # de los ejes, de modo que la nota quede por debajo de la etiqueta del eje
    # en cualquier figura, sin importar su alto.
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
    """Escribe la figura en reports/figuras/ y libera la memoria que ocupaba."""
    RUTA_FIGURAS.mkdir(parents=True, exist_ok=True)
    ruta = RUTA_FIGURAS / nombre
    # bbox_inches="tight" recorta el margen sobrante y, de paso, asegura que
    # los textos colocados fuera de los ejes entren en la imagen.
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    print(f"[graficas] {ruta.relative_to(BASE_DIR)}")


def prevalencia_ponderada(df: pl.DataFrame, columna: str) -> pl.DataFrame:
    """
    Calcula la prevalencia de violencia dentro de cada categoria de la columna.

    Es una proporcion ponderada: suma los factores de expansion de las mujeres
    que reportaron violencia y los divide entre la suma de factores de todas
    las del grupo. Devuelve las categorias ordenadas de menor a mayor.
    """
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
    Histograma de edad_primer_union.

    La figura muestra la anomalia de la variable en lugar de ocultarla: la
    franja de valores imposibles para una edad va sombreada y anotada con su
    peso dentro del total.
    """
    valores = df.filter(pl.col("edad_primer_union").is_not_null())[
        "edad_primer_union"
    ].to_numpy()
    imposibles = int((valores < 10).sum())

    fig, ax = plt.subplots(figsize=(9, 5))
    # Un bin por valor entero: la variable se declara en anios completos, y
    # agrupar en intervalos mas anchos borraria los picos en los multiplos de
    # cinco, que son la huella del redondeo al declarar.
    ax.hist(
        valores,
        bins=np.arange(0, valores.max() + 2, 1),
        color=AZUL,
        edgecolor=SUPERFICIE,
        linewidth=0.4,
    )

    ax.axvspan(-0.5, 9.5, color=NARANJA, alpha=0.10, zorder=0)
    ax.axvline(9.5, color=NARANJA, linewidth=1.6, linestyle="--")
    # La flecha y el texto comparten altura, y el texto se centra
    # verticalmente sobre ella, de modo que la flecha quede horizontal.
    altura = ax.get_ylim()[1] * 0.72
    ax.annotate(
        f"{imposibles:,} valores ({100 * imposibles / len(valores):.1f}%)\n"
        "por debajo de 10 años",
        xy=(9.5, altura),
        xytext=(24, altura),
        fontsize=9,
        color=NARANJA,
        va="center",
        ha="left",
        arrowprops=dict(arrowstyle="->", color=NARANJA, linewidth=1.2),
    )

    preparar_ejes(ax)
    titular(
        ax,
        "Distribución de la variable edad_primer_union",
        "Declarada como edad a la primera unión, pero el "
        f"{100 * imposibles / len(valores):.1f}% de los valores es menor a 10",
    )
    ax.set_xlabel("Valor declarado")
    ax.set_ylabel("Mujeres encuestadas")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {len(valores):,} registros con valor.\n"
        "Los códigos 98 y 99 se excluyen por corresponder a no respuesta.",
    )
    guardar(fig, "01_distribucion_edad_primer_union.png")


def figura_02_ingreso_por_violencia(df: pl.DataFrame) -> None:
    """
    Compara el ingreso de la pareja entre los dos grupos, percentil a percentil.

    La comparacion se hace por percentiles y no con un diagrama de caja porque
    lo relevante es en que tramo de la distribucion se separan los grupos: las
    medianas son casi iguales y la brecha aparece en Q3 y P90. Un boxplot
    resume esos mismos cortes, pero la cola hasta 800,000 obliga a una escala
    en la que las cajas quedan indistinguibles.
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

    # Cinco cortes de la distribucion, de la cola baja a la alta.
    etiquetas = ["P10", "Q1 (P25)", "Mediana", "Q3 (P75)", "P90"]
    cortes = [10, 25, 50, 75, 90]
    v_con = np.percentile(con, cortes)
    v_sin = np.percentile(sin, cortes)

    # Cada percentil ocupa una posicion entera del eje y; las dos barras se
    # desplazan media altura en sentidos opuestos para quedar enfrentadas.
    y = np.arange(len(etiquetas))
    alto = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y + alto / 2, v_sin, height=alto, color=AZUL, label="No reportó violencia")
    ax.barh(y - alto / 2, v_con, height=alto, color=NARANJA, label="Reportó violencia")

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
    # Invertimos el eje para leer los percentiles de menor a mayor hacia abajo.
    ax.invert_yaxis()
    ax.set_xlim(0, max(v_sin.max(), v_con.max()) * 1.18)
    preparar_ejes(ax, eje_valor="x")
    ax.legend(frameon=False, loc="upper right", fontsize=9.5)
    titular(
        ax,
        "La diferencia de ingreso se concentra en la mitad alta de la distribución",
        "Ingreso mensual de la pareja por percentil, según si se reportó violencia",
    )
    ax.set_xlabel("Ingreso de la pareja (pesos)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {len(con):,} con violencia y {len(sin):,} sin violencia; "
        "se excluyen los ingresos en cero.\n"
        "Hasta la mediana ambas distribuciones coinciden; la diferencia aparece en Q3 y P90.\n"
        "La asociación no permite establecer causalidad: también puede reflejar diferencias "
        "en la disposición a reportar.",
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
        "Porcentaje ponderado de mujeres de 15 años y más que la reportaron",
    )
    ax.set_xlabel("Prevalencia (%)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {df.height:,} registros ponderados por factor_expansion.\n"
        "La cifra corresponde a la violencia reportada en la encuesta, no a su ocurrencia.",
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
        "La encuesta es transversal: registra el estado conyugal y la violencia en un mismo\n"
        "momento, por lo que no permite establecer el orden entre ambos.",
    )
    guardar(fig, "04_prevalencia_por_estado_civil.png")


def figura_05_dispersion_edad(df: pl.DataFrame) -> None:
    """
    Diagrama de caja de edad_primer_union segun si se reporto violencia.

    Pone en imagen la comparacion de dispersion del Paso 7: la caja abarca el
    rango intercuartilico y la linea interior marca la mediana, de modo que dos
    cajas iguales describen dos distribuciones con la misma forma.
    """
    sub = df.filter(pl.col("edad_primer_union").is_not_null())
    con = sub.filter(pl.col("sufrio_violencia_pareja") == 1)[
        "edad_primer_union"
    ].to_numpy()
    sin = sub.filter(pl.col("sufrio_violencia_pareja") == 0)[
        "edad_primer_union"
    ].to_numpy()

    fig, ax = plt.subplots(figsize=(9, 4.8))
    cajas = ax.boxplot(
        [sin, con],
        tick_labels=["No reportó\nviolencia", "Reportó\nviolencia"],
        vert=False,
        patch_artist=True,
        widths=0.5,
        showfliers=False,
        medianprops=dict(color=TINTA, linewidth=2),
        whiskerprops=dict(color=TINTA_SUAVE, linewidth=1.2),
        capprops=dict(color=TINTA_SUAVE, linewidth=1.2),
        boxprops=dict(linewidth=0),
    )
    for caja, color in zip(cajas["boxes"], (AZUL, NARANJA)):
        caja.set_facecolor(color)

    # Los cuartiles coinciden en ambos grupos, asi que se rotulan una sola vez.
    q1, mediana, q3 = np.percentile(sin, [25, 50, 75])
    for valor, etiqueta in ((q1, "Q1"), (mediana, "Mediana"), (q3, "Q3")):
        ax.annotate(
            f"{etiqueta} = {valor:.0f}",
            xy=(valor, 2.42),
            fontsize=8.5,
            color=TINTA_SUAVE,
            ha="center",
        )

    ax.set_ylim(0.4, 2.7)
    preparar_ejes(ax, eje_valor="x")
    titular(
        ax,
        "Ambos grupos comparten la misma distribución",
        "Dispersión de edad_primer_union según si se reportó violencia de pareja",
    )
    ax.set_xlabel("Valor declarado")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {len(con):,} con violencia y {len(sin):,} sin violencia; "
        "no se dibujan los valores atípicos.\n"
        "Los dos grupos coinciden en Q1 (2), mediana (6), Q3 (15), IQR (13) y media (10.03): "
        "la variable no discrimina\nentre quienes reportaron violencia y quienes no.",
    )
    guardar(fig, "05_dispersion_edad_por_violencia.png")


def figura_06_escolaridad(df: pl.DataFrame) -> None:
    """
    Distribucion de nivel_escolaridad en la muestra y en la poblacion.

    Contrasta el peso de cada categoria antes y despues de aplicar el factor de
    expansion. La distancia entre ambas barras es el efecto de la ponderacion, y
    explica por que las medidas simples y las ponderadas no coinciden.
    """
    resumen = (
        df.group_by("nivel_escolaridad")
        .agg(
            pl.len().alias("n"),
            pl.col("factor_expansion").sum().alias("poblacion"),
        )
        .with_columns(
            (100 * pl.col("n") / df.height).alias("pct_muestra"),
            (100 * pl.col("poblacion") / df["factor_expansion"].sum()).alias("pct_pob"),
        )
        # Orden natural de las categorias: al ser ordinal no se reordena por
        # frecuencia, porque el orden forma parte de la informacion.
        .sort("nivel_escolaridad")
    )
    categorias = [str(x) for x in resumen["nivel_escolaridad"].to_list()]
    muestra = resumen["pct_muestra"].to_numpy()
    poblacion = resumen["pct_pob"].to_numpy()

    y = np.arange(len(categorias))
    alto = 0.38

    fig, ax = plt.subplots(figsize=(9, 5.2))
    ax.barh(
        y + alto / 2, poblacion, height=alto, color=AZUL, label="Población estimada"
    )
    ax.barh(
        y - alto / 2, muestra, height=alto, color=NARANJA, label="Muestra encuestada"
    )

    for yi, (pob, mue) in enumerate(zip(poblacion, muestra)):
        ax.text(
            pob + 0.6,
            yi + alto / 2,
            f"{pob:.1f}%",
            va="center",
            fontsize=9,
            color=TINTA_SUAVE,
        )
        ax.text(
            mue + 0.6,
            yi - alto / 2,
            f"{mue:.1f}%",
            va="center",
            fontsize=9,
            color=TINTA_SUAVE,
        )

    ax.set_yticks(y, categorias)
    ax.invert_yaxis()
    ax.set_xlim(0, max(poblacion.max(), muestra.max()) * 1.20)
    preparar_ejes(ax, eje_valor="x")
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    titular(
        ax,
        "La ponderación redistribuye el peso entre categorías",
        "Distribución de nivel_escolaridad en la muestra y en la población estimada",
    )
    ax.set_xlabel("Porcentaje del total (%)")
    pie_de_figura(
        ax,
        f"ENDIREH 2021 (INEGI). n = {df.height:,} registros; la población estimada pondera "
        "cada registro por factor_expansion.\n"
        "A1 pierde 7.4 puntos al ponderar y C1 gana 5.3. Esa redistribución es la razón por "
        "la que una medida simple y\nsu versión ponderada no coinciden. Las categorías A1 a "
        "C2 no corresponden al catálogo educativo del INEGI\ny se presentan sin interpretar "
        "su contenido.",
    )
    guardar(fig, "06_escolaridad_muestra_vs_poblacion.png")


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
    figura_05_dispersion_edad(df)
    figura_06_escolaridad(df)
    print("[graficas] Listo: 6 figuras generadas.")


if __name__ == "__main__":
    generar_graficas()

# Proyecto ENDIREH 2021: Análisis Exploratorio sobre Violencia contra las Mujeres

**Universidad Nacional Autónoma de México — Facultad de Ciencias**
Almacenes y Minería de Datos · Práctica 3

---

## 1. Objetivo

Estructurar un proyecto de minería de datos con una arquitectura de carpetas
reproducible, seleccionar un framework de análisis, aplicar un preprocesamiento
riguroso y describir los datos mediante medidas de localización y variabilidad,
como etapas previas a cualquier modelado.

El caso de estudio es la Encuesta Nacional sobre la Dinámica de las Relaciones
en los Hogares (ENDIREH) 2021. Por tratarse de un tema social sensible, el
análisis distingue en todo momento entre la violencia **reportada** en la
encuesta y la ocurrencia del fenómeno, y evita atribuir causalidad a
asociaciones observadas en un diseño transversal.

## 2. Fuente de los datos

Los microdatos provienen de la **ENDIREH 2021**, levantada por el Instituto
Nacional de Estadística y Geografía (INEGI).

- Programa: <https://www.inegi.org.mx/programas/endireh/2021/>
- Descriptor de archivos (diccionario de variables):
  <https://www.inegi.org.mx/contenidos/programas/endireh/2021/doc/endireh2021_fd.pdf>

El análisis no parte de los microdatos originales, repartidos en 28 tablas, sino
del **archivo consolidado** que proporcionó la ayudantía del curso: un único CSV
con 110,127 registros y 24 columnas ya unidas y renombradas.

Ese archivo **no forma parte del repositorio**. El `.gitignore` excluye el
contenido de `data/data-raw/` por su tamaño, conforme a lo indicado en la
práctica; el archivo se entrega por separado. Para reproducir el análisis hay
que colocarlo en `data/data-raw/` antes de ejecutar el pipeline. El script lo
localiza por su extensión, de modo que el nombre del archivo es indistinto.

## 3. Framework de análisis

El pipeline usa **polars** en todas sus etapas.
Frente a pandas, polars ofrece un motor multihilo escrito en Rust y evaluación
perezosa, con una API de sintaxis cercana y es por eso que se eligió para este proyecto.

**numpy** cubre los cálculos que polars no trae de forma nativa: la media
ponderada por el factor de expansión y una implementación propia de cuantiles
ponderados. **matplotlib** genera las figuras.

## 4. Requisitos e instalación

Requiere **Python 3.10 o superior**; el proyecto se desarrolló con Python 3.14.

> [!NOTE]
> Las instrucciones son de acuerdo con el sistema que usamos, que en este caso es
> Arch Linux. En otros sistemas operativos, la instalación de Python y la creación
> del entorno virtual pueden variar.

```bash
# 1. Clonar el repositorio
git clone https://github.com/PippuPippu17/proyecto-endireh-violencia-.git
cd proyecto-endireh-violencia-

# 2. Crear y activar el entorno virtual
python -m venv env
source env/bin/activate

# 3. Instalar las dependencias
pip install -r requirements.txt
```

El entorno debe llamarse `env` o `.venv`: son los dos nombres que el
`.gitignore` excluye.

En Arch Linux, `python -m venv` requiere el paquete `python-pip` instalado a
nivel de sistema.

Dependencias fijadas en `requirements.txt`:

| Paquete      | Versión | Uso                                    |
| ------------ | ------- | -------------------------------------- |
| `polars`     | 1.44.2  | Framework de análisis del pipeline     |
| `pyarrow`    | 25.0.1  | Soporte de formato Parquet             |
| `numpy`      | 2.5.3   | Media ponderada y cuantiles ponderados |
| `matplotlib` | 3.11.2  | Generación de las figuras              |
| `jupyterlab` | 4.6.3   | Ejecución del notebook                 |

## 5. Arquitectura del proyecto

```
proyecto-endireh-violencia-/
├── config/
│   └── rutas.py                    # Rutas estáticas del proyecto
├── data/
│   ├── data-raw/                   # Archivo consolidado (excluido del repositorio)
│   ├── data-processed/             # Datos limpios: endireh_limpio.parquet y .csv
│   ├── data-input-model/           # Datos listos para modelado (vacía en esta etapa)
│   └── data-model/                 # Salidas del modelo (vacía en esta etapa)
├── notebooks/
│   └── eda_endireh.ipynb           # Análisis exploratorio y su explicación
├── reports/
│   └── figuras/                    # Las cinco figuras en PNG
├── src/
│   ├── cleaning/
│   │   └── limpieza.py             # Pipeline de preprocesamiento
│   ├── visualization/
│   │   ├── eda.py                  # Medidas de localización y variabilidad
│   │   └── graficas.py             # Construcción de las figuras
│   └── models/                     # Scripts de modelado (vacía en esta etapa)
├── .gitignore
├── README.md
└── requirements.txt
```

Las carpetas `data-input-model/`, `data-model/` y `src/models/` corresponden a
la etapa de modelado, y permanecen vacías: el Paso 2 de la práctica las incluye
en la estructura de directorios a crear. Cada una contiene un archivo `.gitkeep`
para que Git conserve la carpeta vacía.

Ningún script escribe rutas a mano: todas se derivan de `config/rutas.py`, que
resuelve la raíz del proyecto a partir de la ubicación del propio archivo.

## 6. Cómo ejecutar el pipeline

Con el entorno activo, desde la raíz del proyecto y con el archivo consolidado
ya colocado en `data/data-raw/`:

```bash
# 1. Preprocesamiento
python src/cleaning/limpieza.py
```

Lee el CSV crudo, repara el encoding, convierte los códigos de no respuesta a
nulo, ajusta los tipos, elimina duplicados, imputa donde procede y escribe
`data/data-processed/endireh_limpio.parquet` y su equivalente en CSV. Imprime un
registro de cuántas filas y valores modificó en cada paso.

```bash
# 2. Medidas descriptivas
python src/visualization/eda.py
```

Imprime, para cada variable, las medidas de localización en versión simple y
ponderada, y las de variabilidad comparando el grupo que reportó violencia de
pareja contra el que no.

```bash
# 3. Figuras
python src/visualization/graficas.py
```

Genera las cinco figuras en `reports/figuras/`.

```bash
# 4. Notebook
jupyter lab notebooks/eda_endireh.ipynb
```

Presenta los resultados anteriores con su explicación, la tabla de clasificación
de variables y las figuras. El notebook importa de los scripts de `src/` y no
recalcula nada, de modo que cada resultado tiene un único origen.

Los pasos 1 a 3 son dependientes en ese orden: el EDA y las figuras leen el
Parquet que produce la limpieza, y el notebook requiere que las figuras existan.

## 7. Salida del preprocesamiento

|                        | Archivo crudo | Procesado  |
| ---------------------- | ------------- | ---------- |
| Registros              | 110,127       | 105,752    |
| Columnas               | 24            | 25         |
| Población representada | 50,523,469    | 49,144,897 |

La columna adicional es `ingreso_pareja_imputado`, que marca los registros cuyo
ingreso fue imputado.

El reporte documenta el detalle de cada paso, las decisiones de imputación y las
limitaciones encontradas en tres variables del archivo consolidado.

## 8. Equipo

- Flores Juárez Luis Enrique
- García Morales Carlos Alan
- Rangel Saucedo Melanie Valeria

from pathlib import Path

# 1. Obtenemos la ruta del directorio raíz del proyecto.
# __file__: Ruta del archivo (config/rutas.py)
# .resolve(): Convierte la ruta en absoluta 
# .parent.parent: Sube dos niveles de config/rutas.py a la raíz del proyecto.
BASE_DIR = Path(__file__).resolve().parent.parent

# Ruta a carpeta de datos crudos (archivos CSV sin modificar de la fuente)
RUTA_DATA_RAW = BASE_DIR / "data" / "data-raw"
# Ruta a la carpeta de datos procesados (Parquet o CSV limpios y estandarizados)
RUTA_DATA_PROCESSED = BASE_DIR / "data" / "data-processed"
# Ruta a la carpeta de datos transformados (Para modelos de ML)
RUTA_DATA_INPUT_MODEL = BASE_DIR / "data" / "data-input-model"
# Ruta a la carpeta de salidas del modelo (predicciones, métricas, binarios)
RUTA_DATA_MODEL = BASE_DIR / "data" / "data-model"

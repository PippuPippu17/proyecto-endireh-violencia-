from pathlib import Path

# Ruta raíz del proyecto
BASE_DIR = Path(__file__).resolve().parent.parent

# Rutas de datos
RUTA_DATA_RAW = BASE_DIR / "data" / "data-raw"
RUTA_DATA_PROCESSED = BASE_DIR / "data" / "data-processed"
RUTA_DATA_INPUT_MODEL = BASE_DIR / "data" / "data-input-model"
RUTA_DATA_MODEL = BASE_DIR / "data" / "data-model"

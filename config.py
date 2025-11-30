# config.py
import os
from pathlib import Path

# Пути
BASE_DIR = Path(__file__).parent
CACHE_DIR = BASE_DIR / "cache"
FONTS_DIR = BASE_DIR / "fonts"
TEMP_DIR = Path("/tmp/clustering_bot")

# Создаём директории
CACHE_DIR.mkdir(exist_ok=True)
TEMP_DIR.mkdir(exist_ok=True)

# Лимиты
MAX_PDF_SIZE_MB = 10
MAX_CACHE_AGE_SECONDS = 3600  # 1 час
MAX_CACHE_ITEMS = 100

# Шрифт
FONT_PATH = FONTS_DIR / "DejaVuSans.ttf"

# API
YANDEX_API_KEY = os.getenv('YANDEX_API_KEY')
YANDEX_FOLDER_ID = os.getenv('YANDEX_FOLDER_ID')

# Настройки дефолтной эмбединг-модели
DEFAULT_EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # Текущая (fallback)

# Настройка эмбединг модели через .env
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", 
    DEFAULT_EMBEDDING_MODEL
)

# === Admin ===
ADMIN_TELEGRAM_ID = os.getenv("ADMIN_TELEGRAM_ID")

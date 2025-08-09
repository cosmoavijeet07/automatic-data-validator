"""
Configuration module for the Data Cleaning Application
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
TEMP_DIR = BASE_DIR / "temp"
LOGS_DIR = BASE_DIR / "logs"
OUTPUT_DIR = BASE_DIR / "output"

# Create directories if they don't exist
for dir_path in [DATA_DIR, TEMP_DIR, LOGS_DIR, OUTPUT_DIR]:
    dir_path.mkdir(exist_ok=True)

# API Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DEFAULT_MODEL = "gpt-4-turbo-preview"

# Available models
MODELS = {
    "GPT-4 Turbo": "gpt-4-turbo-preview",
    "GPT-4": "gpt-4",
    "GPT-3.5 Turbo": "gpt-3.5-turbo",
}

# Processing modes
PROCESSING_MODES = {
    "Manual Review": "manual",
    "Automatic": "automatic"
}

# Data type options for schema editing
DATA_TYPES = [
    "object",
    "int64",
    "float64",
    "bool",
    "datetime64",
    "category",
    "string"
]

# Date formats
DATE_FORMATS = [
    "%Y-%m-%d",
    "%d/%m/%Y",
    "%m/%d/%Y",
    "%Y/%m/%d",
    "%d-%m-%Y",
    "%m-%d-%Y",
    "%Y-%m-%d %H:%M:%S",
    "%d/%m/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M:%S",
]

# Max file size (in MB)
MAX_FILE_SIZE = 100

# Retry configuration
MAX_RETRIES = 5
RETRY_DELAY = 2  # seconds

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Session configuration
SESSION_TIMEOUT = 3600  # 1 hour in seconds

# Profiling configuration
PROFILE_CONFIG = {
    "minimal": False,
    "explorative": True,
    "sensitive": False,
    "dark_mode": False,
    "orange_mode": False
}

# Text processing configuration
TEXT_CHUNK_SIZE = 1000
TEXT_OVERLAP = 200
EMBEDDING_MODEL = "text-embedding-ada-002"
VECTOR_DB_COLLECTION = "text_data"

# Missing value patterns
MISSING_PATTERNS = [
    "nan", "NaN", "NAN", "n/a", "N/A", "NA", "na",
    "?", "??", "???", "", " ", "  ", "null", "NULL",
    "None", "none", "NONE", "-", "--", "---",
    "missing", "Missing", "MISSING", "unknown", "Unknown", "UNKNOWN"
]

# Quality check thresholds
QUALITY_THRESHOLDS = {
    "missing_threshold": 0.5,  # Flag columns with >50% missing
    "cardinality_threshold": 0.95,  # Flag high cardinality categorical
    "outlier_zscore": 3,  # Z-score for outlier detection
    "duplicate_threshold": 0.1  # Flag if >10% duplicates
}
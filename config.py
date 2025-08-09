"""
Configuration file for the AI-Powered Data Cleaning Platform
"""

import os
from pathlib import Path

# Application settings
APP_NAME = "AI-Powered Data Cleaning Platform"
VERSION = "1.0.0"

# File handling settings
SUPPORTED_FILE_TYPES = ['csv', 'xlsx', 'xls', 'json', 'txt']
MAX_FILE_SIZE_MB = 100
UPLOAD_DIR = Path("uploads")
TEMP_DIR = Path("temp")
LOGS_DIR = Path("logs")
OUTPUTS_DIR = Path("outputs")

# Create directories if they don't exist
for directory in [UPLOAD_DIR, TEMP_DIR, LOGS_DIR, OUTPUTS_DIR]:
    directory.mkdir(exist_ok=True)

# LLM Model configurations
MODELS = {
    "GPT 4.1": "gpt-4.1-2025-04-14",  # default
    "GPT O4 Mini": "o4-mini-2025-04-16"
}

# OpenAI API settings (you'll need to set these environment variables)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-api-key-here")
# OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")

# Data processing settings
DEFAULT_SAMPLE_SIZE = 1000  # For large datasets, sample this many rows for analysis
MAX_RETRIES = 5  # Maximum retry attempts for LLM calls
RETRY_DELAY = 2  # Seconds to wait between retries

# Schema detection settings
CATEGORICAL_THRESHOLD = 0.05  # If unique values / total values < this, treat as categorical
DATE_FORMATS = [
    '%Y-%m-%d',
    '%m/%d/%Y',
    '%d-%m-%Y',
    '%Y-%m-%d %H:%M:%S',
    '%m/%d/%Y %H:%M:%S',
    '%d/%m/%Y %H:%M:%S',
    '%Y-%m-%dT%H:%M:%S',
    '%Y-%m-%dT%H:%M:%SZ'
]

# Data quality thresholds
MISSING_VALUE_THRESHOLD = 0.5  # Flag columns with > 50% missing values
OUTLIER_THRESHOLD = 3  # Standard deviations for outlier detection
DUPLICATE_THRESHOLD = 0.1  # Flag if > 10% duplicates

# Text processing settings
VECTOR_DB_CHUNK_SIZE = 1000
VECTOR_DB_OVERLAP = 200
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Streamlit settings
STREAMLIT_CONFIG = {
    "page_title": APP_NAME,
    "page_icon": "🧹",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Logging settings
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_MAX_BYTES = 10 * 1024 * 1024  # 10MB
LOG_BACKUP_COUNT = 5

# Session settings
SESSION_TIMEOUT_HOURS = 24
MAX_SESSIONS = 100

# Data profiling libraries to try (in order of preference)
PROFILING_LIBRARIES = [
    'ydata_profiling',
    'pandas_profiling', 
    'sweetviz'
]

# Code execution safety settings
ALLOWED_IMPORTS = [
    'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 
    'sklearn', 'scipy', 'datetime', 'json', 're', 'math',
    'warnings', 'collections', 'itertools', 'functools'
]

BLOCKED_FUNCTIONS = [
    'eval', 'exec', 'compile', '__import__', 'open', 
    'input', 'raw_input', 'file', 'reload', 'vars', 
    'globals', 'locals', 'dir', 'help'
]

# Default prompts for LLM interactions
SCHEMA_DETECTION_PROMPT = """
You are a data science expert. Analyze the provided dataset sample and detect the appropriate schema.
For each column, determine:
1. Data type (int64, float64, object, datetime64[ns], bool, category)
2. Whether it should be categorical
3. Appropriate date format if applicable
4. Any data quality issues

Return your analysis as a JSON object with column names as keys and schema information as values.
"""

DATA_ANALYSIS_PROMPT = """
You are a data quality expert. Analyze the provided dataset and quality report.
Identify:
1. Data quality issues not caught by basic profiling
2. Potential data inconsistencies
3. Recommended cleaning strategies
4. Prioritized list of issues to address

Focus on practical, actionable insights that can be implemented programmatically.
"""

CORRECTION_CODE_PROMPT = """
You are a data cleaning expert. Generate Python code using pandas to clean the provided dataset.
Based on the schema, quality report, and user instructions, create code that:
1. Handles missing values appropriately
2. Removes or fixes outliers
3. Standardizes data formats
4. Removes duplicates
5. Fixes data type issues

Return clean, well-commented Python code that can be executed safely.
Only use standard libraries like pandas, numpy, datetime, and re.
"""

STRATEGY_SUMMARY_PROMPT = """
You are a data analyst. Summarize the data cleaning strategy in simple terms.
Explain what the cleaning code will do in less than 100 words.
Focus on the main cleaning steps and their business impact.
"""

PIPELINE_GENERATION_PROMPT = """
You are a software engineer. Create a complete Python pipeline script that:
1. Accepts a dataset as input
2. Applies all the cleaning steps that were executed
3. Returns a cleaned dataset
4. Includes proper error handling and logging
5. Is modular and reusable

The script should be self-contained and production-ready.
"""

# Error messages
ERROR_MESSAGES = {
    'file_too_large': f'File size exceeds {MAX_FILE_SIZE_MB}MB limit',
    'unsupported_format': f'Unsupported file format. Supported: {", ".join(SUPPORTED_FILE_TYPES)}',
    'api_key_missing': 'OpenAI API key not found. Please set OPENAI_API_KEY environment variable',
    'model_not_found': 'Selected model not available',
    'processing_error': 'Error processing your request. Please try again',
    'execution_error': 'Error executing generated code. Please review and modify',
    'session_expired': 'Your session has expired. Please start over',
}

# Success messages
SUCCESS_MESSAGES = {
    'file_uploaded': 'File uploaded and processed successfully',
    'schema_detected': 'Schema detected successfully',
    'analysis_completed': 'Data analysis completed',
    'correction_applied': 'Data corrections applied successfully',
    'pipeline_generated': 'Pipeline generated successfully'
}
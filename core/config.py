import os
from dataclasses import dataclass

@dataclass
class ModelConfig:
    name: str
    provider: str
    id: str
    default: bool = False

MODELS = {
    "GPT 4.1": ModelConfig(name="GPT 4.1", provider="openai", id="gpt-4.1-2025-04-14", default=True),
    "GPT O4 Mini": ModelConfig(name="GPT O4 Mini", provider="openai", id="o4-mini-2025-04-16", default=False),
}



DATA_DIR = os.environ.get("APP_DATA_DIR", "outputs")
LOG_DIR = os.path.join(DATA_DIR, "logs")
PIPELINE_DIR = os.path.join(DATA_DIR, "pipeline")
CLEANED_DIR = os.path.join(DATA_DIR, "cleaned")
STATE_DIR = os.path.join(DATA_DIR, "state")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(PIPELINE_DIR, exist_ok=True)
os.makedirs(CLEANED_DIR, exist_ok=True)
os.makedirs(STATE_DIR, exist_ok=True)

MAX_RETRIES_PER_STEP = 5
SUMMARY_WORD_LIMIT = 100

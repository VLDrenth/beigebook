import os

from pathlib import Path
from dotenv import load_dotenv

class Config:
    load_dotenv()

    PROJECT_ROOT = Path(__file__).parent.parent.parent  # Go up two levels
    DATA_DIR = PROJECT_ROOT / "out"
    SCRAPED_TEXT_DIR = DATA_DIR / "txt"
    SENTIMENT_OUTPUT_DIR = DATA_DIR / "sentiment_results"
    PLOTS_OUTPUT_DIR = DATA_DIR / "plots"
    EXTERNAL_PATH = DATA_DIR / "external_data"


    # Run settings
    END_YEAR = 2024
    SCORER_TYPE = "huggingface"
    HORIZONS = [1, 2, 3, 4, 5, 6]

    # API configurations
    HF_TOKEN = os.getenv("HF_TOKEN")
    OPENAI_TOKEN = os.getenv("OPENAI_TOKEN")
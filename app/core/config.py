from pydantic import BaseModel
from pathlib import Path
from typing import Optional

class Settings(BaseModel):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Recipe Recommendation API"
    
    # Paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    CACHE_DIR: Path = BASE_DIR / "cache"
    LOG_DIR: Path = BASE_DIR / "logs"
    
    # Data file
    RECIPE_DATA_FILE: Path = DATA_DIR / "relevant_columns_recipes_cleaned.csv"
    
    # Cache files
    BERT_CACHE_DIR: Path = CACHE_DIR / "bert"
    TFIDF_CACHE_DIR: Path = CACHE_DIR / "tfidf"
    
    # Model settings
    MAX_RECOMMENDATIONS: int = 20
    BERT_MODEL_NAME: str = "bert-base-uncased"
    TFIDF_MAX_FEATURES: int = 10000

    # Cache settings
    CACHE_MAX_SIZE: int = 1000
    CACHE_TTL: int = 604800  # Time to live in seconds i.e 7 days
    
    class Config:
        arbitrary_types_allowed = True

settings = Settings()
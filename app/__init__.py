# app/__init__.py
from pathlib import Path

# Create necessary directories
def create_dirs():
    dirs = [
        Path("data"),
        Path("cache/bert"),
        Path("cache/tfidf"),
        Path("logs")
    ]
    
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

create_dirs()
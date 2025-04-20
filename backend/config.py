import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base directory of the project
BASE_DIR = Path(__file__).resolve().parent.parent

# API settings
API_SETTINGS = {
    "title": "DataPrep Pro API",
    "description": "A professional data preprocessing API for machine learning workflows",
    "version": "0.1.0",
    "docs_url": "/docs",
    "redoc_url": "/redoc",
}

# Server settings
SERVER_SETTINGS = {
    "host": os.getenv("API_HOST", "0.0.0.0"),
    "port": int(os.getenv("API_PORT", "8000")),
    "reload": os.getenv("API_RELOAD", "True").lower() == "true",
    "workers": int(os.getenv("API_WORKERS", "1")),
}

# CORS settings
CORS_SETTINGS = {
    "allow_origins": os.getenv("CORS_ORIGINS", "*").split(","),
    "allow_credentials": True,
    "allow_methods": ["*"],
    "allow_headers": ["*"],
}

# Logging settings
LOGGING_SETTINGS = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "default": {
            "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "default",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": BASE_DIR / "logs" / "app.log",
            "formatter": "default",
            "level": "INFO",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}

# File upload settings
UPLOAD_SETTINGS = {
    "max_file_size": int(os.getenv("MAX_FILE_SIZE", "10485760")),  # 10MB
    "allowed_extensions": {
        "tabular": [".csv"],
        "text": [".txt"],
        "image": [".png", ".jpg", ".jpeg"],
    },
}

# Processing settings
PROCESSING_SETTINGS = {
    "tabular": {
        "max_rows": int(os.getenv("MAX_ROWS", "1000000")),
        "max_columns": int(os.getenv("MAX_COLUMNS", "1000")),
        "sample_size": int(os.getenv("SAMPLE_SIZE", "5")),
    },
    "text": {
        "max_length": int(os.getenv("MAX_TEXT_LENGTH", "1000000")),
        "default_language": "english",
    },
    "image": {
        "max_dimension": int(os.getenv("MAX_IMAGE_DIMENSION", "4096")),
        "default_format": "PNG",
        "compression_quality": int(os.getenv("IMAGE_QUALITY", "85")),
    },
}

def get_settings() -> Dict[str, Any]:
    """Get all settings as a dictionary."""
    return {
        "api": API_SETTINGS,
        "server": SERVER_SETTINGS,
        "cors": CORS_SETTINGS,
        "logging": LOGGING_SETTINGS,
        "upload": UPLOAD_SETTINGS,
        "processing": PROCESSING_SETTINGS,
    } 
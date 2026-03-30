# ⚡ DataPrep Pro

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A professional data preprocessing tool for machine learning workflows. Clean, transform, and analyze tabular, text, and image data through a beautiful web interface backed by a FastAPI backend.

---

## Features

### 📊 Tabular Data
- **Missing Values**: Drop, mean, or median imputation
- **Deduplication**: Remove duplicate rows
- **Feature Scaling**: Standard, MinMax, or Robust scaling
- **Categorical Encoding**: Label or One-Hot encoding
- **Outlier Detection**: IQR and Z-score methods
- **Visualizations**: Distributions, correlation matrix, missing value heatmap

### 📝 Text Data
- **NLP Pipeline**: Tokenization, stop word removal, lemmatization
- **Cleaning**: URL removal, email removal, whitespace normalization
- **Language Detection**: Automatic detection with English verification
- **Analysis**: Word frequencies, POS tagging, vocabulary richness
- **Stateful Sessions**: Process once, query frequencies and POS tags via API

### 🖼️ Image Data
- **Transformations**: Resize, grayscale, normalize
- **Adjustments**: Brightness and contrast control
- **Format Conversion**: PNG, JPEG, WEBP
- **Analysis**: Per-channel (R/G/B) mean pixel values

---

## Tech Stack

| Technology | Purpose |
|---|---|
| **FastAPI** | REST API backend |
| **Streamlit** | Interactive web UI |
| **Pydantic v2** | Request/response validation |
| **scikit-learn** | Tabular preprocessing |
| **NLTK** | NLP text processing |
| **langdetect** | Automatic language detection |
| **OpenCV + Pillow** | Image transformations |
| **Plotly** | Interactive visualizations |
| **structlog** | Structured logging |
| **httpx** | Frontend → API HTTP client |

---

## Getting Started

### Prerequisites

- Python 3.11+

### Installation

```bash
# Clone the repository
git clone https://github.com/vismaytk/preproc.git
cd preproc

# Create and activate virtual environment
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/macOS
source venv/bin/activate

# Install dependencies (with dev tools)
pip install -e ".[dev]"

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

# Set up environment
cp .env.example .env
```

### Running Locally

Open two terminals:

```bash
# Terminal 1 — Start the API (port 8000)
python -m uvicorn api.main:app --reload

# Terminal 2 — Start the UI (port 8501)
python -m streamlit run app/Home.py
```

- **UI**: http://localhost:8501
- **API Docs (Swagger)**: http://localhost:8000/docs
- **API Docs (ReDoc)**: http://localhost:8000/redoc

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `POST` | `/tabular/process` | Process CSV data |
| `POST` | `/text/process` | Process text data |
| `GET` | `/text/frequencies/{session_id}` | Word frequencies for a session |
| `GET` | `/text/pos-tags/{session_id}` | POS tags for a session |
| `POST` | `/image/process` | Process image data |

---

## Project Structure

```
dataprep-pro/
├── api/                        ← FastAPI backend
│   ├── core/
│   │   ├── config.py           ← pydantic-settings configuration
│   │   ├── logging.py          ← structlog setup
│   │   └── exceptions.py       ← custom HTTP exceptions
│   ├── processors/
│   │   ├── base.py             ← abstract BaseProcessor
│   │   ├── tabular.py          ← CSV/DataFrame processing
│   │   ├── text.py             ← NLP text processing
│   │   └── image.py            ← image transformations
│   ├── routes/
│   │   ├── tabular.py          ← /tabular/* endpoints
│   │   ├── text.py             ← /text/* endpoints
│   │   └── image.py            ← /image/* endpoints
│   ├── schemas/
│   │   ├── tabular.py          ← Pydantic v2 response models
│   │   ├── text.py
│   │   └── image.py
│   ├── middleware/
│   │   └── logging.py          ← request logging middleware
│   └── main.py                 ← app factory
├── app/                        ← Streamlit frontend
│   ├── pages/
│   │   ├── 1_tabular.py
│   │   ├── 2_text.py
│   │   └── 3_image.py
│   ├── components/
│   │   ├── file_uploader.py
│   │   ├── stats_panel.py
│   │   └── download_button.py
│   ├── utils/
│   │   ├── api_client.py       ← centralized httpx client
│   │   └── charts.py           ← Plotly chart helpers
│   └── Home.py                 ← landing page
├── tests/
│   ├── unit/
│   │   ├── test_tabular.py
│   │   ├── test_text.py
│   │   └── test_image.py
│   ├── integration/
│   │   └── test_api.py
│   └── conftest.py             ← shared fixtures
├── samples/                    ← sample data for demo
│   ├── sample.csv
│   ├── sample.txt
│   └── sample.png
├── .streamlit/
│   └── config.toml             ← Streamlit theme config
├── .env.example                ← environment variable template
├── .gitignore
├── LICENSE
├── pyproject.toml              ← dependencies + tool config
└── README.md
```

---

## Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=api --cov-report=term-missing

# Lint
python -m ruff check api/ app/
```

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/vismaytk">Vismay</a>
</p>
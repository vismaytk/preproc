# ⚡ DataPrep Pro

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![CI](https://github.com/vismaytk/preproc/actions/workflows/ci.yml/badge.svg)](https://github.com/vismaytk/preproc/actions/workflows/ci.yml)

> A professional SaaS data preprocessing tool for machine learning workflows. Clean, transform, and analyze tabular, text, and image data through a beautiful web interface backed by a production-grade API.

<!-- screenshot here -->

## What It Does

DataPrep Pro is a full-stack data preprocessing platform that automates the tedious data cleaning work that machine learning engineers deal with daily. Upload your raw data, configure the preprocessing pipeline, preview the results with interactive visualizations, and download the cleaned output — all in seconds, no code required.

---

## Tech Stack

| Technology | Purpose | Version |
|---|---|---|
| **FastAPI** | REST API backend | 0.109+ |
| **Streamlit** | Interactive web UI | 1.31+ |
| **Pydantic v2** | Request/response validation | 2.4+ |
| **scikit-learn** | Tabular preprocessing (scaling, imputation) | 1.4+ |
| **NLTK** | NLP text processing pipeline | 3.8+ |
| **langdetect** | Automatic language detection | 1.0+ |
| **OpenCV + Pillow** | Image transformations | 4.9+ / 10.2+ |
| **Plotly** | Interactive data visualizations | 5.18+ |
| **structlog** | Structured JSON logging | 24.1+ |
| **slowapi** | API rate limiting | 0.1+ |
| **httpx** | Async HTTP client | 0.27+ |
| **Docker** | Containerized deployment | - |
| **GitHub Actions** | CI/CD pipeline | - |

---

## Getting Started

### Prerequisites

- Python 3.11+
- pip or uv

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

# Install with dev dependencies
pip install -e ".[dev]"

# Download NLTK data
python -c "import nltk; nltk.download('punkt'); nltk.download('punkt_tab'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('averaged_perceptron_tagger'); nltk.download('averaged_perceptron_tagger_eng')"

# Set up environment
cp .env.example .env
```

### Running Locally

```bash
# Option 1: Start both services with the convenience script
bash scripts/start.sh

# Option 2: Start individually
uvicorn api.main:app --reload          # API on :8000
streamlit run app/Home.py              # UI on :8501
```

### Docker

```bash
docker compose up --build
```

Access: **UI** → http://localhost:8501 | **API Docs** → http://localhost:8000/docs

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | API info |
| `GET` | `/health` | Health check |
| `POST` | `/tabular/process` | Process CSV data (missing values, scaling, dedup) |
| `POST` | `/text/process` | Process text data (NLP pipeline) |
| `GET` | `/text/frequencies/{session_id}` | Get word frequencies for a session |
| `GET` | `/text/pos-tags/{session_id}` | Get POS tags for a session |
| `POST` | `/image/process` | Process image data (resize, normalize, convert) |

Full interactive docs available at `/docs` (Swagger) and `/redoc` (ReDoc).

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
- **Language Detection**: Automatic with English detection warning
- **Analysis**: Word frequencies, POS tagging, vocabulary richness
- **Stateful Sessions**: Process once, query frequencies and POS tags via API

### 🖼️ Image Data
- **Transformations**: Resize, grayscale, normalize
- **Adjustments**: Brightness and contrast control
- **Format Conversion**: PNG, JPEG, WEBP
- **Analysis**: Per-channel (R/G/B) mean pixel values

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
│   │   └── charts.py           ← Plotly visualization helpers
│   └── Home.py                 ← landing page
├── tests/
│   ├── unit/
│   │   ├── test_tabular.py
│   │   ├── test_text.py
│   │   └── test_image.py
│   ├── integration/
│   │   └── test_api.py
│   └── conftest.py
├── samples/                    ← sample data files
├── scripts/
│   └── start.sh
├── .github/workflows/
│   └── ci.yml
├── pyproject.toml
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=api --cov-report=term-missing

# Lint
ruff check api/ app/

# Type check
mypy api/
```

---

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Write tests for your changes
4. Ensure `pytest` and `ruff check` pass
5. Commit your changes (`git commit -m 'Add amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  Built with ❤️ by <a href="https://github.com/vismaytk">Vismay TK</a>
</p>
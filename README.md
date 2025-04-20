# Data Preprocessing Tool 🔧

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28.1-red.svg)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> A powerful SaaS tool for preprocessing datasets in machine learning and AI applications. Built with FastAPI and Streamlit, it provides an intuitive interface for handling tabular, text, and image data preprocessing tasks.

## ✨ Features

### 📊 Tabular Data Processing
- **Missing Value Handling**
  - Strategy options: mean imputation, row dropping
  - Column-specific handling
  - Validation of imputed values
- **Feature Scaling**
  - Standard scaling (z-score normalization)
  - Min-max scaling
  - Robust scaling
- **Data Cleaning**
  - Duplicate row removal
  - Column dropping
  - Data type validation
- **Statistics & Analysis**
  - Detailed dataset statistics
  - Missing value analysis
  - Column type detection

### 📝 Text Processing
- **Text Cleaning**
  - Case normalization
  - Punctuation removal
  - Number removal
  - Whitespace normalization
- **NLP Operations**
  - Stopword removal
  - Text lemmatization
  - Word frequency analysis
  - Part-of-speech tagging
- **Analysis**
  - Text statistics
  - Word count metrics
  - Sentence structure analysis

### 🖼️ Image Processing
- **Image Handling**
  - Multiple format support (JPG, PNG)
  - Base64 encoding/decoding
  - Size validation
- **Processing Features**
  - Image transformations
  - Format conversion
  - Preview generation
- **Output Options**
  - Processed image download
  - Multiple export formats
  - Quality control

## 🚀 Quick Start

### Prerequisites
- Python 3.9 or higher
- pip package manager
- Virtual environment (recommended)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/vismaytk/preproc.git
cd preproc
```

2. Create and activate virtual environment:
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Unix/MacOS
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configurations
```

### Running the Application

1. Start the backend server:
```bash
uvicorn backend.main:app --reload
```

2. Start the frontend application:
```bash
streamlit run frontend/app.py
```

3. Access the application:
- Frontend UI: http://localhost:8501
- API Documentation: http://localhost:8000/docs

## 💻 Usage

### Web Interface

1. **Upload Data**
   - Support for CSV, text files, and images
   - Drag-and-drop functionality
   - File validation

2. **Configure Processing**
   - Select preprocessing options
   - Set parameters
   - Preview changes

3. **Process & Download**
   - Execute preprocessing
   - Preview results
   - Download processed data

### API Endpoints

#### Tabular Data
```
POST /tabular/process
GET  /tabular/statistics
```

#### Text Processing
```
POST /text/process
GET  /text/statistics
GET  /text/frequencies
GET  /text/pos-tags
```

#### Image Processing
```
POST /image/process
GET  /image/info
```

## 🛠️ Development

### Project Structure
```
preproc/
├── backend/
│   ├── processors/
│   │   ├── tabular.py   # Tabular data processing
│   │   ├── text.py      # Text processing
│   │   └── image.py     # Image processing
│   └── main.py          # FastAPI application
├── frontend/
│   └── app.py           # Streamlit interface
├── tests/               # Test suite
├── logs/               # Application logs
├── processed_data/     # Processed output
└── uploads/           # Temporary uploads
```

### Running Tests
```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=backend
```

## 🔒 Security

- Input validation for all file uploads
- Size limits for uploads
- API key authentication
- CORS protection
- Secure file handling

## 📝 Configuration

Key configuration options in `.env`:

| Variable | Description | Default |
|----------|-------------|---------|
| MAX_UPLOAD_SIZE | Maximum file size (MB) | 10 |
| MAX_ROWS | Maximum rows in CSV | 1000000 |
| MAX_TEXT_LENGTH | Maximum text length | 1000000 |
| MAX_IMAGE_DIMENSION | Maximum image dimension | 4096 |

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with modern Python tools and libraries
- Inspired by real-world preprocessing needs
- Community contributions welcome 
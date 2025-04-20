# Data Preprocessing Tool

A powerful SaaS tool for preprocessing datasets for machine learning and AI applications. This tool provides both a FastAPI backend and a Streamlit frontend for easy data preprocessing tasks.

## Features

### Tabular Data Processing
- Load and validate CSV files
- Handle missing values (strategies: mean, drop)
- Remove duplicate rows
- Scale numeric features (methods: standard, minmax, robust)
- Drop specified columns
- Get detailed dataset statistics
- Export processed data as CSV

### Text Data Processing
- Load and process text data
- Convert text to lowercase
- Remove punctuation
- Remove numbers
- Remove extra whitespace
- Remove stopwords
- Lemmatize text
- Get word frequencies
- Get part-of-speech tags
- Get text statistics
- Export processed text

### Image Processing
- Load and validate image files
- Process images with various transformations
- Support for base64-encoded images
- Preview processed images
- Download processed images

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd preproc
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Application

1. Start the backend server:
```bash
uvicorn backend.main:app --reload
```

2. Start the frontend application:
```bash
streamlit run frontend/app.py
```

## Usage

### Web Interface
1. Open your browser and navigate to `http://localhost:8501`
2. Upload your data file (CSV, text, or image)
3. Select preprocessing options
4. View and download the processed data

### API Endpoints

#### Tabular Data
- `POST /tabular/process`: Process CSV data
- `GET /tabular/statistics`: Get dataset statistics

#### Text Data
- `POST /text/process`: Process text data
- `GET /text/statistics`: Get text statistics
- `GET /text/frequencies`: Get word frequencies
- `GET /text/pos-tags`: Get part-of-speech tags

#### Image Data
- `POST /image/process`: Process image data
- `GET /image/info`: Get image information

## Development

### Running Tests
```bash
python -m pytest tests/ -v
```

### Project Structure
```
preproc/
├── backend/
│   ├── processors/
│   │   ├── tabular.py
│   │   ├── text.py
│   │   └── image.py
│   └── main.py
├── frontend/
│   └── app.py
├── tests/
│   ├── test_tabular.py
│   ├── test_text.py
│   └── test_image.py
└── requirements.txt
```

## Error Handling
- Comprehensive error handling for all processing operations
- Detailed error messages and logging
- Validation of input data and parameters

## Data Validation
- CSV file format validation
- Text encoding validation
- Image format and size validation
- Parameter validation for all operations

## Logging
- Detailed logging of all operations
- Error tracking and debugging information
- Processing step tracking

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License
This project is licensed under the MIT License - see the LICENSE file for details. 
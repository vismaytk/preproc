# DataPrep Pro 🔧

A professional data preprocessing tool for machine learning and data science workflows.

## Features 🌟

- **Tabular Data Processing**
  - Handle missing values (drop, mean, median)
  - Remove duplicates
  - Scale numeric features (StandardScaler, MinMaxScaler)
  - Column selection and dropping
  - Data type conversion
  - Descriptive statistics
  - Export processed data

- **Text Data Processing**
  - Tokenization
  - Stop words removal
  - Lemmatization
  - Case normalization
  - Special character handling
  - Text statistics
  - Export processed text

- **Image Processing**
  - Resize with aspect ratio preservation
  - Normalization
  - Grayscale conversion
  - Format conversion
  - Basic augmentation
  - Image statistics
  - Export processed images

## Installation 🚀

1. Clone the repository:
```bash
git clone https://github.com/yourusername/dataprep-pro.git
cd dataprep-pro
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage 💻

1. Start the backend server:
```bash
uvicorn backend.main:app --reload
```

2. Start the frontend application:
```bash
streamlit run frontend/app.py
```

3. Open your browser and navigate to:
- Frontend: http://localhost:8501
- API Documentation: http://localhost:8000/docs

## Project Structure 📁

```
dataprep-pro/
├── backend/
│   ├── __init__.py
│   ├── main.py
│   ├── processors/
│   │   ├── __init__.py
│   │   ├── tabular.py
│   │   ├── text.py
│   │   └── image.py
│   └── utils/
│       ├── __init__.py
│       └── validators.py
├── frontend/
│   ├── app.py
│   ├── components/
│   │   ├── __init__.py
│   │   ├── tabular.py
│   │   ├── text.py
│   │   └── image.py
│   └── utils/
│       ├── __init__.py
│       └── helpers.py
├── tests/
│   ├── __init__.py
│   ├── test_tabular.py
│   ├── test_text.py
│   └── test_image.py
├── requirements.txt
├── setup.py
├── .gitignore
└── README.md
```

## Dependencies 📦

- FastAPI
- Streamlit
- Pandas
- NumPy
- Pillow
- OpenCV
- NLTK
- scikit-learn

## Contributing 🤝

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License 📄

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments 🙏

- Thanks to all contributors
- Inspired by the need for a streamlined data preprocessing workflow
- Built with modern Python tools and libraries

## Contact 📧

Your Name - [@yourtwitter](https://twitter.com/yourtwitter) - email@example.com

Project Link: [https://github.com/yourusername/dataprep-pro](https://github.com/yourusername/dataprep-pro) 
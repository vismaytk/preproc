import os

def write_file(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', encoding='utf-8', newline='\n') as f:
        f.write(content)

# Write tabular.py
tabular_content = '''import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class TabularProcessor:
    """
    A class for processing tabular data with various preprocessing operations.
    """
    
    def __init__(self):
        self.original_df = None
        self.processed_df = None
        self.preprocessing_steps = []
        
    def load_data(self, data: bytes) -> None:
        """Load data from bytes into pandas DataFrame."""
        try:
            self.original_df = pd.read_csv(pd.io.common.BytesIO(data))
            self.processed_df = self.original_df.copy()
            logger.info(f"Data loaded successfully with shape {self.original_df.shape}")
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise ValueError(f"Error loading data: {str(e)}")
    
    def handle_missing_values(self, strategy: str = "drop") -> None:
        """Handle missing values in the dataset."""
        try:
            if strategy == "drop":
                self.processed_df = self.processed_df.dropna()
                self.preprocessing_steps.append(("handle_missing", "drop"))
            else:
                numeric_cols = self.processed_df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    imputer = SimpleImputer(strategy=strategy)
                    self.processed_df[numeric_cols] = imputer.fit_transform(self.processed_df[numeric_cols])
                    self.preprocessing_steps.append(("handle_missing", strategy))
            logger.info(f"Handled missing values using strategy: {strategy}")
        except Exception as e:
            logger.error(f"Error handling missing values: {str(e)}")
            raise ValueError(f"Error handling missing values: {str(e)}")
    
    def remove_duplicates(self) -> None:
        """Remove duplicate rows from the dataset."""
        try:
            original_shape = self.processed_df.shape
            self.processed_df = self.processed_df.drop_duplicates()
            self.preprocessing_steps.append(("remove_duplicates", None))
            logger.info(f"Removed {original_shape[0] - self.processed_df.shape[0]} duplicate rows")
        except Exception as e:
            logger.error(f"Error removing duplicates: {str(e)}")
            raise ValueError(f"Error removing duplicates: {str(e)}")
    
    def scale_features(self, method: str = "standard", columns: Optional[List[str]] = None) -> None:
        """Scale numeric features using specified method."""
        try:
            if columns is None:
                columns = self.processed_df.select_dtypes(include=[np.number]).columns
            
            if len(columns) == 0:
                logger.warning("No numeric columns found for scaling")
                return
            
            scaler = None
            if method == "standard":
                scaler = StandardScaler()
            elif method == "minmax":
                scaler = MinMaxScaler()
            elif method == "robust":
                scaler = RobustScaler()
            else:
                raise ValueError(f"Unknown scaling method: {method}")
            
            self.processed_df[columns] = scaler.fit_transform(self.processed_df[columns])
            self.preprocessing_steps.append(("scale_features", {"method": method, "columns": list(columns)}))
            logger.info(f"Scaled features using {method} scaling")
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise ValueError(f"Error scaling features: {str(e)}")
    
    def drop_columns(self, columns: List[str]) -> None:
        """Drop specified columns from the dataset."""
        try:
            self.processed_df = self.processed_df.drop(columns=columns, errors='ignore')
            self.preprocessing_steps.append(("drop_columns", columns))
            logger.info(f"Dropped columns: {columns}")
        except Exception as e:
            logger.error(f"Error dropping columns: {str(e)}")
            raise ValueError(f"Error dropping columns: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical information about the dataset."""
        try:
            stats = {
                "original_shape": self.original_df.shape,
                "processed_shape": self.processed_df.shape,
                "missing_values": self.processed_df.isnull().sum().to_dict(),
                "numeric_columns": list(self.processed_df.select_dtypes(include=[np.number]).columns),
                "categorical_columns": list(self.processed_df.select_dtypes(include=['object']).columns),
                "preprocessing_steps": self.preprocessing_steps
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            raise ValueError(f"Error getting statistics: {str(e)}")
    
    def get_processed_data(self) -> bytes:
        """Get processed data as CSV bytes."""
        try:
            return self.processed_df.to_csv(index=False).encode()
        except Exception as e:
            logger.error(f"Error getting processed data: {str(e)}")
            raise ValueError(f"Error getting processed data: {str(e)}")'''

# Write text.py
text_content = '''import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Dict, Any, List, Optional
import logging
import re
import string

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

logger = logging.getLogger(__name__)

class TextProcessor:
    """
    A class for processing text data with various preprocessing operations.
    """
    
    def __init__(self):
        self.original_text = None
        self.processed_text = None
        self.preprocessing_steps = []
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def load_data(self, data: bytes) -> None:
        """Load text data from bytes."""
        try:
            self.original_text = data.decode('utf-8')
            self.processed_text = self.original_text
            logger.info(f"Text loaded successfully with length {len(self.original_text)}")
        except Exception as e:
            logger.error(f"Error loading text: {str(e)}")
            raise ValueError(f"Error loading text: {str(e)}")
    
    def to_lowercase(self) -> None:
        """Convert text to lowercase."""
        try:
            self.processed_text = self.processed_text.lower()
            self.preprocessing_steps.append(("to_lowercase", None))
            logger.info("Converted text to lowercase")
        except Exception as e:
            logger.error(f"Error converting to lowercase: {str(e)}")
            raise ValueError(f"Error converting to lowercase: {str(e)}")
    
    def remove_punctuation(self) -> None:
        """Remove punctuation from text."""
        try:
            self.processed_text = self.processed_text.translate(
                str.maketrans("", "", string.punctuation)
            )
            self.preprocessing_steps.append(("remove_punctuation", None))
            logger.info("Removed punctuation")
        except Exception as e:
            logger.error(f"Error removing punctuation: {str(e)}")
            raise ValueError(f"Error removing punctuation: {str(e)}")
    
    def remove_numbers(self) -> None:
        """Remove numbers from text."""
        try:
            self.processed_text = re.sub(r'\\d+', '', self.processed_text)
            self.preprocessing_steps.append(("remove_numbers", None))
            logger.info("Removed numbers")
        except Exception as e:
            logger.error(f"Error removing numbers: {str(e)}")
            raise ValueError(f"Error removing numbers: {str(e)}")
    
    def remove_whitespace(self) -> None:
        """Remove extra whitespace from text."""
        try:
            self.processed_text = " ".join(self.processed_text.split())
            self.preprocessing_steps.append(("remove_whitespace", None))
            logger.info("Removed extra whitespace")
        except Exception as e:
            logger.error(f"Error removing whitespace: {str(e)}")
            raise ValueError(f"Error removing whitespace: {str(e)}")
    
    def remove_stopwords(self) -> None:
        """Remove stop words from text."""
        try:
            words = word_tokenize(self.processed_text)
            words = [w for w in words if w.lower() not in self.stop_words]
            self.processed_text = " ".join(words)
            self.preprocessing_steps.append(("remove_stopwords", None))
            logger.info("Removed stop words")
        except Exception as e:
            logger.error(f"Error removing stop words: {str(e)}")
            raise ValueError(f"Error removing stop words: {str(e)}")
    
    def lemmatize_text(self) -> None:
        """Lemmatize text."""
        try:
            words = word_tokenize(self.processed_text)
            words = [self.lemmatizer.lemmatize(w) for w in words]
            self.processed_text = " ".join(words)
            self.preprocessing_steps.append(("lemmatize", None))
            logger.info("Lemmatized text")
        except Exception as e:
            logger.error(f"Error lemmatizing text: {str(e)}")
            raise ValueError(f"Error lemmatizing text: {str(e)}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistical information about the text."""
        try:
            original_words = word_tokenize(self.original_text)
            processed_words = word_tokenize(self.processed_text)
            
            stats = {
                "original_length": len(self.original_text),
                "processed_length": len(self.processed_text),
                "original_word_count": len(original_words),
                "processed_word_count": len(processed_words),
                "original_sentence_count": len(sent_tokenize(self.original_text)),
                "processed_sentence_count": len(sent_tokenize(self.processed_text)),
                "preprocessing_steps": self.preprocessing_steps
            }
            return stats
        except Exception as e:
            logger.error(f"Error getting statistics: {str(e)}")
            raise ValueError(f"Error getting statistics: {str(e)}")
    
    def get_processed_data(self) -> bytes:
        """Get processed text as bytes."""
        try:
            return self.processed_text.encode()
        except Exception as e:
            logger.error(f"Error getting processed data: {str(e)}")
            raise ValueError(f"Error getting processed data: {str(e)}")
    
    def get_word_frequencies(self, top_n: Optional[int] = None) -> Dict[str, int]:
        """Get word frequencies from processed text."""
        try:
            words = word_tokenize(self.processed_text.lower())
            freq_dist = nltk.FreqDist(words)
            if top_n:
                return dict(freq_dist.most_common(top_n))
            return dict(freq_dist)
        except Exception as e:
            logger.error(f"Error getting word frequencies: {str(e)}")
            raise ValueError(f"Error getting word frequencies: {str(e)}")
    
    def get_pos_tags(self) -> List[tuple]:
        """Get part-of-speech tags for words in the processed text."""
        try:
            words = word_tokenize(self.processed_text)
            return nltk.pos_tag(words)
        except Exception as e:
            logger.error(f"Error getting POS tags: {str(e)}")
            raise ValueError(f"Error getting POS tags: {str(e)}")'''

# Write the files
write_file('backend/processors/tabular.py', tabular_content)
write_file('backend/processors/text.py', text_content)
write_file('backend/processors/__init__.py', '"""Processors package initialization file."""') 
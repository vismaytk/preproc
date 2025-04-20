import nltk
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
            self.processed_text = re.sub(r'\d+', '', self.processed_text)
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
            # Split on whitespace for initial word count
            original_words = self.processed_text.split()
            
            # Tokenize and lemmatize
            words = word_tokenize(self.processed_text)
            lemmatized_words = []
            
            for word in words:
                # Only process actual words, skip punctuation and whitespace
                if word.strip() and any(c.isalnum() for c in word):
                    lemmatized = self.lemmatizer.lemmatize(word)
                    lemmatized_words.append(lemmatized)
            
            # Join back into text, ensuring proper spacing
            self.processed_text = " ".join(lemmatized_words)
            
            # Add to preprocessing steps
            self.preprocessing_steps.append(("lemmatize", None))
            logger.info(f"Lemmatized text: words reduced from {len(original_words)} to {len(lemmatized_words)}")
            
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
            raise ValueError(f"Error getting POS tags: {str(e)}")
"""Text data processor with NLP preprocessing operations."""

import re
import string
from typing import Any

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from api.core.logging import get_logger
from api.processors.base import BaseProcessor

# Download required NLTK data
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "averaged_perceptron_tagger",
                 "averaged_perceptron_tagger_eng"]:
    nltk.download(resource, quiet=True)

logger = get_logger(__name__)


class TextProcessor(BaseProcessor):
    """Process text data with various NLP preprocessing operations."""

    def __init__(self) -> None:
        self.original_text: str | None = None
        self.processed_text: str | None = None
        self.preprocessing_steps: list = []
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        self._detected_language: str | None = None
        self._is_english: bool | None = None

    def load_data(self, data: bytes) -> None:
        """Load text data from bytes."""
        try:
            self.original_text = data.decode("utf-8")
            self.processed_text = self.original_text
            self._detect_language()
            logger.info("text_loaded", length=len(self.original_text))
        except Exception as e:
            logger.error("text_load_error", error=str(e))
            raise ValueError(f"Error loading text: {e}")

    def _detect_language(self) -> None:
        """Detect the language of the loaded text."""
        try:
            from langdetect import detect
            if self.original_text and len(self.original_text.strip()) > 10:
                self._detected_language = detect(self.original_text)
                self._is_english = self._detected_language == "en"
                logger.info(
                    "language_detected",
                    language=self._detected_language,
                    is_english=self._is_english,
                )
            else:
                self._detected_language = "unknown"
                self._is_english = None
        except Exception:
            self._detected_language = "unknown"
            self._is_english = None

    def to_lowercase(self) -> None:
        """Convert text to lowercase."""
        assert self.processed_text is not None
        self.processed_text = self.processed_text.lower()
        self.preprocessing_steps.append(("to_lowercase", None))
        logger.info("text_lowercased")

    def remove_punctuation(self) -> None:
        """Remove punctuation from text."""
        assert self.processed_text is not None
        self.processed_text = self.processed_text.translate(
            str.maketrans("", "", string.punctuation)
        )
        self.preprocessing_steps.append(("remove_punctuation", None))
        logger.info("punctuation_removed")

    def remove_numbers(self) -> None:
        """Remove numbers from text."""
        assert self.processed_text is not None
        self.processed_text = re.sub(r"\d+", "", self.processed_text)
        self.preprocessing_steps.append(("remove_numbers", None))
        logger.info("numbers_removed")

    def remove_whitespace(self) -> None:
        """Normalize whitespace in text."""
        assert self.processed_text is not None
        self.processed_text = " ".join(self.processed_text.split())
        self.preprocessing_steps.append(("remove_whitespace", None))
        logger.info("whitespace_normalized")

    def remove_stopwords(self) -> None:
        """Remove English stop words from text."""
        assert self.processed_text is not None
        words = word_tokenize(self.processed_text)
        words = [w for w in words if w.lower() not in self.stop_words]
        self.processed_text = " ".join(words)
        self.preprocessing_steps.append(("remove_stopwords", None))
        logger.info("stopwords_removed")

    def remove_urls(self) -> None:
        """Remove URLs from text."""
        assert self.processed_text is not None
        self.processed_text = re.sub(
            r"https?://\S+|www\.\S+", "", self.processed_text
        )
        self.preprocessing_steps.append(("remove_urls", None))
        logger.info("urls_removed")

    def remove_emails(self) -> None:
        """Remove email addresses from text."""
        assert self.processed_text is not None
        self.processed_text = re.sub(
            r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
            "",
            self.processed_text,
        )
        self.preprocessing_steps.append(("remove_emails", None))
        logger.info("emails_removed")

    def lemmatize_text(self) -> None:
        """Lemmatize text tokens."""
        assert self.processed_text is not None
        words = word_tokenize(self.processed_text)
        lemmatized = [
            self.lemmatizer.lemmatize(word)
            for word in words
            if word.strip() and any(c.isalnum() for c in word)
        ]
        self.processed_text = " ".join(lemmatized)
        self.preprocessing_steps.append(("lemmatize", None))
        logger.info("text_lemmatized")

    def get_word_frequencies(self, top_n: int | None = None) -> dict[str, int]:
        """Get word frequencies from processed text.

        Args:
            top_n: Return only the top N most common words.

        Returns:
            Dictionary mapping words to their frequency counts.
        """
        assert self.processed_text is not None
        words = word_tokenize(self.processed_text.lower())
        # Filter to only alphanumeric tokens
        words = [w for w in words if w.isalnum()]
        freq_dist = nltk.FreqDist(words)
        if top_n:
            return dict(freq_dist.most_common(top_n))
        return dict(freq_dist)

    def get_pos_tags(self) -> list[tuple[str, str]]:
        """Get part-of-speech tags for processed text tokens.

        Returns:
            List of (word, tag) tuples.
        """
        assert self.processed_text is not None
        words = word_tokenize(self.processed_text)
        return nltk.pos_tag(words)

    def get_statistics(self) -> dict[str, Any]:
        """Get comprehensive text statistics."""
        assert self.original_text is not None and self.processed_text is not None

        original_words = word_tokenize(self.original_text)
        processed_words = word_tokenize(self.processed_text)
        alpha_words = [w for w in processed_words if w.isalnum()]

        # Vocabulary richness
        unique_words = set(w.lower() for w in alpha_words)
        vocab_richness = (
            len(unique_words) / len(alpha_words) if alpha_words else 0.0
        )

        # Average word length
        avg_word_length = (
            sum(len(w) for w in alpha_words) / len(alpha_words) if alpha_words else 0.0
        )

        # Top 10 words
        freq = self.get_word_frequencies(top_n=10)

        stats = {
            "original_length": len(self.original_text),
            "processed_length": len(self.processed_text),
            "original_word_count": len(original_words),
            "processed_word_count": len(processed_words),
            "original_sentence_count": len(sent_tokenize(self.original_text)),
            "processed_sentence_count": len(sent_tokenize(self.processed_text)),
            "preprocessing_steps": [
                {"step": step, "params": params} for step, params in self.preprocessing_steps
            ],
            "top_10_words": freq,
            "avg_word_length": round(avg_word_length, 2),
            "vocabulary_richness": round(vocab_richness, 4),
            "detected_language": self._detected_language,
            "is_english": self._is_english,
        }
        return stats

    def get_processed_data(self) -> bytes:
        """Get processed text as bytes."""
        assert self.processed_text is not None
        return self.processed_text.encode("utf-8")

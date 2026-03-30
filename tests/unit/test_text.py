"""Unit tests for TextProcessor."""

import pytest

from api.processors.text import TextProcessor


@pytest.fixture
def processor():
    return TextProcessor()


@pytest.fixture
def loaded_processor(processor, sample_text_data):
    processor.load_data(sample_text_data)
    return processor


class TestTextProcessor:
    """Tests for the TextProcessor class."""

    def test_load_data(self, loaded_processor):
        assert loaded_processor.original_text is not None
        assert loaded_processor.processed_text is not None
        assert len(loaded_processor.original_text) > 0

    def test_language_detection(self, loaded_processor):
        assert loaded_processor._detected_language is not None
        # The sample text is English
        assert loaded_processor._detected_language == "en"
        assert loaded_processor._is_english is True

    def test_to_lowercase(self, loaded_processor):
        loaded_processor.to_lowercase()
        assert "Hello" not in loaded_processor.processed_text
        assert "hello" in loaded_processor.processed_text

    def test_remove_punctuation(self, loaded_processor):
        loaded_processor.remove_punctuation()
        assert "!" not in loaded_processor.processed_text
        assert "..." not in loaded_processor.processed_text

    def test_remove_numbers(self, loaded_processor):
        loaded_processor.remove_numbers()
        assert "123" not in loaded_processor.processed_text

    def test_remove_whitespace(self, loaded_processor):
        loaded_processor.remove_whitespace()
        assert "  " not in loaded_processor.processed_text

    def test_remove_stopwords(self, loaded_processor):
        loaded_processor.remove_stopwords()
        # "the" and "is" should be removed
        words = loaded_processor.processed_text.lower().split()
        assert "the" not in words
        assert "is" not in words

    def test_remove_urls(self, loaded_processor):
        loaded_processor.remove_urls()
        assert "https://example.com" not in loaded_processor.processed_text

    def test_remove_emails(self, loaded_processor):
        loaded_processor.remove_emails()
        assert "test@example.com" not in loaded_processor.processed_text

    def test_lemmatize_text(self, loaded_processor):
        loaded_processor.lemmatize_text()
        # Lemmatization should produce non-empty output
        assert len(loaded_processor.processed_text.strip()) > 0
        assert ("lemmatize", None) in loaded_processor.preprocessing_steps

    def test_get_word_frequencies(self, loaded_processor):
        freq = loaded_processor.get_word_frequencies()
        assert isinstance(freq, dict)
        assert len(freq) > 0

    def test_get_word_frequencies_top_n(self, loaded_processor):
        top_3 = loaded_processor.get_word_frequencies(top_n=3)
        assert len(top_3) <= 3

    def test_get_pos_tags(self, loaded_processor):
        tags = loaded_processor.get_pos_tags()
        assert isinstance(tags, list)
        assert len(tags) > 0
        assert all(isinstance(tag, tuple) and len(tag) == 2 for tag in tags)

    def test_get_statistics(self, loaded_processor):
        stats = loaded_processor.get_statistics()
        assert "original_length" in stats
        assert "processed_length" in stats
        assert "original_word_count" in stats
        assert "processed_word_count" in stats
        assert "top_10_words" in stats
        assert "avg_word_length" in stats
        assert "vocabulary_richness" in stats
        assert "detected_language" in stats
        assert "is_english" in stats
        assert stats["vocabulary_richness"] > 0

    def test_get_processed_data(self, loaded_processor):
        data = loaded_processor.get_processed_data()
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_full_pipeline(self, loaded_processor):
        loaded_processor.to_lowercase()
        loaded_processor.remove_urls()
        loaded_processor.remove_emails()
        loaded_processor.remove_punctuation()
        loaded_processor.remove_numbers()
        loaded_processor.remove_whitespace()
        loaded_processor.remove_stopwords()
        loaded_processor.lemmatize_text()

        processed = loaded_processor.processed_text
        assert processed.islower() or all(c.isalnum() or c.isspace() for c in processed)
        assert "https://example.com" not in processed
        assert "test@example.com" not in processed
        assert "123" not in processed
        assert len(loaded_processor.preprocessing_steps) == 8

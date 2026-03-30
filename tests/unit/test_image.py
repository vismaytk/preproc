"""Unit tests for ImageProcessor."""

import numpy as np
import pytest
from PIL import Image
from io import BytesIO

from api.processors.image import ImageProcessor


@pytest.fixture
def processor():
    return ImageProcessor()


@pytest.fixture
def loaded_processor(processor, sample_image_data):
    processor.load_data(sample_image_data)
    return processor


class TestImageProcessor:
    """Tests for the ImageProcessor class."""

    def test_load_data(self, loaded_processor):
        assert loaded_processor.original_image is not None
        assert loaded_processor.processed_image is not None
        assert loaded_processor.original_image.shape == (50, 50, 3)

    def test_resize(self, loaded_processor):
        loaded_processor.resize(100, 100)
        assert loaded_processor.processed_image.shape[:2] == (100, 100)

    def test_resize_invalid(self, loaded_processor):
        with pytest.raises(ValueError, match="positive"):
            loaded_processor.resize(0, 100)

    def test_to_grayscale(self, loaded_processor):
        loaded_processor.to_grayscale()
        assert len(loaded_processor.processed_image.shape) == 2

    def test_normalize(self, loaded_processor):
        loaded_processor.normalize()
        assert loaded_processor.processed_image.dtype == np.uint8

    def test_convert_format_jpeg(self, loaded_processor):
        loaded_processor.convert_format("JPEG")
        assert loaded_processor.output_format == "JPEG"

    def test_convert_format_webp(self, loaded_processor):
        loaded_processor.convert_format("WEBP")
        assert loaded_processor.output_format == "WEBP"

    def test_convert_format_invalid(self, loaded_processor):
        with pytest.raises(ValueError, match="Unsupported"):
            loaded_processor.convert_format("BMP")

    def test_adjust_brightness(self, loaded_processor):
        original = loaded_processor.processed_image.copy()
        loaded_processor.adjust_brightness(1.5)
        # Brightness should change the image
        assert not np.array_equal(original, loaded_processor.processed_image)

    def test_adjust_contrast(self, loaded_processor):
        original = loaded_processor.processed_image.copy()
        loaded_processor.adjust_contrast(1.5)
        assert not np.array_equal(original, loaded_processor.processed_image)

    def test_get_statistics(self, loaded_processor):
        stats = loaded_processor.get_statistics()
        assert "original_dims" in stats
        assert "processed_dims" in stats
        assert "color_mode" in stats
        assert stats["color_mode"] == "RGB"
        assert "channel_means" in stats
        assert "red" in stats["channel_means"]
        assert "green" in stats["channel_means"]
        assert "blue" in stats["channel_means"]
        assert "original_size_bytes" in stats
        assert "processed_size_bytes" in stats

    def test_get_statistics_grayscale(self, loaded_processor):
        loaded_processor.to_grayscale()
        stats = loaded_processor.get_statistics()
        assert stats["color_mode"] == "Grayscale"
        assert "gray" in stats["channel_means"]

    def test_get_processed_data(self, loaded_processor):
        data = loaded_processor.get_processed_data()
        assert isinstance(data, bytes)
        assert len(data) > 0
        # Verify it's a valid image
        img = Image.open(BytesIO(data))
        assert img.size == (50, 50)

    def test_get_base64_image(self, loaded_processor):
        b64 = loaded_processor.get_base64_image()
        assert isinstance(b64, str)
        assert len(b64) > 0

    def test_full_pipeline(self, loaded_processor):
        loaded_processor.to_grayscale()
        loaded_processor.resize(100, 100)
        loaded_processor.normalize()

        stats = loaded_processor.get_statistics()
        assert stats["processed_dims"][:2] == [100, 100]
        assert stats["color_mode"] == "Grayscale"
        assert len(stats["preprocessing_steps"]) == 3

"""Image data processor with preprocessing operations."""

import base64
from io import BytesIO
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageEnhance

from api.core.logging import get_logger
from api.processors.base import BaseProcessor

logger = get_logger(__name__)


class ImageProcessor(BaseProcessor):
    """Process image data with various transformation operations."""

    def __init__(self) -> None:
        self.original_image: np.ndarray | None = None
        self.processed_image: np.ndarray | None = None
        self.pil_image: Image.Image | None = None
        self.original_size_bytes: int = 0
        self.original_format: str | None = None
        self.output_format: str = "PNG"
        self.preprocessing_steps: list = []

    def load_data(self, data: bytes) -> None:
        """Load image data from bytes."""
        try:
            self.original_size_bytes = len(data)
            self.pil_image = Image.open(BytesIO(data))
            self.original_format = self.pil_image.format or "PNG"

            # Convert to RGB if needed
            if self.pil_image.mode not in ("RGB", "L"):
                self.pil_image = self.pil_image.convert("RGB")

            # Convert to numpy array (BGR for OpenCV)
            img_rgb = np.array(self.pil_image)
            if len(img_rgb.shape) == 3:
                self.original_image = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            else:
                self.original_image = img_rgb

            self.processed_image = self.original_image.copy()
            logger.info(
                "image_loaded",
                shape=self.original_image.shape,
                format=self.original_format,
                size_bytes=self.original_size_bytes,
            )
        except Exception as e:
            logger.error("image_load_error", error=str(e))
            raise ValueError(f"Error loading image: {e}")

    def resize(self, width: int, height: int) -> None:
        """Resize the image to specified dimensions."""
        assert self.processed_image is not None
        if width <= 0 or height <= 0:
            raise ValueError("Dimensions must be positive numbers")
        self.processed_image = cv2.resize(self.processed_image, (width, height))
        self.preprocessing_steps.append(("resize", {"width": width, "height": height}))
        logger.info("image_resized", width=width, height=height)

    def normalize(self) -> None:
        """Normalize pixel values to [0, 1] range and back to uint8."""
        assert self.processed_image is not None
        normalized = self.processed_image.astype("float32") / 255.0
        self.processed_image = (normalized * 255).astype(np.uint8)
        self.preprocessing_steps.append(("normalize", None))
        logger.info("image_normalized")

    def to_grayscale(self) -> None:
        """Convert image to grayscale."""
        assert self.processed_image is not None
        if len(self.processed_image.shape) == 3:
            self.processed_image = cv2.cvtColor(self.processed_image, cv2.COLOR_BGR2GRAY)
        self.preprocessing_steps.append(("grayscale", None))
        logger.info("image_grayscale")

    def convert_format(self, output_format: str) -> None:
        """Set the output format for the processed image.

        Args:
            output_format: One of 'PNG', 'JPEG', or 'WEBP'.
        """
        valid_formats = {"PNG", "JPEG", "WEBP"}
        fmt = output_format.upper()
        if fmt not in valid_formats:
            raise ValueError(f"Unsupported format: {output_format}. Use {valid_formats}")
        self.output_format = fmt
        self.preprocessing_steps.append(("convert_format", fmt))
        logger.info("format_set", format=fmt)

    def adjust_brightness(self, factor: float) -> None:
        """Adjust image brightness.

        Args:
            factor: Brightness factor (1.0 = no change, >1 = brighter, <1 = darker).
        """
        assert self.processed_image is not None
        pil_img = self._to_pil(self.processed_image)
        enhancer = ImageEnhance.Brightness(pil_img)
        enhanced = enhancer.enhance(factor)
        self.processed_image = self._from_pil(enhanced)
        self.preprocessing_steps.append(("adjust_brightness", factor))
        logger.info("brightness_adjusted", factor=factor)

    def adjust_contrast(self, factor: float) -> None:
        """Adjust image contrast.

        Args:
            factor: Contrast factor (1.0 = no change, >1 = more contrast, <1 = less).
        """
        assert self.processed_image is not None
        pil_img = self._to_pil(self.processed_image)
        enhancer = ImageEnhance.Contrast(pil_img)
        enhanced = enhancer.enhance(factor)
        self.processed_image = self._from_pil(enhanced)
        self.preprocessing_steps.append(("adjust_contrast", factor))
        logger.info("contrast_adjusted", factor=factor)

    def get_statistics(self) -> dict[str, Any]:
        """Get image statistics including dimensions, color mode, and channel means."""
        assert self.original_image is not None and self.processed_image is not None

        # Determine color mode
        if len(self.processed_image.shape) == 2:
            color_mode = "Grayscale"
            channel_means = {"gray": round(float(self.processed_image.mean()), 2)}
        else:
            color_mode = "RGB"
            # OpenCV uses BGR ordering
            b, g, r = cv2.split(self.processed_image)
            channel_means = {
                "red": round(float(r.mean()), 2),
                "green": round(float(g.mean()), 2),
                "blue": round(float(b.mean()), 2),
            }

        processed_bytes = self._encode_image()

        return {
            "original_size_bytes": self.original_size_bytes,
            "processed_size_bytes": len(processed_bytes),
            "original_dims": list(self.original_image.shape),
            "processed_dims": list(self.processed_image.shape),
            "color_mode": color_mode,
            "channel_means": channel_means,
            "format": self.output_format,
            "preprocessing_steps": [
                {"step": step, "params": params} for step, params in self.preprocessing_steps
            ],
        }

    def get_processed_data(self) -> bytes:
        """Get processed image as bytes."""
        return self._encode_image()

    def get_base64_image(self) -> str:
        """Get processed image as base64-encoded string."""
        return base64.b64encode(self._encode_image()).decode("utf-8")

    def _encode_image(self) -> bytes:
        """Encode processed image to bytes in the output format."""
        assert self.processed_image is not None
        pil_img = self._to_pil(self.processed_image)
        output = BytesIO()

        save_kwargs: dict[str, Any] = {"format": self.output_format}
        if self.output_format == "JPEG":
            save_kwargs["quality"] = 85
        elif self.output_format == "PNG":
            save_kwargs["optimize"] = True

        pil_img.save(output, **save_kwargs)
        output.seek(0)
        return output.getvalue()

    @staticmethod
    def _to_pil(img: np.ndarray) -> Image.Image:
        """Convert OpenCV numpy array to PIL Image."""
        if len(img.shape) == 2:
            return Image.fromarray(img, mode="L")
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    @staticmethod
    def _from_pil(pil_img: Image.Image) -> np.ndarray:
        """Convert PIL Image to OpenCV numpy array."""
        img_array = np.array(pil_img)
        if len(img_array.shape) == 3:
            return cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        return img_array

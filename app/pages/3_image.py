"""Image Data Preprocessing Page."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import base64
from io import BytesIO
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

from app.components.download_button import download_image_b64
from app.components.file_uploader import file_uploader
from app.components.stats_panel import image_stats_panel
from app.utils.api_client import APIError, process_image
from app.utils.charts import plot_channel_histogram

st.set_page_config(page_title="Image Processing | DataPrep Pro", page_icon="🖼️", layout="wide")

st.title("🖼️ Image Data Preprocessing")
st.markdown("Upload an image to resize, normalize, convert, and apply transformations.")

# --- Sample Image ---
SAMPLE_PATH = Path(__file__).parent.parent.parent / "samples" / "sample.png"


def generate_sample_image() -> bytes:
    """Generate a simple test image if sample doesn't exist."""
    np.random.seed(42)
    # Create a gradient test image
    width, height = 200, 200
    img = np.zeros((height, width, 3), dtype=np.uint8)
    # Red gradient
    img[:, :, 0] = np.linspace(0, 255, width, dtype=np.uint8)
    # Green gradient
    img[:, :, 1] = np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1)
    # Blue channel
    img[:, :, 2] = 128
    # Add some noise
    noise = np.random.randint(0, 30, (height, width, 3), dtype=np.uint8)
    img = np.clip(img.astype(int) + noise, 0, 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    return buf.getvalue()


def load_sample_image() -> bytes:
    """Load the sample image file."""
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH.read_bytes()
    return generate_sample_image()


col_upload, col_sample = st.columns([3, 1])
with col_upload:
    upload_result = file_uploader(
        "Upload Image",
        accepted_types=["png", "jpg", "jpeg", "webp"],
        help_text="Upload an image file (max 5MB)",
        max_size_mb=5.0,
        key="image_upload",
    )
with col_sample:
    st.markdown("##### Or try a sample")
    if st.button("📂 Load Sample Image", key="load_sample_img", use_container_width=True):
        st.session_state["image_sample"] = load_sample_image()
        st.session_state["image_sample_name"] = "sample.png"

# Determine data source
file_content = None
filename = None
if upload_result:
    file_content, filename = upload_result
elif "image_sample" in st.session_state:
    file_content = st.session_state["image_sample"]
    filename = st.session_state["image_sample_name"]

if file_content and filename:
    try:
        image = Image.open(BytesIO(file_content))

        # --- Original Image Info ---
        file_size_mb = len(file_content) / (1024 * 1024)

        st.subheader("📷 Original Image")
        col_img, col_info = st.columns([2, 1])
        with col_img:
            st.image(image, caption=f"Original — {image.size[0]}×{image.size[1]}", use_container_width=True)
        with col_info:
            st.metric("Dimensions", f"{image.size[0]}×{image.size[1]}")
            st.metric("Mode", image.mode)
            st.metric("Format", image.format or "Unknown")
            st.metric("File Size", f"{file_size_mb:.2f} MB")

        # --- Preprocessing Options ---
        st.subheader("⚙️ Preprocessing Options")
        col1, col2 = st.columns(2)

        with col1:
            resize = st.checkbox("Resize Image")
            resize_dimensions = None
            if resize:
                aspect_ratio = image.size[0] / image.size[1]
                width = st.number_input("Width", min_value=1, value=min(image.size[0], 224))
                height = st.number_input("Height", min_value=1, value=int(width / aspect_ratio))
                resize_dimensions = f"{width}x{height}"

            grayscale = st.checkbox("Convert to Grayscale")

        with col2:
            normalize = st.checkbox("Normalize Pixel Values", value=True)

            output_format = st.selectbox(
                "Output Format",
                [None, "PNG", "JPEG", "WEBP"],
                format_func=lambda x: "Original" if x is None else x,
            )

            brightness = st.slider("Brightness", 0.5, 2.0, 1.0, 0.1)
            contrast = st.slider("Contrast", 0.5, 2.0, 1.0, 0.1)

        # --- Process Button ---
        if st.button("🚀 Process Image", type="primary", use_container_width=True):
            try:
                with st.spinner("Applying image transforms..."):
                    result = process_image(
                        file_content=file_content,
                        filename=filename,
                        resize_dimensions=resize_dimensions,
                        normalize=normalize,
                        grayscale=grayscale,
                        output_format=output_format,
                        brightness=brightness if brightness != 1.0 else None,
                        contrast=contrast if contrast != 1.0 else None,
                    )

                st.divider()

                # Statistics
                image_stats_panel(
                    result.original_shape,
                    result.processed_shape,
                    result.color_mode,
                    result.original_size_bytes,
                    result.processed_size_bytes,
                    result.format,
                )

                # --- Side-by-side Images ---
                st.subheader("🖼️ Before / After")
                col_before, col_after = st.columns(2)

                with col_before:
                    st.image(image, caption="Original", use_container_width=True)

                with col_after:
                    processed_bytes = base64.b64decode(result.processed_image_b64)
                    processed_image = Image.open(BytesIO(processed_bytes))
                    st.image(processed_image, caption="Processed", use_container_width=True)

                # --- Channel Histograms ---
                with st.expander("📊 Channel Analysis", expanded=False):
                    plot_channel_histogram(result.channel_means, "(Processed)")

                # Download
                fmt_ext = result.format.lower()
                if fmt_ext == "jpeg":
                    fmt_ext = "jpg"
                download_image_b64(
                    result.processed_image_b64,
                    filename=f"processed_image.{fmt_ext}",
                    mime=f"image/{result.format.lower()}",
                    key="download_img",
                )

            except APIError as e:
                st.error(f"❌ API Error: {e.detail}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    except Exception as e:
        st.error(f"Error loading image: {str(e)}")

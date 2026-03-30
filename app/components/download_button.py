"""Styled download button component."""

import base64

import streamlit as st


def download_button(
    data: bytes,
    filename: str,
    mime: str,
    label: str = "⬇️ Download Processed File",
    key: str | None = None,
) -> None:
    """Display a styled download button.

    Args:
        data: File content as bytes.
        filename: Name of the downloaded file.
        mime: MIME type of the file.
        label: Button label text.
        key: Unique Streamlit widget key.
    """
    st.download_button(
        label=label,
        data=data,
        file_name=filename,
        mime=mime,
        key=key,
        type="primary",
    )


def download_image_b64(
    b64_data: str,
    filename: str = "processed_image.png",
    mime: str = "image/png",
    label: str = "⬇️ Download Processed Image",
    key: str | None = None,
) -> None:
    """Download button for base64-encoded image data."""
    raw_bytes = base64.b64decode(b64_data)
    download_button(raw_bytes, filename, mime, label, key)

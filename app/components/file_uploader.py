"""Reusable file upload component with validation and styling."""


import streamlit as st


def file_uploader(
    label: str,
    accepted_types: list[str],
    help_text: str = "",
    max_size_mb: float = 10.0,
    key: str | None = None,
) -> tuple[bytes, str] | None:
    """Display a styled file uploader with validation.

    Args:
        label: Upload label text.
        accepted_types: List of accepted file extensions (e.g., ["csv"]).
        help_text: Help text shown below the uploader.
        max_size_mb: Maximum file size in MB.
        key: Unique Streamlit widget key.

    Returns:
        Tuple of (file_content_bytes, filename) or None if no file uploaded.
    """
    uploaded_file = st.file_uploader(
        label,
        type=accepted_types,
        help=help_text or f"Accepted formats: {', '.join(accepted_types)} (max {max_size_mb}MB)",
        key=key,
    )

    if uploaded_file is None:
        return None

    # Size check
    file_content = uploaded_file.getvalue()
    file_size_mb = len(file_content) / (1024 * 1024)

    if file_size_mb > max_size_mb:
        st.error(f"❌ File size ({file_size_mb:.1f}MB) exceeds the {max_size_mb}MB limit.")
        return None

    if len(file_content) == 0:
        st.error("❌ The uploaded file is empty.")
        return None

    # Show file info
    st.caption(f"📄 **{uploaded_file.name}** — {file_size_mb:.2f} MB")

    return file_content, uploaded_file.name

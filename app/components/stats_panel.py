"""Reusable statistics panel component."""

from typing import Any

import streamlit as st


def stats_panel(
    metrics: list[dict[str, Any]],
    columns: int = 4,
) -> None:
    """Display a row of metric cards.

    Args:
        metrics: List of dicts with keys 'label', 'value', and optionally 'delta'.
        columns: Number of columns to display metrics in.
    """
    cols = st.columns(columns)
    for i, metric in enumerate(metrics):
        with cols[i % columns]:
            delta = metric.get("delta")
            st.metric(
                label=metric["label"],
                value=metric["value"],
                delta=delta,
            )


def processing_summary(
    original_shape: list[int],
    processed_shape: list[int],
    rows_removed: int,
    steps: list[dict[str, Any]],
) -> None:
    """Display a summary of processing results."""
    st.success("✅ Processing complete!")

    metrics = [
        {"label": "Original Rows", "value": original_shape[0]},
        {"label": "Processed Rows", "value": processed_shape[0]},
        {"label": "Rows Removed", "value": rows_removed, "delta": f"-{rows_removed}"},
        {"label": "Columns", "value": processed_shape[1]},
    ]
    stats_panel(metrics)

    # Steps summary
    step_names = [s.get("step", "unknown") for s in steps]
    readable_steps = {
        "handle_missing": "🔧 Handle Missing Values",
        "remove_duplicates": "🗑️ Remove Duplicates",
        "scale_features": "📏 Scale Features",
        "drop_columns": "✂️ Drop Columns",
        "encode_categoricals": "🔢 Encode Categoricals",
    }
    with st.expander("📋 Processing Steps Applied", expanded=False):
        for step in step_names:
            st.write(readable_steps.get(step, f"• {step}"))


def text_stats_panel(
    original_length: int,
    processed_length: int,
    original_word_count: int,
    processed_word_count: int,
    detected_language: str | None = None,
    is_english: bool | None = None,
    avg_word_length: float = 0.0,
    vocabulary_richness: float = 0.0,
) -> None:
    """Display text processing statistics."""
    st.success("✅ Text processing complete!")

    metrics = [
        {"label": "Original Length", "value": f"{original_length:,} chars"},
        {"label": "Processed Length", "value": f"{processed_length:,} chars"},
        {"label": "Original Words", "value": f"{original_word_count:,}"},
        {"label": "Processed Words", "value": f"{processed_word_count:,}"},
    ]
    stats_panel(metrics)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Word Length", f"{avg_word_length:.1f} chars")
    with col2:
        st.metric("Vocabulary Richness", f"{vocabulary_richness:.2%}")
    with col3:
        if detected_language:
            lang_display = detected_language.upper()
            if is_english is False:
                st.warning(f"⚠️ Language: {lang_display} (not English)")
            else:
                st.metric("Language", lang_display)


def image_stats_panel(
    original_shape: list[int],
    processed_shape: list[int],
    color_mode: str,
    original_size_bytes: int,
    processed_size_bytes: int,
    format_str: str,
) -> None:
    """Display image processing statistics."""
    st.success("✅ Image processing complete!")

    orig_dims = f"{original_shape[1]}×{original_shape[0]}" if len(original_shape) >= 2 else str(original_shape)
    proc_dims = f"{processed_shape[1]}×{processed_shape[0]}" if len(processed_shape) >= 2 else str(processed_shape)

    metrics = [
        {"label": "Original Size", "value": orig_dims},
        {"label": "Processed Size", "value": proc_dims},
        {"label": "Color Mode", "value": color_mode},
        {"label": "Format", "value": format_str},
    ]
    stats_panel(metrics)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Original File Size", f"{original_size_bytes / 1024:.1f} KB")
    with col2:
        st.metric("Processed File Size", f"{processed_size_bytes / 1024:.1f} KB")

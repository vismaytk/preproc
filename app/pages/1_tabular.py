"""Tabular Data Preprocessing Page."""

import json
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from app.components.download_button import download_button
from app.components.file_uploader import file_uploader
from app.components.stats_panel import processing_summary
from app.utils.api_client import APIError, process_tabular
from app.utils.charts import (
    plot_categorical_distributions,
    plot_correlation_matrix,
    plot_missing_values,
    plot_numeric_distributions,
)

st.set_page_config(page_title="Tabular Processing | DataPrep Pro", page_icon="📊", layout="wide")

st.title("📊 Tabular Data Preprocessing")
st.markdown("Upload a CSV file to clean, transform, and analyze your tabular data.")

# --- Sample Data ---
SAMPLE_PATH = Path(__file__).parent.parent.parent / "samples" / "sample.csv"


def load_sample_csv() -> bytes:
    """Load the sample CSV file."""
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH.read_bytes()
    # Generate inline if file doesn't exist
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "age": np.random.choice([25, 30, 35, 40, 45, np.nan], n),
        "income": np.random.normal(60000, 15000, n).round(2),
        "score": np.random.choice([np.nan, 0.5, 0.6, 0.7, 0.8, 0.9], n),
        "department": np.random.choice(["Engineering", "Marketing", "Sales", "HR", None], n),
        "experience_years": np.random.randint(1, 20, n).astype(float),
    })
    # Add duplicates
    df = pd.concat([df, df.iloc[:5]], ignore_index=True)
    return df.to_csv(index=False).encode()


col_upload, col_sample = st.columns([3, 1])
with col_upload:
    upload_result = file_uploader(
        "Upload CSV file",
        accepted_types=["csv"],
        help_text="Upload a CSV file (max 10MB)",
        max_size_mb=10.0,
        key="tabular_upload",
    )
with col_sample:
    st.markdown("##### Or try a sample")
    if st.button("📂 Load Sample CSV", key="load_sample_csv", use_container_width=True):
        st.session_state["tabular_sample"] = load_sample_csv()
        st.session_state["tabular_sample_name"] = "sample.csv"

# Determine data source
file_content = None
filename = None
if upload_result:
    file_content, filename = upload_result
elif "tabular_sample" in st.session_state:
    file_content = st.session_state["tabular_sample"]
    filename = st.session_state["tabular_sample_name"]

if file_content and filename:
    try:
        df = pd.read_csv(BytesIO(file_content))

        if df.empty:
            st.error("The uploaded CSV file is empty.")
            st.stop()

        # --- Original Data Preview ---
        with st.expander("🔍 Data Preview", expanded=True):
            st.dataframe(df.head(10), use_container_width=True)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Rows", df.shape[0])
            with col2:
                st.metric("Columns", df.shape[1])
            with col3:
                st.metric("Missing Cells", int(df.isnull().sum().sum()))

        # --- Descriptive Statistics ---
        with st.expander("📈 Descriptive Statistics", expanded=False):
            st.dataframe(df.describe().round(3), use_container_width=True)

        # --- Missing Values Chart (BEFORE processing) ---
        with st.expander("🕳️ Missing Values Analysis", expanded=False):
            plot_missing_values(df)

        # --- Preprocessing Options ---
        st.subheader("⚙️ Preprocessing Options")
        col1, col2 = st.columns(2)

        with col1:
            handle_missing = st.selectbox(
                "Handle Missing Values",
                ["drop", "mean", "median"],
                help="drop: remove rows with missing values\nmean: fill with column mean\nmedian: fill with column median",
            )
            remove_duplicates = st.checkbox("Remove Duplicate Rows", value=True)

        with col2:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            scaling_method = None
            if len(numeric_cols) > 0:
                scaling_method = st.selectbox(
                    "Scaling Method",
                    [None, "standard", "minmax", "robust"],
                    format_func=lambda x: "None" if x is None else x.title(),
                )
            else:
                st.info("No numeric columns found for scaling")

            columns_to_drop = st.multiselect(
                "Select Columns to Drop",
                df.columns.tolist(),
            )

        # --- Process Button ---
        if st.button("🚀 Process Data", type="primary", use_container_width=True):
            progress = st.progress(0, text="Loading data...")

            try:
                progress.progress(20, text="Handling missing values...")

                with st.spinner("Processing your CSV..."):
                    result = process_tabular(
                        file_content=file_content,
                        filename=filename,
                        handle_missing=handle_missing,
                        remove_duplicates=remove_duplicates,
                        columns_to_drop=json.dumps(columns_to_drop) if columns_to_drop else None,
                        scaling_method=scaling_method,
                    )

                progress.progress(80, text="Generating results...")

                # --- Results ---
                st.divider()
                processing_summary(
                    result.original_shape,
                    result.processed_shape,
                    result.rows_removed,
                    result.preprocessing_steps,
                )

                # Diff summary
                diff_parts = []
                if result.rows_removed > 0:
                    diff_parts.append(f"**{result.rows_removed}** rows removed")
                for step in result.preprocessing_steps:
                    if step.get("step") == "remove_duplicates":
                        params = step.get("params", {})
                        if isinstance(params, dict):
                            dup_count = params.get("removed", 0)
                            if dup_count:
                                diff_parts.append(f"**{dup_count}** duplicates removed")
                    elif step.get("step") == "scale_features":
                        params = step.get("params", {})
                        if isinstance(params, dict):
                            method = params.get("method", "")
                            cols = params.get("columns", [])
                            diff_parts.append(f"**{len(cols)}** columns scaled ({method})")
                if diff_parts:
                    st.info("📊 " + " · ".join(diff_parts))

                progress.progress(100, text="Done!")

                if result.processed_df is not None:
                    # Processed data preview
                    with st.expander("📋 Processed Data Preview", expanded=True):
                        st.dataframe(result.processed_df.head(10), use_container_width=True)

                    # --- Side-by-side distributions (before/after) ---
                    with st.expander("📊 Numeric Distributions (Before / After)", expanded=False):
                        col_before, col_after = st.columns(2)
                        with col_before:
                            st.markdown("**Before Processing**")
                            plot_numeric_distributions(df, "(Original)")
                        with col_after:
                            st.markdown("**After Processing**")
                            plot_numeric_distributions(result.processed_df, "(Processed)")

                    # Correlation matrix (after)
                    with st.expander("🔗 Correlation Matrix", expanded=False):
                        plot_correlation_matrix(result.processed_df)

                    # Categorical distributions
                    if result.categorical_columns:
                        with st.expander("📊 Categorical Distributions", expanded=False):
                            plot_categorical_distributions(result.processed_df)

                    # Download
                    csv_bytes = result.processed_df.to_csv(index=False).encode()
                    download_button(
                        csv_bytes,
                        "processed_data.csv",
                        "text/csv",
                        label="⬇️ Download Processed CSV",
                        key="download_csv",
                    )

            except APIError as e:
                progress.empty()
                st.error(f"❌ API Error: {e.detail}")
            except Exception as e:
                progress.empty()
                st.error(f"❌ Error: {str(e)}")

    except Exception as e:
        st.error(f"Error reading CSV file: {str(e)}")
        st.markdown("""
        Please ensure your CSV file:
        - Is not empty
        - Has proper column headers
        - Uses UTF-8 encoding
        """)

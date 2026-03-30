"""Plotly chart helpers for the Streamlit UI.

Ported and improved from frontend/utils/helpers.py — these were
fully implemented but never called anywhere in the frontend.
"""


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


def plot_missing_values(df: pd.DataFrame) -> None:
    """Display a bar chart of missing values per column."""
    missing = df.isnull().sum()
    if not missing.any():
        st.info("✅ No missing values found in the dataset.")
        return

    missing_df = pd.DataFrame({
        "Column": missing.index,
        "Missing Values": missing.values,
        "Percentage": (missing.values / len(df) * 100).round(2),
    })
    missing_df = missing_df[missing_df["Missing Values"] > 0].sort_values(
        "Missing Values", ascending=True
    )

    fig = px.bar(
        missing_df,
        x="Missing Values",
        y="Column",
        text="Percentage",
        orientation="h",
        title="Missing Values by Column",
        color="Percentage",
        color_continuous_scale="Reds",
    )
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(
        xaxis_title="Count",
        yaxis_title="",
        showlegend=False,
        height=max(300, len(missing_df) * 35),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_numeric_distributions(df: pd.DataFrame, title_suffix: str = "") -> None:
    """Display histogram distributions for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        st.info("No numeric columns found.")
        return

    for col in numeric_cols:
        fig = px.histogram(
            df,
            x=col,
            title=f"Distribution of {col} {title_suffix}".strip(),
            nbins=30,
            color_discrete_sequence=["#667eea"],
        )
        fig.update_layout(
            xaxis_title=col,
            yaxis_title="Frequency",
            height=300,
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """Display a correlation matrix heatmap for numeric columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        st.info("Not enough numeric columns for correlation analysis.")
        return

    corr = df[numeric_cols].corr()
    fig = px.imshow(
        corr,
        title="Correlation Matrix",
        labels=dict(color="Correlation"),
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        text_auto=".2f",
    )
    fig.update_layout(
        height=max(400, len(numeric_cols) * 40),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_categorical_distributions(df: pd.DataFrame, max_categories: int = 20) -> None:
    """Display bar charts for categorical columns."""
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) == 0:
        st.info("No categorical columns found.")
        return

    for col in cat_cols:
        value_counts = df[col].value_counts()
        if len(value_counts) > max_categories:
            st.caption(f"Showing top {max_categories} of {len(value_counts)} categories for '{col}'")
            value_counts = value_counts.head(max_categories)

        fig = px.bar(
            x=value_counts.values,
            y=value_counts.index,
            orientation="h",
            title=f"Distribution of {col}",
            color_discrete_sequence=["#764ba2"],
        )
        fig.update_layout(
            xaxis_title="Count",
            yaxis_title=col,
            height=max(300, len(value_counts) * 25),
            margin=dict(l=10, r=10, t=40, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)


def plot_word_frequencies(frequencies: dict[str, int], top_n: int = 20) -> None:
    """Display a horizontal bar chart of word frequencies."""
    if not frequencies:
        st.info("No word frequency data available.")
        return

    # Sort and limit
    sorted_freq = sorted(frequencies.items(), key=lambda x: x[1], reverse=True)[:top_n]
    words = [item[0] for item in sorted_freq]
    counts = [item[1] for item in sorted_freq]

    fig = px.bar(
        x=counts,
        y=words,
        orientation="h",
        title=f"Top {len(words)} Most Frequent Words",
        color=counts,
        color_continuous_scale="Viridis",
    )
    fig.update_layout(
        xaxis_title="Frequency",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        showlegend=False,
        height=max(400, len(words) * 25),
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_word_count_comparison(original_count: int, processed_count: int) -> None:
    """Display a before/after word count comparison bar chart."""
    fig = go.Figure(data=[
        go.Bar(
            x=["Original", "Processed"],
            y=[original_count, processed_count],
            marker_color=["#667eea", "#764ba2"],
            text=[original_count, processed_count],
            textposition="outside",
        )
    ])
    fig.update_layout(
        title="Word Count: Before vs After",
        yaxis_title="Word Count",
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)


def plot_channel_histogram(
    channel_means: dict[str, float],
    title_suffix: str = "",
) -> None:
    """Display a bar chart of per-channel mean pixel values."""
    if not channel_means:
        return

    channels = list(channel_means.keys())
    values = list(channel_means.values())

    color_map = {
        "red": "#ef4444",
        "green": "#22c55e",
        "blue": "#3b82f6",
        "gray": "#6b7280",
    }
    colors = [color_map.get(ch.lower(), "#6b7280") for ch in channels]

    fig = go.Figure(data=[
        go.Bar(
            x=channels,
            y=values,
            marker_color=colors,
            text=[f"{v:.1f}" for v in values],
            textposition="outside",
        )
    ])
    fig.update_layout(
        title=f"Channel Mean Pixel Values {title_suffix}".strip(),
        yaxis_title="Mean Value (0-255)",
        yaxis_range=[0, 280],
        height=350,
        margin=dict(l=10, r=10, t=40, b=10),
    )
    st.plotly_chart(fig, use_container_width=True)

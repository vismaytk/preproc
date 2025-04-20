import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional
import base64
from io import BytesIO
from PIL import Image

def show_error(message: str) -> None:
    """Display an error message."""
    st.error(message)

def show_success(message: str) -> None:
    """Display a success message."""
    st.success(message)

def show_info(message: str) -> None:
    """Display an info message."""
    st.info(message)

def show_warning(message: str) -> None:
    """Display a warning message."""
    st.warning(message)

def plot_missing_values(df: pd.DataFrame) -> None:
    """
    Create a bar plot of missing values in the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    """
    missing = df.isnull().sum()
    if missing.any():
        missing_df = pd.DataFrame({
            'Column': missing.index,
            'Missing Values': missing.values,
            'Percentage': (missing.values / len(df) * 100).round(2)
        })
        missing_df = missing_df[missing_df['Missing Values'] > 0]
        
        fig = px.bar(
            missing_df,
            x='Column',
            y='Missing Values',
            text='Percentage',
            title='Missing Values by Column',
            labels={'Missing Values': 'Count', 'Column': 'Column Name'}
        )
        fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig)
    else:
        st.info("No missing values found in the dataset.")

def plot_numeric_distributions(df: pd.DataFrame) -> None:
    """
    Create distribution plots for numeric columns.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        for col in numeric_cols:
            fig = px.histogram(
                df,
                x=col,
                title=f'Distribution of {col}',
                labels={col: 'Value', 'count': 'Frequency'}
            )
            st.plotly_chart(fig)
    else:
        st.info("No numeric columns found in the dataset.")

def plot_categorical_distributions(df: pd.DataFrame, max_categories: int = 20) -> None:
    """
    Create bar plots for categorical columns.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        max_categories (int): Maximum number of categories to display
    """
    categorical_cols = df.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            value_counts = df[col].value_counts()
            if len(value_counts) > max_categories:
                st.warning(f"Column '{col}' has too many unique values ({len(value_counts)}). Showing top {max_categories}.")
                value_counts = value_counts.head(max_categories)
            
            fig = px.bar(
                x=value_counts.index,
                y=value_counts.values,
                title=f'Distribution of {col}',
                labels={'x': col, 'y': 'Count'}
            )
            st.plotly_chart(fig)
    else:
        st.info("No categorical columns found in the dataset.")

def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Create a correlation matrix heatmap for numeric columns.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        corr = df[numeric_cols].corr()
        fig = px.imshow(
            corr,
            title='Correlation Matrix',
            labels=dict(color="Correlation"),
            color_continuous_scale='RdBu'
        )
        st.plotly_chart(fig)
    else:
        st.info("Not enough numeric columns for correlation analysis.")

def get_file_download_link(data: Any, filename: str, mime: str) -> str:
    """
    Create a download link for file data.
    
    Args:
        data (Any): Data to download
        filename (str): Name of the file
        mime (str): MIME type of the file
        
    Returns:
        str: HTML string for download link
    """
    if isinstance(data, pd.DataFrame):
        buffer = BytesIO()
        data.to_csv(buffer, index=False)
        data = buffer.getvalue()
    elif isinstance(data, Image.Image):
        buffer = BytesIO()
        data.save(buffer, format='PNG')
        data = buffer.getvalue()
    elif isinstance(data, str):
        data = data.encode()
    
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:{mime};base64,{b64}" download="{filename}">Download {filename}</a>'
    return href

def show_dataset_info(df: pd.DataFrame) -> None:
    """
    Display comprehensive information about the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    """
    st.subheader("Dataset Information")
    
    # Basic info
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    
    # Column types
    st.write("\nColumn Types:")
    dtypes_df = pd.DataFrame({
        'Column': df.dtypes.index,
        'Type': df.dtypes.values.astype(str)
    })
    st.dataframe(dtypes_df)
    
    # Memory usage
    memory_usage = df.memory_usage(deep=True)
    total_memory = memory_usage.sum()
    st.write(f"\nTotal memory usage: {total_memory / 1024 / 1024:.2f} MB")
    
    # Unique values
    st.write("\nUnique Values per Column:")
    unique_counts = pd.DataFrame({
        'Column': df.columns,
        'Unique Values': [df[col].nunique() for col in df.columns],
        'Percentage': [(df[col].nunique() / len(df) * 100).round(2) for col in df.columns]
    })
    st.dataframe(unique_counts)

def show_text_statistics(text: str) -> None:
    """
    Display statistics about text data.
    
    Args:
        text (str): Text to analyze
    """
    st.subheader("Text Statistics")
    
    # Basic stats
    words = text.split()
    sentences = text.split('.')
    
    stats = {
        "Character count": len(text),
        "Word count": len(words),
        "Sentence count": len(sentences),
        "Average word length": sum(len(word) for word in words) / len(words),
        "Average sentence length": len(words) / len(sentences)
    }
    
    for key, value in stats.items():
        st.write(f"{key}: {value:.2f}")
    
    # Word frequency
    word_freq = pd.Series(words).value_counts()
    st.write("\nTop 10 most frequent words:")
    st.dataframe(word_freq.head(10)) 
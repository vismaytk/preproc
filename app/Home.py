"""DataPrep Pro — Home Page (Landing Page).

This replaces the old streamlit_app.py as the Streamlit entry point.
"""

import streamlit as st

st.set_page_config(
    page_title="DataPrep Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS ---
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .main-header h1 {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 3rem;
        font-weight: 800;
    }
    .tagline {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .feature-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid;
    }
    .feature-card h3 {
        margin-top: 0;
    }
    .stat-box {
        text-align: center;
        padding: 1rem;
        background: #f0f2f6;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown('<div class="main-header"><h1>⚡ DataPrep Pro</h1></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="tagline">Professional data preprocessing for machine learning workflows. '
    'Clean, transform, and analyze your data in seconds.</p>',
    unsafe_allow_html=True,
)

st.divider()

# --- Feature Cards ---
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### 📊 Tabular Data")
    st.markdown("""
    - Handle missing values (drop, mean, median)
    - Remove duplicates
    - Scale numeric features
    - Encode categorical variables
    - Outlier detection (IQR, Z-score)
    - Interactive visualizations
    """)
    if st.button("→ Go to Tabular", key="nav_tabular", use_container_width=True):
        st.switch_page("pages/1_tabular.py")

with col2:
    st.markdown("### 📝 Text Data")
    st.markdown("""
    - Stop word removal
    - Lemmatization
    - URL & email removal
    - Language detection
    - Word frequency analysis
    - POS tagging
    """)
    if st.button("→ Go to Text", key="nav_text", use_container_width=True):
        st.switch_page("pages/2_text.py")

with col3:
    st.markdown("### 🖼️ Image Data")
    st.markdown("""
    - Resize & crop
    - Normalize pixel values
    - Grayscale conversion
    - Brightness & contrast adjustment
    - Format conversion (PNG/JPEG/WEBP)
    - Per-channel analysis
    """)
    if st.button("→ Go to Image", key="nav_image", use_container_width=True):
        st.switch_page("pages/3_image.py")

st.divider()

# --- Quick Start ---
st.subheader("🚀 Quick Start")
st.markdown("""
Each processing page includes **sample data** — click the "Load Sample" button on any page
to see DataPrep Pro in action without uploading anything.

**Three steps to clean data:**
1. **Upload** your CSV, text, or image file
2. **Configure** preprocessing options
3. **Download** the processed result

That's it! No sign-up, no API keys, no setup.
""")

# --- Tech Stack ---
st.divider()
st.subheader("🛠️ Built With")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**FastAPI**")
    st.caption("High-performance API")
with col2:
    st.markdown("**Streamlit**")
    st.caption("Interactive UI")
with col3:
    st.markdown("**scikit-learn**")
    st.caption("ML preprocessing")
with col4:
    st.markdown("**NLTK**")
    st.caption("NLP pipeline")

# --- Footer ---
st.divider()
st.markdown(
    '<p style="text-align: center; color: #999; font-size: 0.85rem;">'
    "DataPrep Pro v1.0.0 · Built with ❤️ by Vismay TK"
    "</p>",
    unsafe_allow_html=True,
)

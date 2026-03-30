"""Text Data Preprocessing Page."""

from pathlib import Path

import streamlit as st

from app.components.download_button import download_button
from app.components.file_uploader import file_uploader
from app.components.stats_panel import text_stats_panel
from app.utils.api_client import APIError, get_word_frequencies, process_text
from app.utils.charts import plot_word_count_comparison, plot_word_frequencies

st.set_page_config(page_title="Text Processing | DataPrep Pro", page_icon="📝", layout="wide")

st.title("📝 Text Data Preprocessing")
st.markdown("Upload a text file to clean and analyze with NLP techniques.")

# --- Sample Data ---
SAMPLE_PATH = Path(__file__).parent.parent.parent / "samples" / "sample.txt"

SAMPLE_TEXT = """Natural language processing (NLP) is a subfield of linguistics, computer science, and artificial intelligence concerned with the interactions between computers and human language. The goal is to enable computers to understand, interpret, and generate human language in a way that is both meaningful and useful. NLP combines computational linguistics with statistical, machine learning, and deep learning models.

Common NLP tasks include text classification, named entity recognition, sentiment analysis, machine translation, and question answering. These tasks require understanding of syntax, semantics, and pragmatics. Modern NLP heavily relies on transformer architectures like BERT and GPT, which have achieved state-of-the-art results across many benchmarks.

For more info visit https://example.com or email contact@example.com. The field continues to evolve rapidly with new breakthroughs in 2024 and 2025."""


def load_sample_text() -> bytes:
    """Load the sample text file."""
    if SAMPLE_PATH.exists():
        return SAMPLE_PATH.read_bytes()
    return SAMPLE_TEXT.encode()


col_upload, col_sample = st.columns([3, 1])
with col_upload:
    upload_result = file_uploader(
        "Upload TXT file",
        accepted_types=["txt"],
        help_text="Upload a text file (max 5MB)",
        max_size_mb=5.0,
        key="text_upload",
    )
with col_sample:
    st.markdown("##### Or try a sample")
    if st.button("📂 Load Sample Text", key="load_sample_text", use_container_width=True):
        st.session_state["text_sample"] = load_sample_text()
        st.session_state["text_sample_name"] = "sample.txt"

# Determine data source
file_content = None
filename = None
if upload_result:
    file_content, filename = upload_result
elif "text_sample" in st.session_state:
    file_content = st.session_state["text_sample"]
    filename = st.session_state["text_sample_name"]

if file_content and filename:
    text_content = file_content.decode("utf-8", errors="replace")

    # Original text preview
    with st.expander("📖 Original Text Preview", expanded=True):
        st.text_area("", text_content, height=200, key="original_text_preview", disabled=True)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Characters", f"{len(text_content):,}")
        with col2:
            st.metric("Words", f"{len(text_content.split()):,}")

    # --- Preprocessing Options ---
    st.subheader("⚙️ Preprocessing Options")
    col1, col2, col3 = st.columns(3)

    with col1:
        remove_stopwords = st.checkbox("Remove Stop Words", value=True)
        lemmatize = st.checkbox("Lemmatize Text", value=True)

    with col2:
        lowercase = st.checkbox("Convert to Lowercase", value=True)
        remove_urls = st.checkbox("Remove URLs", value=False)

    with col3:
        remove_emails = st.checkbox("Remove Email Addresses", value=False)

    # --- Process Button ---
    if st.button("🚀 Process Text", type="primary", use_container_width=True):
        try:
            with st.spinner("Running NLP pipeline..."):
                result = process_text(
                    file_content=file_content,
                    filename=filename,
                    remove_stopwords=remove_stopwords,
                    lemmatize=lemmatize,
                    lowercase=lowercase,
                    remove_urls=remove_urls,
                    remove_emails=remove_emails,
                )

            st.divider()

            # Statistics
            text_stats_panel(
                original_length=result.original_length,
                processed_length=result.processed_length,
                original_word_count=result.original_word_count,
                processed_word_count=result.processed_word_count,
                detected_language=result.detected_language,
                is_english=result.is_english,
                avg_word_length=result.avg_word_length,
                vocabulary_richness=result.vocabulary_richness,
            )

            # Processed text
            with st.expander("📝 Processed Text", expanded=True):
                st.text_area(
                    "", result.processed_text, height=200,
                    key="processed_text_preview", disabled=True,
                )

            # --- Charts ---
            col_chart1, col_chart2 = st.columns(2)

            with col_chart1:
                # Word count comparison
                plot_word_count_comparison(
                    result.original_word_count,
                    result.processed_word_count,
                )

            with col_chart2:
                # Word frequencies from the response
                if result.top_10_words:
                    plot_word_frequencies(result.top_10_words, top_n=20)

            # Try to get full word frequencies from the session endpoint
            if result.session_id:
                try:
                    full_freq = get_word_frequencies(result.session_id, top_n=20)
                    if full_freq:
                        with st.expander("📊 Full Word Frequency Analysis", expanded=False):
                            plot_word_frequencies(full_freq, top_n=20)
                except Exception:
                    pass  # Session endpoint may not be available

            # Download
            download_button(
                result.processed_text.encode(),
                "processed_text.txt",
                "text/plain",
                label="⬇️ Download Processed Text",
                key="download_txt",
            )

        except APIError as e:
            st.error(f"❌ API Error: {e.detail}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

import os
import re
import streamlit as st
from typing import List, Dict, Union
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from google import genai

load_dotenv(override=True)

# ============ Caching and Initialization ============

@st.cache_resource
def load_sentence_transformer_model():
    """Load the Sentence-Transformer model once and cache it."""
    try:
        return SentenceTransformer("all-MiniLM-L6-v2")
    except Exception as e:
        st.error(f"Error loading Sentence-Transformer model: {e}")
        return None

@st.cache_resource
def get_gemini_client():
    """Initialize the Gemini client once and cache it."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        st.error("Please set the GOOGLE_API_KEY environment variable.")
        return None
    try:
        client = genai.Client(api_key=api_key)
        return client
    except Exception as e:
        st.error(f"Error initializing Gemini client: {e}")
        return None

# ============ Chunking Functions ============

def fixed_size_chunking(text: str, chunk_size: int, overlap: int = 0) -> List[str]:
    """Break text into fixed-size chunks with optional overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start += chunk_size - overlap
    return chunks

def recursive_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """Recursively split text by paragraphs -> sentences -> fallback fixed size."""
    paragraphs = text.split("\n\n")
    chunks = []
    
    for para in paragraphs:
        sentences = re.split(r'(?<=[.!?]) +', para)
        current_chunk_text = ""
        for sentence in sentences:
            if len(current_chunk_text) + len(sentence) + 1 <= chunk_size:
                current_chunk_text += sentence + " "
            else:
                if current_chunk_text:
                    chunks.append(current_chunk_text.strip())
                # Add overlap for the next chunk
                overlap_text = current_chunk_text[-overlap:] if overlap > 0 else ""
                current_chunk_text = overlap_text + sentence + " "
        if current_chunk_text:
            chunks.append(current_chunk_text.strip())
            
    return chunks

def generalized_chunking(text: str, fallback_chunk_size: int = 500) -> List[Dict[str, str]]:
    """Chunk text by headings (#, CAPS, or keywords) with fallback."""
    lines = text.split("\n")
    chunks = []
    current_section = {"header": "Introduction", "content": ""}

    for line in lines:
        clean_line = line.strip()
        is_markdown_header = bool(re.match(r'^(#+)\s', clean_line))
        is_caps_heading = clean_line.isupper() and len(clean_line.split()) < 12
        is_keyword_heading = bool(re.match(r'^(Chapter|Section|Conclusion|Abstract|Introduction|Methodology|Results|Discussion|Appendix|References)\b', clean_line, re.I))

        if is_markdown_header or is_caps_heading or is_keyword_heading:
            if current_section["content"].strip():
                chunks.append(current_section)
            current_section = {"header": clean_line, "content": ""}
        else:
            current_section["content"] += " " + clean_line

    if current_section["content"].strip():
        chunks.append(current_section)

    if len(chunks) <= 1:
        raw_text = chunks[0]["content"].strip() if chunks else text
        fallback_chunks = fixed_size_chunking(raw_text, fallback_chunk_size)
        return [{"header": f"Chunk {i+1}", "content": c} for i, c in enumerate(fallback_chunks)]

    return chunks

@st.cache_data(show_spinner="Performing semantic chunking...")
def semantic_chunking(text: str, threshold: float = 0.7) -> List[str]:
    """Split sentences into chunks based on semantic similarity."""
    try:
        model = load_sentence_transformer_model()
        sentences = text.split(". ")
        embeddings = model.encode(sentences, show_progress_bar=False)

        chunks, current_chunk = [], [sentences[0]]
        for i in range(1, len(sentences)):
            sim = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            if sim < threshold:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
            current_chunk.append(sentences[i])
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks
    except Exception as e:
        st.error(f"Semantic chunking failed: {e}")
        return [text]

@st.cache_data(show_spinner="Using LLM to chunk text...")
def llm_based_chunking(text: str, chunk_size: int = 500) -> List[str]:
    """Use an LLM to chunk text into coherent sections."""
    client = get_gemini_client()
    if not client:
        return [text]
    
    prompt = f"""
    Break the following text into coherent chunks of about {chunk_size} words each. 
    Each chunk should represent a complete idea or section. Do not add any extra commentary or formatting, just provide the raw chunks.
    ---
    Text:
    {text}
    ---
    """
    try:
        response = response = client.models.generate_content(
    model="gemini-2.5-flash", contents=prompt
)
        chunks = response.text.split("\n\n")
        return [c.strip() for c in chunks if c.strip()]
    except Exception as e:
        st.error(f"LLM-based chunking failed: {e}")
        return [text]

def safe_chunking_with_fallback(text: str, primary_func, **kwargs):
    """Wrapper to ensure a chunking function returns valid output, with a fallback."""
    try:
        chunks = primary_func(text, **kwargs)
        if not chunks or len(chunks) == 0:
            st.warning("Primary chunking method returned no chunks. Falling back to fixed-size.")
            return fixed_size_chunking(text, kwargs.get("chunk_size", 500))
        return chunks
    except Exception as e:
        st.error(f"Primary chunking failed with error: {e}. Falling back to fixed-size chunking.")
        return fixed_size_chunking(text, kwargs.get("chunk_size", 500))

# ============ Streamlit UI ============
st.set_page_config(layout="wide")


st.title("📑 Text Chunking Playground for RAG")
st.markdown("""
    ### 🌟 Features
        
    - **Multiple Chunking Strategies**: Compare fixed-size, recursive, document-structure-aware, semantic, token-based, and LLM-based approaches
    - **Advanced Analytics**: Detailed metrics including token counts, cost estimates, and size distributions
    - **Interactive Visualizations**: Charts and graphs to understand chunking performance
    - **Flexible Export Options**: Download results in JSON, CSV, or TXT formats
    - **Real-time Comparison**: Side-by-side analysis of different strategies
        
    ### 📚 Supported Strategies
        
    | Strategy | Description | Best For |
    |----------|-------------|----------|
    | **Fixed-size** | Equal-sized chunks with overlap | Simple, consistent processing |
    | **Recursive** | Hierarchical splitting at natural boundaries | General-purpose text |
    | **Document-based** | Structure-aware (headers, sections) | Structured documents |
    | **Semantic** | Groups semantically similar content | Maintaining context |
    | **LLM-based** | AI-powered intelligent splitting | Complex, nuanced content |
        
    ### 🚀 Getting Started
        
    1. Upload your document (TXT, PDF, DOCX, or Markdown)
    2. Select one or more chunking strategies to compare
    3. Adjust parameters like chunk size and overlap
    4. Run the analysis and explore results
    5. Export your preferred chunking results
    """)

# Use a container for the controls to group them visually
with st.container():
    file = st.file_uploader("Upload a TXT or PDF file", type=["txt", "pdf"])
    st.markdown("---")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        strategy = st.selectbox(
            "Select Chunking Strategy",
            ["Fixed-size", "Recursive", "Document-based", "Semantic", "LLM-based"]
        )
    
    with col2:
        chunk_size = st.slider("Chunk Size", 100, 1000, 500, 50)
    
    with col3:
        if strategy == "Fixed-size" or strategy == "Recursive":
            overlap = st.slider("Overlap", 0, 200, 50, 10)
        elif strategy == "Semantic":
            threshold = st.slider("Semantic Similarity Threshold", 0.1, 0.95, 0.7, 0.05)

if file:
    text = ""
    with st.spinner("Extracting text from file..."):
        if file.name.endswith(".pdf"):
            reader = PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n"
        else:
            text = file.read().decode("utf-8")

    st.subheader("Original Text Preview")
    with st.expander("Click to view full text"):
        st.text_area("Original Text", text, height=200, disabled=True)

    if st.button("✨ Run Chunking"):
        if strategy == "Fixed-size":
            chunks = safe_chunking_with_fallback(text, fixed_size_chunking, chunk_size=chunk_size, overlap=overlap)
        elif strategy == "Recursive":
            chunks = safe_chunking_with_fallback(text, recursive_chunking, chunk_size=chunk_size, overlap=overlap)
        elif strategy == "Document-based":
            chunks = safe_chunking_with_fallback(text, generalized_chunking, fallback_chunk_size=chunk_size)
        elif strategy == "Semantic":
            chunks = safe_chunking_with_fallback(text, semantic_chunking, threshold=threshold)
        elif strategy == "LLM-based":
            chunks = safe_chunking_with_fallback(text, llm_based_chunking, chunk_size=chunk_size)
        else:
            chunks = [text]

        st.markdown("---")
        st.subheader("Chunking Results")
        st.success(f"Total Chunks: {len(chunks)}")
        
        # Display the chunks in collapsible expanders
        for i, chunk in enumerate(chunks):
            if i >= 20:
                st.info("Displaying the first 20 chunks only to save space. The full list is available in the application.")
                break
            
            with st.expander(f"Chunk {i+1}", expanded=False):
                if isinstance(chunk, dict):
                    st.markdown(f"**Header:** {chunk.get('header', 'N/A')}")
                    st.write(chunk["content"])
                else:
                    st.write(chunk)
else:
    st.info("💡 **Please upload a file to get started.**")

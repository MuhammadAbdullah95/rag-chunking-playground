# 📑 Text Chunking Playground for RAG

An interactive Streamlit-based playground for experimenting with multiple text chunking strategies for RAG (Retrieval-Augmented Generation) pipelines.

This tool allows you to upload documents (TXT, PDF, DOCX, Markdown) and analyze how different chunking strategies split text into chunks, making it easier to compare, visualize, and optimize for your use case.

## 🚀 Features

### Multiple Chunking Strategies
- Fixed-size (with overlap)
- Recursive (paragraphs → sentences → fallback)
- Document-based (structure-aware: headers, sections)
- Semantic (sentence embeddings + cosine similarity)
- LLM-based (Gemini-powered intelligent chunking)

### Interactive Playground
- Upload `.txt`, `.pdf`, `.docx`, or `.md` files
- Adjust parameters like `chunk_size`, `overlap`, and `semantic threshold`
- Compare strategies side by side

### User-Friendly Interface
- Preview original text
- Expandable chunk results
- First 20 chunks displayed for quick inspection

### Caching for Speed
- Sentence-Transformer model is loaded once
- Gemini client cached for efficiency

---

## 📂 Project Structure
```
chunking-playground/
│── app.py              # Streamlit main app (your code)
│── requirements.txt    # Python dependencies
│── README.md           # Project documentation
│── .env.example        # Example env file
```

---

## ⚡ Quickstart

1️⃣ Clone the repository
```bash
git clone https://github.com/MuhammadAbdullah95/rag-chunking-playground.git
cd chunking-playground
```

2️⃣ Create a virtual environment (with uv)
```bash
uv venv
.venv\Scripts\activate   # Windows
source .venv/bin/activate   # Linux / Mac
```

3️⃣ Install dependencies
```bash
uv sync
```

4️⃣ Set up environment variables

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=<your_gemini_api_key>
```

▶️ Run the App
```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501)

---

## 📊 Chunking Strategies

| Strategy       | Description                                      | Best For                   |
|----------------|--------------------------------------------------|-----------------------------|
| Fixed-size     | Equal-sized chunks with optional overlap         | Simple, consistent processing |
| Recursive      | Hierarchical splitting at natural boundaries     | General-purpose text        |
| Document-based | Structure-aware (headers, sections)              | Structured documents (reports, papers) |
| Semantic       | Uses embeddings & similarity thresholds          | Context preservation        |
| LLM-based      | Gemini generates coherent chunks                 | Complex, nuanced content    |

---

## 🔧 Example Usage
1. Upload a `.pdf` research paper.
2. Select **Semantic Chunking** with threshold `0.7`.
3. Compare with **Fixed-size Chunking** (`chunk_size=500`, `overlap=50`).
4. Expand results to inspect chunk boundaries.

---

## 📦 Requirements
Main libraries used:
- `streamlit`
- `python-dotenv`
- `PyPDF2`
- `sentence-transformers`
- `scikit-learn`
- `google-generativeai`
- `docx2txt`

---

## 🛠 Development Notes
- **Caching** is enabled for:
  - Sentence Transformer model (`@st.cache_resource`)
  - Gemini Client (`@st.cache_resource`)
  - Semantic & LLM chunking (`@st.cache_data`)

- **Fallbacks** ensure:
  - If a strategy fails → it falls back to fixed-size chunking.
  - Document-based chunking defaults to fixed-size if no headers are found.

---

## 📜 License
This project is open-source under the **MIT License**.

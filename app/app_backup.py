import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import ollama


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Financial Research Assistant",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

# app.py is inside:
# AI-Financial-Research-Assistant/app/
#
# parent       -> app/
# parent.parent -> AI-Financial-Research-Assistant/

BASE_DIR = Path(__file__).resolve().parent.parent

CHUNKS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "annual_report_chunks.csv"
)

EMBEDDINGS_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "annual_report_embeddings.npy"
)


# ============================================================
# 3. CHECK REQUIRED FILES
# ============================================================

if not CHUNKS_PATH.exists():

    st.error(
        f"""
        ❌ Chunks file was not found.

        Expected location:

        {CHUNKS_PATH}
        """
    )

    st.stop()


if not EMBEDDINGS_PATH.exists():

    st.error(
        f"""
        ❌ Embeddings file was not found.

        Expected location:

        {EMBEDDINGS_PATH}
        """
    )

    st.stop()


# ============================================================
# 4. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


model = load_embedding_model()


# ============================================================
# 5. LOAD FINANCIAL REPORT DATA
# ============================================================

@st.cache_data
def load_financial_data():

    chunks = pd.read_csv(
        CHUNKS_PATH
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    return chunks, embeddings


chunks_df, embeddings = load_financial_data()


# ============================================================
# 6. VALIDATE DATA
# ============================================================

if len(chunks_df) != len(embeddings):

    st.error(
        f"""
        ❌ Data mismatch.

        Number of chunks: {len(chunks_df)}

        Number of embeddings: {len(embeddings)}

        These numbers must be equal.
        """
    )

    st.stop()


# ============================================================
# 7. SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    top_k=5,
    min_score=0.35
):

    """
    Search the financial report using
    semantic similarity.
    """

    # Convert user question into an embedding
    query_embedding = model.encode(
        query
    )

    # Compare question with all document chunks
    scores = cosine_similarity(
        [query_embedding],
        embeddings
    )[0]

    # Get highest scoring chunks
    top_indices = np.argsort(
        scores
    )[-top_k:][::-1]

    results = []

    for rank, idx in enumerate(
        top_indices,
        start=1
    ):

        score = float(
            scores[idx]
        )

        # Ignore weak matches
        if score < min_score:
            continue

        row = chunks_df.iloc[idx]

        results.append(
            {
                "rank": rank,

                "page": int(
                    row["page"]
                ),

                "chunk": int(
                    row["chunk"]
                ),

                "score": score,

                "text": str(
                    row["text"]
                )
            }
        )

    return results


# ============================================================
# 8. BUILD CONTEXT
# ============================================================

def build_context(results):

    """
    Convert retrieved chunks into
    context for the LLM.
    """

    context_parts = []

    for result in results:

        context_parts.append(
            f"""
SOURCE {result['rank']}

Page: {result['page']}
Chunk: {result['chunk']}
Similarity Score: {result['score']:.4f}

Text:
{result['text']}
"""
        )

    return "\n".join(
        context_parts
    )


# ============================================================
# 9. ASK LOCAL LLM
# ============================================================

def ask_financial_assistant(
    query
):

    """
    Retrieve relevant report sections
    and ask Llama 3.2 to answer the question.
    """

    # --------------------------------------------------------
    # Semantic retrieval
    # --------------------------------------------------------

    results = semantic_search(
        query=query,
        top_k=5,
        min_score=0.35
    )

    # --------------------------------------------------------
    # No relevant information
    # --------------------------------------------------------

    if not results:

        return (
            "I could not find relevant information "
            "in the financial report.",
            []
        )

    # --------------------------------------------------------
    # Build context
    # --------------------------------------------------------

    context = build_context(
        results
    )

    # --------------------------------------------------------
    # Prompt
    # --------------------------------------------------------

    prompt = f"""
You are an AI Financial Research Assistant.

Your job is to answer questions about the
financial report provided in the context.

IMPORTANT RULES:

1. Use ONLY the provided financial report context.
2. Do NOT invent information.
3. Do NOT invent financial numbers.
4. Do NOT use outside knowledge.
5. If the answer cannot be found in the context,
   clearly say that the information was not found.
6. Mention the relevant page number when possible.
7. Give a concise and professional answer.
8. If multiple sources provide useful information,
   combine them carefully.
9. Preserve financial numbers accurately.

FINANCIAL REPORT CONTEXT:

{context}


USER QUESTION:

{query}


ANSWER:
"""

    # --------------------------------------------------------
    # Send prompt to Ollama
    # --------------------------------------------------------

    try:

        response = ollama.chat(
            model="llama3.2:3b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

    except Exception as e:

        return (
            f"""
            ❌ Could not connect to Ollama.

            Make sure Ollama is running and that
            the model llama3.2:3b is installed.

            Error:
            {e}
            """,
            results
        )

    # --------------------------------------------------------
    # Extract answer
    # --------------------------------------------------------

    answer = response[
        "message"
    ][
        "content"
    ]

    return (
        answer,
        results
    )


# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Assistant Settings"
    )

    st.write(
        "### Model"
    )

    st.info(
        "Llama 3.2 3B"
    )

    st.write(
        "### Embedding Model"
    )

    st.info(
        "all-MiniLM-L6-v2"
    )

    st.write(
        "### Search Method"
    )

    st.info(
        "Semantic Search"
    )

    st.write(
        "### Knowledge Source"
    )

    st.info(
        "Financial Annual Report"
    )

    st.divider()

    st.write(
        "### System"
    )

    st.success(
        "RAG system ready"
    )


# ============================================================
# 11. MAIN TITLE
# ============================================================

st.title(
    "📊 AI Financial Research Assistant"
)

st.markdown(
    """
Ask questions about the financial report
using natural language.

The assistant uses **Retrieval-Augmented Generation (RAG)**
to find relevant information before generating an answer.
"""
)


# ============================================================
# 12. SYSTEM INFORMATION
# ============================================================

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Document Chunks",
        len(chunks_df)
    )

with col2:

    st.metric(
        "Embedding Vectors",
        len(embeddings)
    )

with col3:

    st.metric(
        "LLM",
        "Llama 3.2 3B"
    )


st.divider()


# ============================================================
# 13. QUESTION INPUT
# ============================================================

query = st.text_input(
    "💬 Ask a financial question",
    placeholder=(
        "Example: How many branches does the bank have?"
    )
)


# ============================================================
# 14. ASK BUTTON
# ============================================================

ask_button = st.button(
    "🔍 Ask Financial Assistant",
    type="primary"
)


# ============================================================
# 15. PROCESS QUESTION
# ============================================================

if ask_button:

    if not query.strip():

        st.warning(
            "⚠️ Please enter a question first."
        )

    else:

        # ----------------------------------------------------
        # Show spinner
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Searching the financial report..."
        ):

            answer, sources = (
                ask_financial_assistant(
                    query
                )
            )

        # ----------------------------------------------------
        # Display answer
        # ----------------------------------------------------

        st.subheader(
            "💡 Answer"
        )

        st.write(
            answer
        )

        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        if sources:

            st.divider()

            st.subheader(
                "📚 Retrieved Sources"
            )

            for source in sources:

                page = source[
                    "page"
                ]

                chunk = source[
                    "chunk"
                ]

                score = source[
                    "score"
                ]

                with st.expander(
                    f"📄 Page {page} | "
                    f"Chunk {chunk} | "
                    f"Similarity {score:.4f}"
                ):

                    st.write(
                        source["text"]
                    )


# ============================================================
# 16. EXAMPLE QUESTIONS
# ============================================================

st.divider()

st.subheader(
    "💡 Example Questions"
)

examples = [
    "How many branches does the bank have?",
    "What was the bank's profit?",
    "What were the bank's total assets?",
    "What was the bank's revenue?",
    "What was the bank's capital?",
]

for example in examples:

    st.write(
        f"• {example}"
    )


# ============================================================
# 17. FOOTER
# ============================================================

st.divider()

st.caption(
    "AI Financial Research Assistant | "
    "RAG + Sentence Transformers + Ollama"
)

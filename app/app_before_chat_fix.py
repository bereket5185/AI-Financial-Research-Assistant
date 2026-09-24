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
    page_icon="🏦",
    layout="wide"
)


# ============================================================
# 2. PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

CHUNKS_PATH = DATA_DIR / "annual_report_chunks.csv"
EMBEDDINGS_PATH = DATA_DIR / "annual_report_embeddings.npy"


# ============================================================
# 3. CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 36px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666;
        margin-bottom: 25px;
    }

    .metric-card {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #ddd;
        background-color: #ffffff;
        text-align: center;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 20px;
        margin-bottom: 15px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 4. CHECK DATA FILES
# ============================================================

if not CHUNKS_PATH.exists():

    st.error(
        f"Chunks file not found:\n\n{CHUNKS_PATH}"
    )

    st.stop()


if not EMBEDDINGS_PATH.exists():

    st.error(
        f"Embeddings file not found:\n\n{EMBEDDINGS_PATH}"
    )

    st.stop()


# ============================================================
# 5. LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    return model


embedding_model = load_embedding_model()


# ============================================================
# 6. LOAD FINANCIAL DATA
# ============================================================

@st.cache_data
def load_financial_data():

    df = pd.read_csv(
        CHUNKS_PATH
    )

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    return df, embeddings


df, embeddings = load_financial_data()


# ============================================================
# 7. VALIDATE DATA
# ============================================================

if len(df) != len(embeddings):

    st.error(
        "The number of chunks and embeddings do not match."
    )

    st.stop()


# ============================================================
# 8. SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    top_k=5,
    min_score=0.30
):

    query_embedding = embedding_model.encode(
        [query]
    )

    scores = cosine_similarity(
        query_embedding,
        embeddings
    )[0]

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:

        score = float(scores[index])

        if score < min_score:
            continue

        row = df.iloc[index]

        results.append(
            {
                "text": row["text"],
                "page": row.get("page", "Unknown"),
                "chunk": row.get("chunk", "Unknown"),
                "score": score
            }
        )

    return results


# ============================================================
# 9. BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(results):

        context_parts.append(
            f"""
SOURCE {i + 1}

Page: {result['page']}
Chunk: {result['chunk']}

Content:
{result['text']}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# 10. ASK FINANCIAL ASSISTANT
# ============================================================

def ask_financial_assistant(question):

    results = semantic_search(
        question,
        top_k=5,
        min_score=0.30
    )

    if not results:

        return (
            "I could not find sufficiently relevant "
            "information in the annual report.",
            []
        )

    context = build_context(
        results
    )

    prompt = f"""
You are an AI Financial Research Assistant.

Answer the user's question using ONLY the
financial report information provided below.

IMPORTANT RULES:

1. Do not invent financial numbers.
2. Do not use outside knowledge.
3. If the information is not available,
   say that it was not found in the report.
4. Explain the answer clearly.
5. When possible, mention the relevant page.
6. Preserve financial units such as million,
   billion, ETB, USD, or percentages.
7. If multiple figures are relevant, explain them.

FINANCIAL REPORT CONTEXT:

{context}

USER QUESTION:

{question}

Provide a concise professional financial answer.
"""

    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    return answer, results


# ============================================================
# 11. SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🏦 Research Assistant")

    st.markdown("---")

    st.subheader("System")

    st.write(
        "🤖 LLM: llama3.2:3b"
    )

    st.write(
        "🧠 Embeddings: all-MiniLM-L6-v2"
    )

    st.write(
        "🔎 Search: Semantic Search"
    )

    st.write(
        "📚 Source: Annual Report"
    )

    st.markdown("---")

    st.subheader("Example Questions")

    st.write(
        "• What was the total asset?"
    )

    st.write(
        "• How many branches did the bank have?"
    )

    st.write(
        "• What was the net profit?"
    )

    st.write(
        "• What was the capital adequacy ratio?"
    )

    st.write(
        "• What were the major risks?"
    )



# ============================================================
# 12. CHAT MEMORY
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []


if "last_sources" not in st.session_state:
    st.session_state.last_sources = []
# ============================================================
# 12. HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🏦 AI Financial Research Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Intelligent analysis of financial reports using '
    'Semantic Search + Local LLM'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# 13. DASHBOARD METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "📄 Report Chunks",
        f"{len(df):,}"
    )


with col2:

    st.metric(
        "🧠 Embeddings",
        f"{len(embeddings):,}"
    )


with col3:

    st.metric(
        "🤖 AI Model",
        "Llama 3.2"
    )


with col4:

    st.metric(
        "🔎 Search",
        "Semantic"
    )


st.markdown("---")


# ============================================================
# 14. CHAT INTERFACE
# ============================================================

st.markdown(
    '<div class="section-title">💬 Financial Research Chat</div>',
    unsafe_allow_html=True
)


# Display previous conversation
for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# Chat input
question = st.chat_input(
    "Ask a financial question..."
)


if question:

    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)


    # Save user question
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )


    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching the annual report..."
        ):

            try:

                answer, results = ask_financial_assistant(
                    question
                )

                st.markdown(answer)

                # Save sources
                st.session_state.last_sources = results

                # Save assistant answer
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as e:

                error_message = (
                    f"An error occurred: {e}"
                )

                st.error(error_message)

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": error_message
                    }
                )


# ============================================================
# 15. SOURCES
# ============================================================

if st.session_state.last_sources:

    st.markdown(
        '<div class="section-title">📚 Sources</div>',
        unsafe_allow_html=True
    )

    for i, result in enumerate(
        st.session_state.last_sources
    ):

        with st.expander(
            f"Source {i + 1} — "
            f"Page {result['page']} — "
            f"Similarity {result['score']:.3f}"
        ):

            st.write(
                f"**Page:** {result['page']}"
            )

            st.write(
                f"**Chunk:** {result['chunk']}"
            )

            st.write(
                f"**Similarity:** "
                f"{result['score']:.3f}"
            )

            st.markdown("---")

            st.write(
                result["text"]
            )
# ============================================================
# 15. ANALYZE BUTTON
# ============================================================

if st.button(
    "🔍 Analyze",
    type="primary",
    use_container_width=True
):

    if not question.strip():

        st.warning(
            "Please enter a financial question."
        )

    else:

        with st.spinner(
            "Searching the annual report and generating analysis..."
        ):

            try:

                answer, results = ask_financial_assistant(
                    question
                )

                st.markdown(
                    '<div class="section-title">🤖 AI Analysis</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    answer
                )

                # ==================================================
                # SOURCES
                # ==================================================

                st.markdown(
                    '<div class="section-title">📚 Sources</div>',
                    unsafe_allow_html=True
                )

                for i, result in enumerate(results):

                    with st.expander(
                        f"Source {i + 1} — "
                        f"Page {result['page']} — "
                        f"Similarity {result['score']:.3f}"
                    ):

                        st.write(
                            f"**Page:** {result['page']}"
                        )

                        st.write(
                            f"**Chunk:** {result['chunk']}"
                        )

                        st.write(
                            f"**Similarity:** "
                            f"{result['score']:.3f}"
                        )

                        st.markdown("---")

                        st.write(
                            result["text"]
                        )

            except Exception as e:

                st.error(
                    f"An error occurred:\n\n{e}"
                )


# ============================================================
# 16. EXAMPLE QUESTIONS
# ============================================================

st.markdown("---")

st.markdown(
    '<div class="section-title">💡 Try These Questions</div>',
    unsafe_allow_html=True
)

example_col1, example_col2, example_col3 = st.columns(3)


with example_col1:

    st.info(
        "📊 What was the total asset?"
    )


with example_col2:

    st.info(
        "🏦 How many branches did the bank have?"
    )


with example_col3:

    st.info(
        "💰 What was the net profit?"
    )

st.markdown("---")

if st.button(
    "🗑️ Clear Conversation",
    use_container_width=True
):

    st.session_state.messages = []

    st.session_state.last_sources = []

    st.rerun()
# ============================================================
# 17. FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "AI Financial Research Assistant • "
    "Powered by Semantic Search + Ollama"
)
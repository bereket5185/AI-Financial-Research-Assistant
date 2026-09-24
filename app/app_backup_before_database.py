import json
import re
from pathlib import Path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import ollama


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Financial Research Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    /* ── Google Font ───────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    /* ── Root Variables ────────────────────────── */
    :root {
        --bg-primary:    #0d1117;
        --bg-secondary:  #161b22;
        --bg-card:       #1c2333;
        --border:        #30363d;
        --accent:        #2563eb;
        --accent-light:  #3b82f6;
        --accent-glow:   rgba(37,99,235,0.18);
        --text-primary:  #e6edf3;
        --text-secondary:#8b949e;
        --text-muted:    #6e7681;
        --green:         #22c55e;
        --red:           #ef4444;
        --yellow:        #eab308;
    }

    /* ── Base ──────────────────────────────────── */
    html, body {
        font-family: 'Inter', sans-serif !important;
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }

    /* Apply font to all Streamlit generated CSS classes */
    [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }

    /* Main content text */
    .stMarkdown, .stMarkdown p, .stMarkdown li,
    .stText, p, li, span:not([class*="stMetric"]) {
        color: #e6edf3;
    }

    .main { background-color: var(--bg-primary) !important; }

    .block-container {
        padding-top: 1.8rem !important;
        padding-bottom: 2.5rem !important;
        max-width: 1480px !important;
    }

    /* ── Sidebar ───────────────────────────────── */
    section[data-testid="stSidebar"] {
        background-color: var(--bg-secondary) !important;
        border-right: 1px solid var(--border) !important;
    }

    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] .stSelectbox label {
        color: var(--text-secondary) !important;
        font-size: 0.82rem !important;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        font-size: 0.78rem !important;
        letter-spacing: 0.08em !important;
        text-transform: uppercase !important;
    }

    /* ── Header ────────────────────────────────── */
    .main-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 1.4rem 1.8rem;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 100%);
        border: 1px solid var(--border);
        border-radius: 14px;
        margin-bottom: 1.4rem;
    }

    .main-header-icon {
        font-size: 2.4rem;
        line-height: 1;
    }

    .main-header-text h1 {
        font-size: 1.7rem !important;
        font-weight: 700 !important;
        color: var(--text-primary) !important;
        margin: 0 !important;
        letter-spacing: -0.02em;
    }

    .main-header-text p {
        font-size: 0.87rem !important;
        color: var(--text-secondary) !important;
        margin: 0.15rem 0 0 0 !important;
    }

    /* ── KPI Cards ─────────────────────────────── */
    [data-testid="stMetric"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 1rem 1.2rem !important;
        transition: border-color 0.2s;
    }

    [data-testid="stMetric"]:hover {
        border-color: var(--accent-light) !important;
    }

    [data-testid="stMetricLabel"] {
        color: var(--text-secondary) !important;
        font-size: 0.78rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }

    [data-testid="stMetricValue"] {
        color: var(--text-primary) !important;
        font-size: 1.5rem !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricDelta"] {
        font-size: 0.82rem !important;
    }

    /* ── Chat Messages ─────────────────────────── */
    [data-testid="stChatMessageContent"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        padding: 1rem 1.2rem !important;
        font-size: 0.93rem !important;
        line-height: 1.65 !important;
    }

    /* ── Chat Input ────────────────────────────── */
    [data-testid="stChatInput"] textarea,
    [data-testid="stChatInput"] textarea:focus,
    [data-testid="stChatInput"] textarea:active {
        background-color: #1c2333 !important;
        color: #e6edf3 !important;
        caret-color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        font-size: 0.93rem !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #6e7681 !important;
        opacity: 1 !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.18) !important;
    }

    /* ── All text inputs & textareas (general) ── */
    input[type="text"],
    input[type="search"],
    input[type="number"],
    textarea,
    .stTextInput input,
    .stTextArea textarea {
        background-color: #1c2333 !important;
        color: #e6edf3 !important;
        caret-color: #e6edf3 !important;
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
    }

    input[type="text"]::placeholder,
    textarea::placeholder,
    .stTextInput input::placeholder,
    .stTextArea textarea::placeholder {
        color: #6e7681 !important;
        opacity: 1 !important;
    }

    /* ── Expanders ─────────────────────────────── */
    [data-testid="stExpander"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        margin-bottom: 0.6rem !important;
    }

    [data-testid="stExpander"] summary {
        color: var(--text-secondary) !important;
        font-size: 0.87rem !important;
        font-weight: 500 !important;
    }

    [data-testid="stExpander"] summary:hover {
        color: var(--accent-light) !important;
    }

    /* ── Buttons ───────────────────────────────── */
    .stButton > button {
        background: var(--accent) !important;
        color: #fff !important;
        border: none !important;
        border-radius: 8px !important;
        font-size: 0.87rem !important;
        font-weight: 500 !important;
        padding: 0.5rem 1.2rem !important;
        transition: background 0.2s, transform 0.1s;
    }

    .stButton > button:hover {
        background: var(--accent-light) !important;
        transform: translateY(-1px);
    }

    .stButton > button:active {
        transform: translateY(0px);
    }

    /* ── Download Buttons ──────────────────────── */
    .stDownloadButton > button {
        background: var(--bg-card) !important;
        color: var(--accent-light) !important;
        border: 1px solid var(--accent) !important;
        border-radius: 8px !important;
        font-size: 0.87rem !important;
        font-weight: 500 !important;
    }

    .stDownloadButton > button:hover {
        background: var(--accent-glow) !important;
    }

    /* ── Checkboxes & Selects ──────────────────── */
    .stCheckbox label {
        color: var(--text-secondary) !important;
        font-size: 0.87rem !important;
    }

    .stMultiSelect [data-baseweb="tag"] {
        background-color: var(--accent) !important;
        border-radius: 4px !important;
    }

    /* MultiSelect & Selectbox inputs */
    .stMultiSelect [data-baseweb="input"],
    .stMultiSelect input,
    .stSelectbox input {
        background-color: #1c2333 !important;
        color: #e6edf3 !important;
        caret-color: #e6edf3 !important;
    }

    /* Dropdown list items */
    [data-baseweb="menu"] li,
    [data-baseweb="select"] div[role="option"],
    [data-baseweb="popover"] li {
        background-color: #1c2333 !important;
        color: #e6edf3 !important;
    }

    [data-baseweb="menu"] li:hover,
    [data-baseweb="popover"] li:hover {
        background-color: #2d3748 !important;
    }

    /* ── DataFrames ────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border) !important;
        border-radius: 10px !important;
        overflow: hidden;
    }

    /* ── Dividers ──────────────────────────────── */
    hr {
        border-color: var(--border) !important;
        margin: 1.2rem 0 !important;
    }

    /* ── Status / Alerts ───────────────────────── */
    .stSuccess, [data-testid="stAlert"][data-baseweb="notification"] {
        background: rgba(34,197,94,0.12) !important;
        border: 1px solid rgba(34,197,94,0.4) !important;
        border-radius: 8px !important;
        color: var(--green) !important;
    }

    .stInfo {
        background: rgba(37,99,235,0.1) !important;
        border: 1px solid rgba(37,99,235,0.3) !important;
        border-radius: 8px !important;
    }

    .stWarning {
        background: rgba(234,179,8,0.1) !important;
        border: 1px solid rgba(234,179,8,0.35) !important;
        border-radius: 8px !important;
    }

    /* ── Section Headers ───────────────────────── */
    h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 600 !important;
        letter-spacing: -0.01em;
    }

    /* ── Source Cards ──────────────────────────── */
    .source-card {
        background: var(--bg-card);
        padding: 0.9rem 1.1rem;
        margin-bottom: 0.6rem;
        border-radius: 10px;
        border: 1px solid var(--border);
        font-size: 0.87rem;
        line-height: 1.6;
    }

    /* ── Spinner ───────────────────────────────── */
    .stSpinner > div {
        border-top-color: var(--accent-light) !important;
    }

    /* ── Code Blocks ───────────────────────────── */
    code, pre {
        background: #0d1117 !important;
        border: 1px solid var(--border) !important;
        border-radius: 6px !important;
        color: #a5d6ff !important;
        font-size: 0.85rem !important;
    }

    /* ── Caption ───────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-size: 0.78rem !important;
    }

    /* ── Selectbox ─────────────────────────────── */
    .stSelectbox > div > div {
        background: var(--bg-card) !important;
        border-color: var(--border) !important;
        color: var(--text-primary) !important;
        border-radius: 8px !important;
    }

    /* ── Scrollbar ─────────────────────────────── */
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--bg-primary); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: #484f58; }

    /* ── Badge pill ────────────────────────────── */
    .badge {
        display: inline-block;
        padding: 0.15rem 0.55rem;
        border-radius: 99px;
        font-size: 0.73rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .badge-blue  { background: rgba(37,99,235,0.2);  color: #60a5fa; border: 1px solid rgba(37,99,235,0.4); }
    .badge-green { background: rgba(34,197,94,0.15); color: #4ade80; border: 1px solid rgba(34,197,94,0.35); }
    .badge-gray  { background: rgba(110,118,129,0.2);color: #8b949e; border: 1px solid rgba(110,118,129,0.35); }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data" / "processed"

# New Step 7 files
MULTI_CHUNKS_PATH = DATA_DIR / "annual_reports_chunks.csv"
MULTI_EMBEDDINGS_PATH = DATA_DIR / "annual_reports_embeddings.npy"

# Old files kept as fallback
OLD_CHUNKS_PATH = DATA_DIR / "annual_report_chunks.csv"
OLD_EMBEDDINGS_PATH = DATA_DIR / "annual_report_embeddings.npy"

OLLAMA_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_chunks():
    if MULTI_CHUNKS_PATH.exists():
        path = MULTI_CHUNKS_PATH
    elif OLD_CHUNKS_PATH.exists():
        path = OLD_CHUNKS_PATH
    else:
        st.error(
            "No processed chunks file was found.\n\n"
            f"Expected:\n{MULTI_CHUNKS_PATH}\n\n"
            f"or:\n{OLD_CHUNKS_PATH}\n\n"
            "Run preprocess_multi_documents.py first."
        )
        st.stop()

    df = pd.read_csv(path)

    # Backward compatibility with the old single-document dataset
    if "document" not in df.columns:
        df["document"] = "annual_report.pdf"

    if "year" not in df.columns:
        df["year"] = np.nan

    if "page" not in df.columns:
        df["page"] = "Unknown"

    if "chunk" not in df.columns:
        df["chunk"] = "Unknown"

    if "text" not in df.columns:
        st.error("The chunks CSV does not contain a 'text' column.")
        st.stop()

    return df, path


@st.cache_data
def load_embeddings():
    if MULTI_EMBEDDINGS_PATH.exists():
        path = MULTI_EMBEDDINGS_PATH
    elif OLD_EMBEDDINGS_PATH.exists():
        path = OLD_EMBEDDINGS_PATH
    else:
        st.error(
            "No embeddings file was found.\n\n"
            f"Expected:\n{MULTI_EMBEDDINGS_PATH}\n\n"
            f"or:\n{OLD_EMBEDDINGS_PATH}\n\n"
            "Run preprocess_multi_documents.py first."
        )
        st.stop()

    return np.load(path), path


@st.cache_resource
def load_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL)


chunks_df, chunks_file_used = load_chunks()
embeddings, embeddings_file_used = load_embeddings()
embedding_model = load_embedding_model()


# ============================================================
# DATA VALIDATION
# ============================================================

if len(chunks_df) != len(embeddings):
    st.error(
        "The number of chunks does not match the number of embeddings.\n\n"
        f"Chunks: {len(chunks_df):,}\n"
        f"Embeddings: {len(embeddings):,}\n\n"
        "Run preprocess_multi_documents.py again."
    )
    st.stop()


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_json_response(text):
    text = str(text).strip()
    text = re.sub(r"^```json\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^```\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_financial_value(value, unit=""):
    """
    Convert a financial value into a numeric base value.

    Examples:
        2.5 billion -> 2500000000
        800 million -> 800000000
        25 thousand -> 25000
    """
    number = safe_float(value)

    if number is None:
        return None

    unit_text = str(unit).lower().replace(",", "").strip()

    multipliers = {
        "thousand": 1_000,
        "thousands": 1_000,
        "k": 1_000,
        "million": 1_000_000,
        "millions": 1_000_000,
        "m": 1_000_000,
        "billion": 1_000_000_000,
        "billions": 1_000_000_000,
        "bn": 1_000_000_000,
        "b": 1_000_000_000,
        "trillion": 1_000_000_000_000,
        "trillions": 1_000_000_000_000,
        "tn": 1_000_000_000_000,
        "t": 1_000_000_000_000,
    }

    for key, multiplier in multipliers.items():
        if key in unit_text:
            return number * multiplier

    return number


def format_number(value):
    if value is None:
        return "N/A"

    value = float(value)

    if abs(value) >= 1_000_000_000_000:
        return f"{value / 1_000_000_000_000:,.2f} trillion"
    if abs(value) >= 1_000_000_000:
        return f"{value / 1_000_000_000:,.2f} billion"
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:,.2f} million"
    if abs(value) >= 1_000:
        return f"{value / 1_000:,.2f} thousand"

    return f"{value:,.2f}"


def units_compatible(record_a, record_b):
    """
    Values are compatible if their explicit units are equal,
    or if one/both units are unspecified.
    """
    unit_a = str(record_a.get("unit", "")).strip().lower()
    unit_b = str(record_b.get("unit", "")).strip().lower()

    if not unit_a or unit_a in {"nan", "none"}:
        return True

    if not unit_b or unit_b in {"nan", "none"}:
        return True

    return unit_a == unit_b


def extract_year_from_period(period):
    match = re.search(r"(19|20)\d{2}", str(period))
    return int(match.group()) if match else None


def year_from_record(record):
    year = safe_float(record.get("period"))
    if year is not None:
        return int(year)

    return extract_year_from_period(record.get("period", ""))


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    query,
    top_k=5,
    min_score=0.30,
    selected_documents=None,
    selected_years=None,
):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True,
    )

    similarities = cosine_similarity(
        query_embedding,
        embeddings,
    )[0]

    selected_documents = selected_documents or []
    selected_years = selected_years or []

    candidates = []

    for idx, score in enumerate(similarities):
        row = chunks_df.iloc[idx]

        if selected_documents:
            if str(row["document"]) not in selected_documents:
                continue

        if selected_years:
            row_year = row.get("year", np.nan)

            try:
                row_year = int(float(row_year))
            except (TypeError, ValueError):
                row_year = None

            if row_year not in selected_years:
                continue

        score = float(score)

        if score < min_score:
            continue

        candidates.append((idx, score))

    candidates.sort(key=lambda x: x[1], reverse=True)

    results = []

    for idx, score in candidates[:top_k]:
        row = chunks_df.iloc[idx]

        results.append(
            {
                "index": int(idx),
                "score": score,
                "similarity": score,
                "text": str(row["text"]),
                "document": str(row.get("document", "Unknown")),
                "year": row.get("year", "Unknown"),
                "page": row.get("page", "Unknown"),
                "chunk": row.get("chunk", "Unknown"),
            }
        )

    return results


# ============================================================
# QUERY REWRITING
# ============================================================

def rewrite_search_query(question, conversation_history=None):
    question = str(question).strip()

    if not conversation_history:
        return question

    recent_history = conversation_history[-6:]

    history_lines = []

    for message in recent_history:
        role = message.get("role", "")
        content = str(message.get("content", "")).strip()

        if content:
            history_lines.append(
                f"{role.upper()}: {content}"
            )

    if not history_lines:
        return question

    history_text = "\n".join(history_lines)

    prompt = f"""
You are a financial-report search-query rewriting system.

Rewrite the CURRENT QUESTION into one standalone search query
for an annual financial report.

Use conversation history to resolve:
- it
- they
- this
- that
- previous year
- last year
- same metric
- above figure
- same company

Rules:
1. Do not answer the question.
2. Do not invent numbers.
3. Do not invent facts.
4. Preserve the original meaning.
5. Return ONLY the rewritten search query.
6. Do not write "Search query:".

CONVERSATION:
{history_text}

CURRENT QUESTION:
{question}
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Return only one standalone search query.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        rewritten = response["message"]["content"].strip()

        if rewritten:
            return rewritten

    except Exception:
        pass

    return question


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):
    if not results:
        return ""

    context_parts = []

    for i, result in enumerate(results, start=1):
        context_parts.append(
            f"""
SOURCE {i}
Document: {result.get("document", "Unknown")}
Year: {result.get("year", "Unknown")}
Page: {result.get("page", "Unknown")}
Chunk: {result.get("chunk", "Unknown")}
Similarity: {result.get("score", 0):.4f}

{result.get("text", "")}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# STRUCTURED FINANCIAL EXTRACTION
# ============================================================

def extract_financial_data(question, context):
    if not context:
        return []

    prompt = f"""
You are a financial data extraction system.

Question:
{question}

Financial report evidence:
{context}

Extract ONLY financial figures explicitly supported by the evidence.

Return ONLY valid JSON.

Required format:
[
  {{
    "metric": "total assets",
    "value": 123456,
    "unit": "million",
    "period": "2024",
    "page": "23",
    "source": "annual_report_2024.pdf"
  }}
]

Rules:
1. Do not invent values.
2. Do not infer missing values.
3. Extract only figures visible in the evidence.
4. Keep the metric name specific.
5. Keep the reporting period.
6. Keep the unit if explicitly available.
7. page and source must come from the evidence.
8. If no financial figure is explicitly supported, return [].
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        raw = clean_json_response(
            response["message"]["content"]
        )

        data = json.loads(raw)

        if not isinstance(data, list):
            return []

        valid_records = []

        for item in data:
            if not isinstance(item, dict):
                continue

            if "metric" not in item or "value" not in item:
                continue

            valid_records.append(
                {
                    "metric": str(item.get("metric", "")),
                    "value": item.get("value"),
                    "unit": str(item.get("unit", "")),
                    "period": str(item.get("period", "")),
                    "page": str(item.get("page", "")),
                    "source": str(item.get("source", "")),
                }
            )

        return valid_records

    except Exception:
        return []


# ============================================================
# DETERMINISTIC CALCULATIONS
# ============================================================

def calculate_difference(current, previous):
    return current - previous


def calculate_percentage_change(current, previous):
    if previous == 0:
        return None

    return ((current - previous) / abs(previous)) * 100


def calculate_ratio(numerator, denominator):
    if denominator == 0:
        return None

    return numerator / denominator


def calculate_margin(profit, revenue):
    if revenue == 0:
        return None

    return (profit / revenue) * 100


def prepare_records_for_calculation(records):
    prepared = []

    for record in records:
        numeric_value = normalize_financial_value(
            record.get("value"),
            record.get("unit", ""),
        )

        if numeric_value is None:
            continue

        new_record = dict(record)
        new_record["numeric_value"] = numeric_value
        new_record["year"] = year_from_record(record)

        prepared.append(new_record)

    return prepared


def find_comparable_pair(records):
    prepared = prepare_records_for_calculation(records)

    if len(prepared) < 2:
        return None, None

    groups = {}

    for record in prepared:
        metric = str(record.get("metric", "")).strip().lower()
        groups.setdefault(metric, []).append(record)

    best_pair = None
    best_year_gap = None

    for _, group in groups.items():
        if len(group) < 2:
            continue

        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a = group[i]
                b = group[j]

                if not units_compatible(a, b):
                    continue

                ya = a.get("year")
                yb = b.get("year")

                if ya is not None and yb is not None:
                    current, previous = (
                        (a, b) if ya > yb else (b, a)
                    )
                    gap = abs(ya - yb)

                    if best_pair is None or gap < best_year_gap:
                        best_pair = (current, previous)
                        best_year_gap = gap

    return best_pair if best_pair else (None, None)


def build_structured_calculation(question, records):
    if not records:
        return None

    question_lower = question.lower()

    current, previous = find_comparable_pair(records)

    if current is None or previous is None:
        return None

    current_value = current["numeric_value"]
    previous_value = previous["numeric_value"]

    result = {
        "metric": current["metric"],
        "current_period": current.get("period", ""),
        "previous_period": previous.get("period", ""),
        "current_value": current_value,
        "previous_value": previous_value,
        "difference": calculate_difference(
            current_value,
            previous_value,
        ),
        "percentage_change": calculate_percentage_change(
            current_value,
            previous_value,
        ),
        "current_source": current.get("source", ""),
        "current_page": current.get("page", ""),
        "previous_source": previous.get("source", ""),
        "previous_page": previous.get("page", ""),
    }

    if "ratio" in question_lower:
        result["ratio"] = calculate_ratio(
            current_value,
            previous_value,
        )

    return result


# ============================================================
# RESEARCH AGENT — PLANNER
# ============================================================

def create_research_plan(question):
    prompt = f"""
You are a financial research planning assistant.

Analyze this financial question:

{question}

Create a short research plan.

Allowed task types:
- metric_lookup
- historical_lookup
- comparison
- calculation
- narrative_search

Return ONLY valid JSON:

{{
  "research_goal": "short description",
  "tasks": [
    {{
      "type": "metric_lookup",
      "description": "specific task"
    }}
  ]
}}

Rules:
1. Do not answer the question.
2. Do not invent financial values.
3. Use only tasks needed to answer the question.
4. Keep the plan concise.
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        raw = clean_json_response(
            response["message"]["content"]
        )

        plan = json.loads(raw)

        if not isinstance(plan, dict):
            raise ValueError("Invalid plan")

        if not isinstance(plan.get("tasks"), list):
            raise ValueError("Invalid tasks")

        return plan

    except Exception:
        return {
            "research_goal": question,
            "tasks": [
                {
                    "type": "metric_lookup",
                    "description": question,
                }
            ],
        }


# ============================================================
# RESEARCH AGENT — EXECUTION
# ============================================================

def execute_research_plan(
    question,
    plan,
    selected_documents=None,
    selected_years=None,
):
    research_results = []

    for task in plan.get("tasks", []):
        task_type = task.get("type", "")
        description = task.get("description", "")

        search_query = rewrite_search_query(
            description,
            None,
        )

        results = semantic_search(
            search_query,
            top_k=5,
            min_score=0.30,
            selected_documents=selected_documents,
            selected_years=selected_years,
        )

        research_results.append(
            {
                "task_type": task_type,
                "description": description,
                "search_query": search_query,
                "results": results,
            }
        )

    return research_results


def flatten_research_results(research_results):
    flattened = []
    seen = set()

    for research in research_results:
        for result in research.get("results", []):
            key = (
                result.get("document"),
                result.get("page"),
                result.get("chunk"),
            )

            if key in seen:
                continue

            seen.add(key)
            flattened.append(result)

    return flattened


def build_agent_context(research_results):
    parts = []

    counter = 1

    for research in research_results:
        parts.append(
            f"\n=== RESEARCH TASK {counter} ===\n"
            f"Type: {research.get('task_type', '')}\n"
            f"Task: {research.get('description', '')}\n"
            f"Search query: {research.get('search_query', '')}\n"
        )

        for result in research.get("results", []):
            parts.append(
                f"""
SOURCE {counter}
Document: {result.get("document", "Unknown")}
Year: {result.get("year", "Unknown")}
Page: {result.get("page", "Unknown")}
Chunk: {result.get("chunk", "Unknown")}
Similarity: {result.get("score", 0):.4f}

{result.get("text", "")}
"""
            )

            counter += 1

    return "\n".join(parts)


def synthesize_research(
    question,
    research_context,
    calculation=None,
):
    calculation_text = "No deterministic calculation was produced."

    if calculation:
        calculation_text = f"""
Metric: {calculation["metric"]}
Current period: {calculation["current_period"]}
Current value: {format_number(calculation["current_value"])}
Previous period: {calculation["previous_period"]}
Previous value: {format_number(calculation["previous_value"])}
Absolute difference: {format_number(calculation["difference"])}
Percentage change: {
    "N/A"
    if calculation["percentage_change"] is None
    else f'{calculation["percentage_change"]:.2f}%'
}
"""

    prompt = f"""
You are an AI Financial Research Assistant.

Answer the user's question using ONLY the supplied research evidence.

USER QUESTION:
{question}

RESEARCH EVIDENCE:
{research_context}

DETERMINISTIC PYTHON CALCULATION:
{calculation_text}

Rules:
1. Do not invent financial numbers.
2. Do not use outside financial information.
3. If information is missing, say so.
4. Use exact periods when available.
5. If a Python calculation is supplied, use it instead of recalculating.
6. Explain the comparison clearly.
7. Distinguish reported figures from interpretation.
8. Mention document and page sources.
9. Be professional and concise.
10. Do not claim that a source supports information it does not contain.

Format the answer with useful headings when appropriate.
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a careful financial research assistant.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return response["message"]["content"].strip()

    except Exception as e:
        return f"Error communicating with Ollama: {e}"


# ============================================================
# COMPLETE RESEARCH AGENT
# ============================================================

def run_financial_research_agent(
    question,
    conversation_history=None,
    selected_documents=None,
    selected_years=None,
):
    # 1. Rewrite the user's question using conversation context
    search_question = rewrite_search_query(
        question,
        conversation_history,
    )

    # 2. Create research plan
    plan = create_research_plan(search_question)

    # 3. Execute plan
    research_results = execute_research_plan(
        search_question,
        plan,
        selected_documents=selected_documents,
        selected_years=selected_years,
    )

    # 4. Combine evidence
    flat_results = flatten_research_results(
        research_results
    )

    research_context = build_agent_context(
        research_results
    )

    # 5. Structured extraction
    extracted_records = extract_financial_data(
        search_question,
        research_context,
    )

    # 6. Deterministic calculation
    calculation = build_structured_calculation(
        search_question,
        extracted_records,
    )

    # 7. Final synthesis
    answer = synthesize_research(
        question,
        research_context,
        calculation,
    )

    return {
        "search_question": search_question,
        "plan": plan,
        "research_results": research_results,
        "flat_results": flat_results,
        "extracted_records": extracted_records,
        "calculation": calculation,
        "answer": answer,
    }



# ============================================================
# PHASE 2 — STEP 9: FINANCIAL REPORT GENERATOR
# ============================================================

def build_research_report_data(agent_result):
    """Build structured data used by the Markdown report generator."""
    calculation = agent_result.get("calculation")
    extracted = agent_result.get("extracted_records", [])
    research_results = agent_result.get("research_results", [])
    flat_results = agent_result.get("flat_results", [])

    return {
        "question": agent_result.get("question", ""),
        "answer": agent_result.get("answer", ""),
        "search_question": agent_result.get("search_question", ""),
        "plan": agent_result.get("plan", {}),
        "extracted_records": extracted,
        "calculation": calculation,
        "research_results": research_results,
        "flat_results": flat_results,
    }


def generate_financial_report(question, report_data):
    """Generate a professional Markdown research report from agent evidence."""
    answer = report_data.get("answer", "")
    records = report_data.get("extracted_records", [])
    calculation = report_data.get("calculation")
    flat_results = report_data.get("flat_results", [])

    lines = [
        "# Financial Research Report",
        "",
        f"**Research question:** {question}",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "",
        "## 1. Executive Summary",
        "",
        answer or "No synthesized answer was produced.",
        "",
    ]

    if records:
        lines += [
            "## 2. Key Financial Figures",
            "",
            "| Metric | Value | Unit | Period | Page | Source |",
            "|---|---:|---|---|---:|---|",
        ]
        for r in records:
            lines.append(
                f"| {r.get('metric','')} | {r.get('value','')} | "
                f"{r.get('unit','')} | {r.get('period','')} | "
                f"{r.get('page','')} | {r.get('source','')} |"
            )
        lines.append("")

    if calculation:
        pct = calculation.get("percentage_change")
        pct_text = "N/A" if pct is None else f"{pct:.2f}%"
        lines += [
            "## 3. Period Comparison",
            "",
            f"- **Metric:** {calculation.get('metric','')}",
            f"- **Current period:** {calculation.get('current_period','')}",
            f"- **Current value:** {format_number(calculation.get('current_value'))}",
            f"- **Previous period:** {calculation.get('previous_period','')}",
            f"- **Previous value:** {format_number(calculation.get('previous_value'))}",
            f"- **Absolute difference:** {format_number(calculation.get('difference'))}",
            f"- **Percentage change:** {pct_text}",
            "",
        ]

    lines += [
        "## 4. Supporting Evidence",
        "",
    ]

    if flat_results:
        for i, result in enumerate(flat_results, 1):
            lines += [
                f"### Source {i}",
                f"- **Document:** {result.get('document','Unknown')}",
                f"- **Year:** {result.get('year','Unknown')}",
                f"- **Page:** {result.get('page','Unknown')}",
                f"- **Chunk:** {result.get('chunk','Unknown')}",
                f"- **Similarity:** {result.get('score',0):.4f}",
                "",
                str(result.get("text", "")).strip(),
                "",
            ]
    else:
        lines.append("No supporting evidence was retrieved.")

    lines += [
        "## 5. Methodology",
        "",
        "This report was produced using multi-document semantic retrieval, "
        "structured financial extraction, deterministic Python calculations, "
        "and a local Llama 3.2 model for synthesis.",
        "",
        "## 6. Limitations",
        "",
        "Figures and conclusions are limited to the financial-report evidence "
        "retrieved by the system. Missing or unsupported information is not "
        "filled with outside data.",
    ]

    return "\n".join(lines)


# ============================================================
# PHASE 2 — STEP 10: ADVANCED FINANCIAL AGENT TOOLS
# ============================================================

def tool_search_reports(
    query,
    top_k=5,
    selected_documents=None,
    selected_years=None,
):
    """Tool: search the indexed annual reports."""
    return semantic_search(
        query,
        top_k=top_k,
        min_score=0.30,
        selected_documents=selected_documents,
        selected_years=selected_years,
    )


def tool_extract_metric(question, results):
    """Tool: extract explicitly supported financial figures."""
    context = build_context(results)
    return extract_financial_data(question, context)


def tool_compare_periods(question, records):
    """Tool: compare periods deterministically."""
    return build_structured_calculation(question, records)


def tool_calculate_change(current, previous):
    """Tool: deterministic financial change calculation."""
    difference = calculate_difference(current, previous)
    percentage = calculate_percentage_change(current, previous)
    return {
        "current": current,
        "previous": previous,
        "difference": difference,
        "percentage_change": percentage,
    }


def tool_find_explanation(query, selected_documents=None, selected_years=None):
    """Tool: retrieve narrative evidence that may explain a financial change."""
    explanation_queries = [
        query,
        f"reasons for {query}",
        f"factors affecting {query}",
        f"management discussion {query}",
    ]

    all_results = []
    seen = set()

    for q in explanation_queries:
        results = tool_search_reports(
            q,
            top_k=3,
            selected_documents=selected_documents,
            selected_years=selected_years,
        )
        for result in results:
            key = (
                result.get("document"),
                result.get("page"),
                result.get("chunk"),
            )
            if key not in seen:
                seen.add(key)
                all_results.append(result)

    all_results.sort(
        key=lambda x: float(x.get("score", 0)),
        reverse=True,
    )
    return all_results[:8]


def classify_advanced_tool_need(question):
    """Use Llama to decide which specialized tools are required."""
    prompt = f"""
You are a financial research tool planner.

Classify the user's question into one or more tools.

Allowed tools:
- search_reports: retrieve relevant financial report evidence
- extract_metric: extract explicit financial figures
- compare_periods: compare the same metric across periods
- calculate_change: calculate a numeric difference or percentage change
- find_explanation: find narrative evidence explaining a change

Return ONLY valid JSON:
{{
  "tools": ["search_reports", "extract_metric"],
  "reason": "short reason"
}}

Rules:
1. Always include search_reports.
2. Include extract_metric for financial figures.
3. Include compare_periods for year/period comparisons.
4. Include calculate_change for difference or percentage questions.
5. Include find_explanation when the user asks why, reason, driver, factor,
   explanation, cause, or major change.
6. Do not answer the question.
7. Do not invent facts.
8. Return only JSON.

QUESTION:
{question}
"""
    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Return only valid JSON.",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        data = json.loads(
            clean_json_response(response["message"]["content"])
        )

        allowed = {
            "search_reports",
            "extract_metric",
            "compare_periods",
            "calculate_change",
            "find_explanation",
        }

        tools = [
            tool for tool in data.get("tools", [])
            if tool in allowed
        ]

        if "search_reports" not in tools:
            tools.insert(0, "search_reports")

        return {
            "tools": tools,
            "reason": str(data.get("reason", "")),
        }

    except Exception:
        # Safe deterministic fallback.
        q = question.lower()
        tools = ["search_reports"]

        if any(
            word in q
            for word in [
                "asset", "revenue", "profit", "income", "loan",
                "deposit", "capital", "expense", "balance", "figure",
                "amount", "financial",
            ]
        ):
            tools.append("extract_metric")

        if any(
            word in q
            for word in [
                "compare", "previous year", "last year", "between",
                "change", "growth", "from 2023", "from 2024",
            ]
        ):
            tools.append("compare_periods")

        if any(
            word in q
            for word in [
                "percentage", "percent", "difference", "increase",
                "decrease", "growth rate",
            ]
        ):
            tools.append("calculate_change")

        if any(
            word in q
            for word in [
                "why", "reason", "reasons", "driver", "drivers",
                "factor", "factors", "explain", "explanation",
            ]
        ):
            tools.append("find_explanation")

        return {
            "tools": list(dict.fromkeys(tools)),
            "reason": "Fallback rule-based tool selection.",
        }


def execute_advanced_tools(
    question,
    conversation_history=None,
    selected_documents=None,
    selected_years=None,
):
    """
    Advanced agent execution:
    1. Rewrite the question.
    2. Select specialized tools.
    3. Search reports.
    4. Extract figures.
    5. Compare/calculate.
    6. Search for explanations when requested.
    """
    search_question = rewrite_search_query(
        question,
        conversation_history,
    )

    tool_plan = classify_advanced_tool_need(search_question)
    tools = tool_plan.get("tools", [])

    search_results = tool_search_reports(
        search_question,
        top_k=7,
        selected_documents=selected_documents,
        selected_years=selected_years,
    )

    extracted_records = []
    if "extract_metric" in tools or "compare_periods" in tools:
        extracted_records = tool_extract_metric(
            search_question,
            search_results,
        )

    calculation = None
    if "compare_periods" in tools or "calculate_change" in tools:
        calculation = tool_compare_periods(
            search_question,
            extracted_records,
        )

    explanation_results = []
    if "find_explanation" in tools:
        explanation_results = tool_find_explanation(
            search_question,
            selected_documents=selected_documents,
            selected_years=selected_years,
        )

    # If a calculation tool is explicitly required but the structured
    # extractor did not find a pair, do not fabricate one.
    if (
        "calculate_change" in tools
        and calculation is None
        and len(extracted_records) >= 2
    ):
        prepared = prepare_records_for_calculation(extracted_records)
        if len(prepared) >= 2:
            current = prepared[-1]["numeric_value"]
            previous = prepared[-2]["numeric_value"]
            calculation = tool_calculate_change(
                current,
                previous,
            )
            calculation["metric"] = prepared[-1].get("metric", "")
            calculation["current_period"] = prepared[-1].get("period", "")
            calculation["previous_period"] = prepared[-2].get("period", "")

    # Combine evidence while avoiding duplicate sources.
    combined_results = []
    seen = set()

    for result in search_results + explanation_results:
        key = (
            result.get("document"),
            result.get("page"),
            result.get("chunk"),
        )
        if key in seen:
            continue
        seen.add(key)
        combined_results.append(result)

    combined_results.sort(
        key=lambda x: float(x.get("score", 0)),
        reverse=True,
    )

    return {
        "search_question": search_question,
        "tool_plan": tool_plan,
        "tools_used": tools,
        "search_results": search_results,
        "explanation_results": explanation_results,
        "combined_results": combined_results[:12],
        "extracted_records": extracted_records,
        "calculation": calculation,
    }


def synthesize_advanced_research(
    question,
    tool_result,
):
    """Final synthesis using evidence and deterministic calculations."""
    context = build_context(tool_result.get("combined_results", []))
    calculation = tool_result.get("calculation")
    extracted = tool_result.get("extracted_records", [])
    tools_used = tool_result.get("tools_used", [])

    calculation_text = "No deterministic calculation was produced."

    if calculation:
        if "difference" in calculation:
            pct = calculation.get("percentage_change")
            calculation_text = f"""
Metric: {calculation.get("metric", "")}
Current period: {calculation.get("current_period", "")}
Current value: {format_number(calculation.get("current", calculation.get("current_value")))}
Previous period: {calculation.get("previous_period", "")}
Previous value: {format_number(calculation.get("previous", calculation.get("previous_value")))}
Absolute difference: {format_number(calculation.get("difference"))}
Percentage change: {
    "N/A" if pct is None else f"{pct:.2f}%"
}
"""
        else:
            calculation_text = str(calculation)

    prompt = f"""
You are an advanced AI Financial Research Assistant.

Answer the user's question using ONLY the supplied report evidence.

USER QUESTION:
{question}

TOOLS USED:
{", ".join(tools_used)}

EXTRACTED FINANCIAL FIGURES:
{json.dumps(extracted, indent=2, ensure_ascii=False)}

DETERMINISTIC CALCULATION:
{calculation_text}

REPORT EVIDENCE:
{context}

RULES:
1. Do not invent financial numbers or facts.
2. Do not use outside financial information.
3. If evidence is missing, say so clearly.
4. Prefer extracted figures over guessed values.
5. Prefer deterministic Python calculations over mental arithmetic.
6. If explanation evidence is available, distinguish reported facts
   from interpretation.
7. Mention document, year, and page when useful.
8. Do not claim that a source supports information it does not contain.
9. Give a professional, concise financial research answer.
10. If multiple periods are involved, state the periods explicitly.

Use headings and bullets when they improve clarity.
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful advanced financial research "
                        "assistant. Use only supplied evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )
        return response["message"]["content"].strip()

    except Exception as e:
        return f"Error communicating with Ollama: {e}"


def run_advanced_financial_agent(
    question,
    conversation_history=None,
    selected_documents=None,
    selected_years=None,
):
    tool_result = execute_advanced_tools(
        question,
        conversation_history=conversation_history,
        selected_documents=selected_documents,
        selected_years=selected_years,
    )

    answer = synthesize_advanced_research(
        question,
        tool_result,
    )

    return {
        "question": question,
        "answer": answer,
        **tool_result,
        "flat_results": tool_result["combined_results"],
    }



# ============================================================
# SIMPLE RAG FALLBACK
# ============================================================

def ask_financial_assistant(
    question,
    conversation_history=None,
    selected_documents=None,
    selected_years=None,
):
    search_query = rewrite_search_query(
        question,
        conversation_history,
    )

    results = semantic_search(
        search_query,
        top_k=5,
        min_score=0.30,
        selected_documents=selected_documents,
        selected_years=selected_years,
    )

    if not results:
        return (
            "I could not find sufficiently relevant information "
            "in the available financial reports.",
            [],
            search_query,
        )

    context = build_context(results)

    prompt = f"""
You are an AI Financial Research Assistant.

Answer using ONLY the financial report context below.

Question:
{question}

Context:
{context}

Rules:
1. Do not invent numbers.
2. Do not use outside information.
3. If information is unavailable, say so.
4. Mention relevant document and page.
5. Keep the answer professional and concise.
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
        )

        answer = response["message"]["content"].strip()

    except Exception as e:
        answer = f"Error communicating with Ollama: {e}"

    return answer, results, search_query


# ============================================================
# ANALYTICS
# ============================================================

def prepare_financial_dataframe(records):
    rows = []

    for record in records:
        numeric_value = normalize_financial_value(
            record.get("value"),
            record.get("unit", ""),
        )

        if numeric_value is None:
            continue

        rows.append(
            {
                "Metric": record.get("metric", ""),
                "Value": numeric_value,
                "Unit": record.get("unit", ""),
                "Period": record.get("period", ""),
                "Year": year_from_record(record),
                "Page": record.get("page", ""),
                "Source": record.get("source", ""),
            }
        )

    return pd.DataFrame(rows)


def create_financial_chart(records):
    df = prepare_financial_dataframe(records)

    if df.empty:
        return None, None

    metric_groups = df.groupby("Metric")

    # Use the first metric that has at least two time points
    for metric, group in metric_groups:
        group = group.copy()

        if len(group) < 2:
            continue

        if group["Year"].notna().any():
            group = group.sort_values("Year")

        fig, ax = plt.subplots(figsize=(10, 5))

        x_labels = group["Period"].astype(str).tolist()
        y_values = group["Value"].tolist()

        ax.plot(
            x_labels,
            y_values,
            marker="o",
        )

        ax.set_title(f"Trend: {metric}")
        ax.set_xlabel("Period")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.25)

        plt.xticks(rotation=30)
        plt.tight_layout()

        return fig, metric

    return None, None



# ============================================================
# PHASE 2 — STEP 12: FINANCIAL INTELLIGENCE DASHBOARD
# ============================================================

def metric_alias(text):
    """Normalize metric names so related records can be grouped."""
    text = str(text or "").strip().lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)

    aliases = {
        "total assets": "Total Assets",
        "assets total": "Total Assets",
        "total asset": "Total Assets",
        "total liabilities": "Total Liabilities",
        "liabilities total": "Total Liabilities",
        "total liability": "Total Liabilities",
        "total equity": "Equity",
        "shareholders equity": "Equity",
        "shareholder equity": "Equity",
        "total revenue": "Revenue",
        "revenue": "Revenue",
        "net profit": "Net Profit",
        "profit after tax": "Net Profit",
        "profit for the year": "Net Profit",
        "net income": "Net Profit",
        "total loans": "Loans",
        "gross loans": "Loans",
        "loans and advances": "Loans",
        "total deposits": "Deposits",
        "customer deposits": "Deposits",
    }

    return aliases.get(text, str(text).title())


def prepare_dashboard_dataframe(records):
    """Create a clean analytical dataframe from extracted financial records."""
    rows = []

    for record in records or []:
        numeric_value = normalize_financial_value(
            record.get("value"),
            record.get("unit", ""),
        )

        if numeric_value is None:
            continue

        year = year_from_record(record)

        rows.append(
            {
                "Metric": metric_alias(record.get("metric", "")),
                "Original Metric": record.get("metric", ""),
                "Value": numeric_value,
                "Reported Value": record.get("value", ""),
                "Unit": record.get("unit", ""),
                "Period": record.get("period", ""),
                "Year": year,
                "Page": record.get("page", ""),
                "Source": record.get("source", ""),
            }
        )

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    if "Year" in df.columns:
        df["Year"] = pd.to_numeric(
            df["Year"],
            errors="coerce",
        )

    return df


def get_metric_series(df, metric):
    """Return one metric ordered chronologically."""
    if df is None or df.empty:
        return pd.DataFrame()

    subset = df[df["Metric"] == metric].copy()

    if subset.empty:
        return subset

    subset = subset.drop_duplicates(
        subset=["Metric", "Period", "Year", "Value"]
    )

    if subset["Year"].notna().any():
        subset = subset.sort_values(
            ["Year", "Period"],
            na_position="last",
        )
    else:
        subset = subset.sort_values("Period")

    return subset


def dashboard_kpi_value(df, metric):
    """Return the latest available value for a metric."""
    series = get_metric_series(df, metric)

    if series.empty:
        return None

    return float(series.iloc[-1]["Value"])


def calculate_series_growth(series):
    """Calculate period-to-period changes for a metric series."""
    if series is None or series.empty:
        return pd.DataFrame()

    rows = []

    previous_value = None
    previous_period = None

    for _, row in series.iterrows():
        current_value = float(row["Value"])

        difference = None
        percentage = None

        if previous_value is not None:
            difference = current_value - previous_value

            if previous_value != 0:
                percentage = (
                    difference / abs(previous_value)
                ) * 100

        rows.append(
            {
                "Metric": row["Metric"],
                "Period": row["Period"],
                "Year": row["Year"],
                "Value": current_value,
                "Previous Period": previous_period,
                "Previous Value": previous_value,
                "Change": difference,
                "% Change": percentage,
            }
        )

        previous_value = current_value
        previous_period = row["Period"]

    return pd.DataFrame(rows)


def create_metric_trend_chart(df, metric):
    """Create a trend chart for the selected financial metric."""
    series = get_metric_series(df, metric)

    if len(series) < 2:
        return None

    fig, ax = plt.subplots(figsize=(11, 5))

    labels = series["Period"].astype(str).tolist()
    values = series["Value"].astype(float).tolist()

    ax.plot(
        labels,
        values,
        marker="o",
        linewidth=2,
    )

    ax.set_title(f"{metric} Trend")
    ax.set_xlabel("Reporting Period")
    ax.set_ylabel("Normalized Value")
    ax.grid(True, alpha=0.25)

    plt.xticks(rotation=30)
    plt.tight_layout()

    return fig


def build_growth_table(df, metric):
    """Return a readable period-growth table."""
    series = get_metric_series(df, metric)
    growth = calculate_series_growth(series)

    if growth.empty:
        return growth

    display = growth[
        [
            "Period",
            "Value",
            "Previous Period",
            "Previous Value",
            "Change",
            "% Change",
        ]
    ].copy()

    return display


def calculate_financial_ratios(df):
    """
    Calculate ratios only when the required metrics are explicitly available.
    Values are normalized to a common numeric base before calculation.
    """
    results = []

    latest = {}

    if df is None or df.empty:
        return pd.DataFrame()

    for metric in df["Metric"].dropna().unique():
        series = get_metric_series(df, metric)
        if not series.empty:
            latest[metric] = float(series.iloc[-1]["Value"])

    # Profit Margin = Net Profit / Revenue
    if "Net Profit" in latest and "Revenue" in latest:
        revenue = latest["Revenue"]
        if revenue != 0:
            results.append(
                {
                    "Ratio": "Profit Margin",
                    "Value": (
                        latest["Net Profit"] / revenue
                    ) * 100,
                    "Unit": "%",
                    "Basis": "Net Profit / Revenue",
                }
            )

    # Loan-to-Deposit Ratio = Loans / Deposits
    if "Loans" in latest and "Deposits" in latest:
        deposits = latest["Deposits"]
        if deposits != 0:
            results.append(
                {
                    "Ratio": "Loan-to-Deposit Ratio",
                    "Value": (
                        latest["Loans"] / deposits
                    ) * 100,
                    "Unit": "%",
                    "Basis": "Loans / Deposits",
                }
            )

    # ROA = Net Profit / Total Assets
    if "Net Profit" in latest and "Total Assets" in latest:
        assets = latest["Total Assets"]
        if assets != 0:
            results.append(
                {
                    "Ratio": "Return on Assets (ROA)",
                    "Value": (
                        latest["Net Profit"] / assets
                    ) * 100,
                    "Unit": "%",
                    "Basis": "Net Profit / Total Assets",
                }
            )

    return pd.DataFrame(results)


def build_financial_csv(df):
    """Prepare dashboard data for CSV download."""
    if df is None or df.empty:
        return ""

    export_df = df.copy()

    return export_df.to_csv(
        index=False
    )


def dashboard_summary_text(df):
    """Create a compact summary for the local LLM explanation."""
    if df is None or df.empty:
        return "No structured financial figures are currently available."

    parts = []

    for metric in sorted(df["Metric"].dropna().unique()):
        series = get_metric_series(df, metric)

        if series.empty:
            continue

        latest = series.iloc[-1]

        text = (
            f"{metric}: latest period "
            f"{latest.get('Period', '')}, "
            f"value {format_number(latest.get('Value'))}."
        )

        if len(series) >= 2:
            previous = series.iloc[-2]
            difference = (
                float(latest["Value"])
                - float(previous["Value"])
            )

            pct = None
            if float(previous["Value"]) != 0:
                pct = (
                    difference
                    / abs(float(previous["Value"]))
                ) * 100

            text += (
                f" Previous period {previous.get('Period', '')}, "
                f"value {format_number(previous.get('Value'))}; "
                f"change {format_number(difference)}, "
                f"percentage change "
                f"{'N/A' if pct is None else f'{pct:.2f}%'}."
            )

        parts.append(text)

    return "\n".join(parts)


def explain_dashboard_with_llm(df, metric=None):
    """Ask Llama to explain dashboard figures using only extracted data."""
    if df is None or df.empty:
        return (
            "There are no extracted financial figures available "
            "for an AI explanation."
        )

    if metric:
        selected_df = get_metric_series(df, metric)
    else:
        selected_df = df

    if selected_df.empty:
        return (
            f"No extracted figures are available for {metric}."
        )

    records = selected_df[
        [
            "Metric",
            "Value",
            "Period",
            "Year",
            "Page",
            "Source",
        ]
    ].to_dict(orient="records")

    prompt = f"""
You are a careful financial research analyst.

Explain the financial dashboard using ONLY the extracted figures below.

Selected metric:
{metric or "All available metrics"}

Extracted figures:
{json.dumps(records, indent=2, ensure_ascii=False)}

Rules:
1. Do not invent numbers or facts.
2. Do not use outside financial information.
3. State the reporting periods explicitly.
4. If there is a change, describe the calculated direction and magnitude
   only when the supplied values support it.
5. Distinguish reported figures from interpretation.
6. Mention source/page when useful.
7. If the evidence is insufficient to explain a cause, say that the
   dashboard alone does not establish the cause.
8. Keep the explanation professional and concise.

Provide:
- Key observation
- Period comparison
- Possible interpretation supported by the supplied evidence
- Evidence limitation
"""

    try:
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a careful financial analyst. "
                        "Use only supplied evidence."
                    ),
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
        )

        return response["message"]["content"].strip()

    except Exception as e:
        return f"Error communicating with Ollama: {e}"


def render_financial_intelligence_dashboard(records):
    """Render the complete Step 12 dashboard."""
    if not records:
        st.info(
            "Ask a financial question first. "
            "The dashboard will appear when structured financial "
            "figures have been extracted."
        )
        return

    df = prepare_dashboard_dataframe(records)

    if df.empty:
        st.warning(
            "No numeric financial figures were extracted from the "
            "current research evidence."
        )
        return

    st.markdown("---")
    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.8rem;">'
        '<span style="font-size:1.4rem;">📊</span>'
        '<span style="font-size:1.2rem;font-weight:700;color:#e6edf3;">Financial Intelligence Dashboard</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    metrics = sorted(
        df["Metric"].dropna().unique().tolist()
    )

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------
    preferred_metrics = [
        "Total Assets",
        "Loans",
        "Deposits",
        "Net Profit",
    ]

    kpi_metrics = [
        metric
        for metric in preferred_metrics
        if metric in metrics
    ]

    if not kpi_metrics:
        kpi_metrics = metrics[:4]

    kpi_columns = st.columns(
        max(1, min(4, len(kpi_metrics)))
    )

    for index, metric in enumerate(kpi_metrics[:4]):
        value = dashboard_kpi_value(
            df,
            metric,
        )

        with kpi_columns[index]:
            st.metric(
                metric,
                format_number(value),
            )

    # --------------------------------------------------------
    # DATA OVERVIEW
    # --------------------------------------------------------
    st.markdown("### 📋 Extracted Financial Data")

    overview_df = df[
        [
            "Metric",
            "Value",
            "Unit",
            "Period",
            "Year",
            "Page",
            "Source",
        ]
    ].copy()

    st.dataframe(
        overview_df,
        use_container_width=True,
        hide_index=True,
    )

    # --------------------------------------------------------
    # METRIC SELECTION
    # --------------------------------------------------------
    st.markdown("### 📈 Metric Trend Analysis")

    selected_metric = st.selectbox(
        "Select financial metric",
        metrics,
        index=0,
    )

    series = get_metric_series(
        df,
        selected_metric,
    )

    if len(series) >= 2:

        chart = create_metric_trend_chart(
            df,
            selected_metric,
        )

        if chart is not None:
            st.pyplot(
                chart,
                use_container_width=True,
            )
            plt.close(chart)

        growth_df = build_growth_table(
            df,
            selected_metric,
        )

        if not growth_df.empty:
            st.markdown(
                "#### Period Growth Analysis"
            )

            st.dataframe(
                growth_df,
                use_container_width=True,
                hide_index=True,
            )

    else:
        st.info(
            f"{selected_metric} has fewer than two reporting "
            "periods in the extracted data, so a trend cannot "
            "be calculated."
        )

    # --------------------------------------------------------
    # RATIOS
    # --------------------------------------------------------
    ratios_df = calculate_financial_ratios(df)

    if not ratios_df.empty:
        st.markdown("---")
        st.markdown("### 🧮 Financial Ratios")

        st.dataframe(
            ratios_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Ratios are calculated only when the required "
            "financial figures are explicitly available in "
            "the extracted report evidence."
        )

    # --------------------------------------------------------
    # AI EXPLANATION
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("### 🤖 AI Dashboard Explanation")

    explanation_key = (
        f"dashboard_explanation_{selected_metric}"
    )

    if st.button(
        f"Explain {selected_metric} trend",
        use_container_width=False,
        key=explanation_key,
    ):
        with st.spinner(
            "🤖 Llama is analyzing the selected financial metric..."
        ):
            explanation = explain_dashboard_with_llm(
                df,
                selected_metric,
            )

        st.session_state.dashboard_explanation = explanation

    if st.session_state.get(
        "dashboard_explanation",
        "",
    ):
        st.markdown(
            st.session_state.dashboard_explanation
        )

    # --------------------------------------------------------
    # EXPORT
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📥 Export Dashboard Data")

    csv_data = build_financial_csv(df)

    st.download_button(
        "⬇️ Download Financial Figures CSV",
        data=csv_data,
        file_name="financial_dashboard_data.csv",
        mime="text/csv",
        use_container_width=True,
    )

    # --------------------------------------------------------
    # DATASET SUMMARY
    # --------------------------------------------------------
    st.markdown("---")
    st.markdown("### 📌 Dashboard Summary")

    summary_col1, summary_col2, summary_col3 = st.columns(3)

    with summary_col1:
        st.metric(
            "Financial Figures",
            len(df),
        )

    with summary_col2:
        st.metric(
            "Financial Metrics",
            df["Metric"].nunique(),
        )

    with summary_col3:
        st.metric(
            "Reporting Periods",
            df["Period"].nunique(),
        )

    with st.expander(
        "View normalized analytical data",
        expanded=False,
    ):
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_search_query" not in st.session_state:
    st.session_state.last_search_query = ""

if "last_results" not in st.session_state:
    st.session_state.last_results = []

if "last_plan" not in st.session_state:
    st.session_state.last_plan = None

if "last_extracted_records" not in st.session_state:
    st.session_state.last_extracted_records = []

if "last_calculation" not in st.session_state:
    st.session_state.last_calculation = None

if "last_research_results" not in st.session_state:
    st.session_state.last_research_results = []

if "last_tool_plan" not in st.session_state:
    st.session_state.last_tool_plan = None

if "last_tools_used" not in st.session_state:
    st.session_state.last_tools_used = []

if "last_report" not in st.session_state:
    st.session_state.last_report = ""

if "dashboard_explanation" not in st.session_state:
    st.session_state.dashboard_explanation = ""

if "dashboard_metric" not in st.session_state:
    st.session_state.dashboard_metric = ""

if "dashboard_history" not in st.session_state:
    st.session_state.dashboard_history = []



# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.5rem;">
            <span style="font-size:1.6rem;">📈</span>
            <div>
                <div style="font-size:1rem;font-weight:700;color:#e6edf3;">Financial Assistant</div>
                <div style="font-size:0.73rem;color:#8b949e;">AI-powered research</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("**⚙️ SYSTEM**")

    st.markdown(
        '<div style="display:flex;align-items:center;gap:0.5rem;margin:0.4rem 0;">'
        '<span style="color:#22c55e;font-size:0.8rem;">●</span>'
        '<span style="font-size:0.84rem;color:#8b949e;">Local LLM Connected</span>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-size:0.8rem;color:#8b949e;margin:0.2rem 0;">Model: '
        f'<span style="color:#60a5fa;font-weight:600;">{OLLAMA_MODEL}</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.8rem;color:#8b949e;margin:0.2rem 0 0.8rem 0;">Embeddings: '
        f'<span style="color:#60a5fa;font-weight:600;">{EMBEDDING_MODEL}</span></div>',
        unsafe_allow_html=True,
    )

    st.markdown("---")

    st.markdown("**🤖 RESEARCH MODE**")

    use_agent = st.checkbox(
        "Financial Research Agent",
        value=True,
        help=(
            "The agent plans multiple research tasks, "
            "retrieves evidence, extracts figures and "
            "uses Python for calculations."
        ),
    )

    use_advanced_agent = st.checkbox(
        "Advanced Financial Agent",
        value=True,
        help=(
            "The advanced agent selects specialized tools for "
            "report search, metric extraction, period comparison, "
            "calculations and explanation search."
        ),
    )

    use_dashboard = st.checkbox(
        "Financial Intelligence Dashboard",
        value=True,
        help=(
            "Show KPIs, financial trends, growth analysis, "
            "ratios, AI explanations and CSV export."
        ),
    )

    st.markdown("---")

    st.markdown("**📂 DOCUMENT FILTERS**")

    documents = sorted(
        chunks_df["document"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    selected_documents = st.multiselect(
        "Documents",
        documents,
        default=[],
    )

    valid_years = []

    for value in chunks_df["year"].dropna().unique():
        try:
            valid_years.append(int(float(value)))
        except (TypeError, ValueError):
            pass

    valid_years = sorted(set(valid_years), reverse=True)

    selected_years = st.multiselect(
        "Years",
        valid_years,
        default=[],
    )

    st.markdown("---")

    st.markdown("**📊 DATASET**")

    stat_col1, stat_col2 = st.columns(2)
    with stat_col1:
        st.markdown(
            f'<div style="background:#1c2333;border:1px solid #30363d;border-radius:8px;'
            f'padding:0.6rem 0.8rem;text-align:center;">'
            f'<div style="font-size:1.2rem;font-weight:700;color:#e6edf3;">{len(chunks_df):,}</div>'
            f'<div style="font-size:0.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.05em;">Chunks</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    with stat_col2:
        st.markdown(
            f'<div style="background:#1c2333;border:1px solid #30363d;border-radius:8px;'
            f'padding:0.6rem 0.8rem;text-align:center;">'
            f'<div style="font-size:1.2rem;font-weight:700;color:#e6edf3;">{len(documents):,}</div>'
            f'<div style="font-size:0.72rem;color:#8b949e;text-transform:uppercase;letter-spacing:0.05em;">Docs</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if valid_years:
        st.markdown(
            f'<div style="font-size:0.8rem;color:#8b949e;margin-top:0.5rem;">'
            f'📅 Coverage: <span style="color:#e6edf3;font-weight:600;">'
            f'{min(valid_years)} – {max(valid_years)}</span></div>',
            unsafe_allow_html=True,
        )

    st.markdown("---")

    if st.button(
        "🗑️ Clear Conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.last_search_query = ""
        st.session_state.last_results = []
        st.session_state.last_plan = None
        st.session_state.last_extracted_records = []
        st.session_state.last_calculation = None
        st.session_state.last_research_results = []
        st.session_state.last_tool_plan = None
        st.session_state.last_tools_used = []
        st.session_state.last_report = ""
        st.session_state.dashboard_explanation = ""
        st.session_state.dashboard_metric = ""
        st.session_state.dashboard_history = []
        st.rerun()


# ============================================================
# HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <div class="main-header-icon">📈</div>
        <div class="main-header-text">
            <h1>AI Financial Research Assistant</h1>
            <p>Multi-document semantic search · Llama 3.2 · Structured extraction · Deterministic analytics</p>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Report Chunks",
        f"{len(chunks_df):,}",
    )

with col2:
    st.metric(
        "Documents Indexed",
        f"{len(documents):,}",
    )

with col3:
    st.metric(
        "LLM Engine",
        "Llama 3.2",
    )

with col4:
    st.metric(
        "Conversation Turns",
        len(st.session_state.messages),
    )

st.markdown("---")


# ============================================================
# CHAT DISPLAY
# ============================================================

for message in st.session_state.messages:

    role = message.get("role")
    content = message.get("content", "")

    if role == "user":
        with st.chat_message("user"):
            st.markdown(content)

    elif role == "assistant":
        with st.chat_message("assistant"):
            st.markdown(content)


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a financial research question..."
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if question:

    # Display user message
    with st.chat_message("user"):
        st.markdown(question)

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # Previous conversation excludes current question
    previous_messages = (
        st.session_state.messages[:-1]
    )

    with st.chat_message("assistant"):

        if use_agent:

            with st.spinner(
                "🤖 Research agent is planning and analyzing..."
            ):
                agent_result = run_financial_research_agent(
                    question,
                    conversation_history=previous_messages,
                    selected_documents=selected_documents,
                    selected_years=selected_years,
                )

            answer = agent_result["answer"]

            st.session_state.last_search_query = (
                agent_result["search_question"]
            )

            st.session_state.last_results = (
                agent_result["flat_results"]
            )

            st.session_state.last_plan = (
                agent_result["plan"]
            )

            st.session_state.last_extracted_records = (
                agent_result["extracted_records"]
            )

            st.session_state.last_calculation = (
                agent_result["calculation"]
            )

            st.session_state.last_research_results = (
                agent_result["research_results"]
            )

        else:

            with st.spinner(
                "🔎 Searching financial reports..."
            ):
                answer, results, search_query = (
                    ask_financial_assistant(
                        question,
                        previous_messages,
                        selected_documents=selected_documents,
                        selected_years=selected_years,
                    )
                )

            st.session_state.last_search_query = search_query
            st.session_state.last_results = results
            st.session_state.last_plan = None
            st.session_state.last_extracted_records = []
            st.session_state.last_calculation = None
            st.session_state.last_research_results = []
            st.session_state.last_tool_plan = None
            st.session_state.last_tools_used = []
            st.session_state.last_report = ""

        # Main answer
        st.markdown(answer)

        # --------------------------------------------------------
        # Search query
        # --------------------------------------------------------

        if st.session_state.last_search_query:

            with st.expander(
                "🔎 Search query used",
                expanded=False,
            ):
                st.code(
                    st.session_state.last_search_query,
                    language="text",
                )

        # --------------------------------------------------------
        # Research plan
        # --------------------------------------------------------

        if use_agent and st.session_state.last_plan:

            with st.expander(
                "🧠 Research plan",
                expanded=False,
            ):
                st.json(
                    st.session_state.last_plan
                )

        # --------------------------------------------------------
        # Extracted financial figures
        # --------------------------------------------------------

        extracted = (
            st.session_state.last_extracted_records
        )

        if extracted:

            with st.expander(
                "📊 Extracted financial figures",
                expanded=False,
            ):

                extracted_df = pd.DataFrame(
                    extracted
                )

                st.dataframe(
                    extracted_df,
                    use_container_width=True,
                    hide_index=True,
                )

        # --------------------------------------------------------
        # Deterministic calculation
        # --------------------------------------------------------

        calculation = (
            st.session_state.last_calculation
        )

        if calculation:

            with st.expander(
                "🧮 Calculation details",
                expanded=False,
            ):

                c1, c2, c3 = st.columns(3)

                with c1:
                    st.metric(
                        "Current",
                        format_number(
                            calculation["current_value"]
                        ),
                    )

                with c2:
                    st.metric(
                        "Previous",
                        format_number(
                            calculation["previous_value"]
                        ),
                    )

                with c3:
                    percentage = (
                        calculation["percentage_change"]
                    )

                    st.metric(
                        "% Change",
                        (
                            "N/A"
                            if percentage is None
                            else f"{percentage:.2f}%"
                        ),
                    )

                st.write(
                    f"**Absolute difference:** "
                    f"{format_number(calculation['difference'])}"
                )

                st.write(
                    f"**Metric:** {calculation['metric']}"
                )

                st.write(
                    f"**Current period:** "
                    f"{calculation['current_period']}"
                )

                st.write(
                    f"**Previous period:** "
                    f"{calculation['previous_period']}"
                )

        # --------------------------------------------------------
        # Research tasks
        # --------------------------------------------------------

        if use_agent and st.session_state.last_research_results:

            with st.expander(
                "🔬 Research tasks and evidence",
                expanded=False,
            ):

                for task_index, research in enumerate(
                    st.session_state.last_research_results,
                    start=1,
                ):

                    st.markdown(
                        f"### Task {task_index}: "
                        f"{research.get('task_type', '')}"
                    )

                    st.write(
                        research.get(
                            "description",
                            "",
                        )
                    )

                    st.code(
                        research.get(
                            "search_query",
                            "",
                        ),
                        language="text",
                    )

                    task_results = research.get(
                        "results",
                        [],
                    )

                    for source_index, result in enumerate(
                        task_results,
                        start=1,
                    ):

                        st.markdown(
                            f"""
**Source {source_index}**

📄 Document: **{result.get("document", "Unknown")}**

📅 Year: **{result.get("year", "Unknown")}**

📖 Page: **{result.get("page", "Unknown")}**

🧩 Chunk: **{result.get("chunk", "Unknown")}**

🎯 Similarity: **{result.get("score", 0):.4f}**
"""
                        )

                        st.write(
                            result.get(
                                "text",
                                "",
                            )
                        )

                        st.markdown("---")

        # --------------------------------------------------------
        # Sources
        # --------------------------------------------------------

        results = st.session_state.last_results

        if results:

            with st.expander(
                f"📚 Sources ({len(results)})",
                expanded=False,
            ):

                for i, result in enumerate(
                    results,
                    start=1,
                ):

                    st.markdown(
                        f"""
**SOURCE {i}**

📄 Document: **{result.get("document", "Unknown")}**

📅 Year: **{result.get("year", "Unknown")}**

📖 Page: **{result.get("page", "Unknown")}**

🧩 Chunk: **{result.get("chunk", "Unknown")}**

🎯 Similarity: **{result.get("score", 0):.4f}**
"""
                    )

                    st.write(
                        result.get(
                            "text",
                            "",
                        )
                    )

                    if i < len(results):
                        st.markdown("---")

        # --------------------------------------------------------
        # Financial Research Report
        # --------------------------------------------------------
        if use_advanced_agent and st.session_state.last_report:
            st.markdown("---")
            st.subheader("📄 Financial Research Report")

            with st.expander(
                "View generated report",
                expanded=False,
            ):
                st.markdown(
                    st.session_state.last_report
                )

            st.download_button(
                "⬇️ Download Markdown Report",
                data=st.session_state.last_report,
                file_name="financial_research_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
        }
    )


# ============================================================
# PHASE 2 — STEP 12: FINANCIAL INTELLIGENCE DASHBOARD
# ============================================================

if use_dashboard:
    render_financial_intelligence_dashboard(
        st.session_state.last_extracted_records
    )

# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown("---")

st.markdown("### 💡 Example Research Questions")

example_col1, example_col2 = st.columns(2)

with example_col1:
    st.markdown(
        """
        <div style="background:#1c2333;border:1px solid #30363d;border-radius:10px;padding:1rem 1.2rem;">
        <div style="font-size:0.78rem;font-weight:600;color:#8b949e;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.7rem;">Lookup & Comparison</div>
        <ul style="margin:0;padding-left:1.1rem;color:#c9d1d9;font-size:0.87rem;line-height:1.9;">
        <li>What was the bank's total assets in 2024?</li>
        <li>How did total assets change from 2023?</li>
        <li>Compare total assets between 2023 and 2024.</li>
        <li>What was the percentage change in total assets?</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )

with example_col2:
    st.markdown(
        """
        <div style="background:#1c2333;border:1px solid #30363d;border-radius:10px;padding:1rem 1.2rem;">
        <div style="font-size:0.78rem;font-weight:600;color:#8b949e;letter-spacing:0.06em;text-transform:uppercase;margin-bottom:0.7rem;">Analytics & Trends</div>
        <ul style="margin:0;padding-left:1.1rem;color:#c9d1d9;font-size:0.87rem;line-height:1.9;">
        <li>Compare profit between 2023 and 2024.</li>
        <li>What was the total loan balance in each year?</li>
        <li>Show the trend in total assets.</li>
        <li>Calculate the loan-to-deposit ratio.</li>
        <li>Explain the selected financial trend.</li>
        </ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.markdown(
    '<div style="text-align:center;color:#6e7681;font-size:0.78rem;padding:0.5rem 0;">'
    'AI Financial Research Assistant &nbsp;·&nbsp; '
    'Multi-document RAG &nbsp;·&nbsp; Llama 3.2 &nbsp;·&nbsp; '
    'Sentence Transformers &nbsp;·&nbsp; Structured Analytics'
    '</div>',
    unsafe_allow_html=True,
)

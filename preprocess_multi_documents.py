from pathlib import Path
import re

import numpy as np
import pandas as pd
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PROJECT PATHS
# ==========================================

PROJECT_DIR = Path(__file__).resolve().parent

DOCUMENTS_DIR = PROJECT_DIR / "data" / "documents"
PROCESSED_DIR = PROJECT_DIR / "data" / "processed"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ==========================================
# 2. SETTINGS
# ==========================================

CHUNK_SIZE = 1500
OVERLAP = 200

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ==========================================
# 3. FIND YEAR FROM FILE NAME
# ==========================================

def detect_year(filename):
    match = re.search(r"(19|20)\d{2}", filename)

    if match:
        return int(match.group())

    return None


# ==========================================
# 4. CLEAN TEXT
# ==========================================

def clean_text(text):
    if not text:
        return ""

    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)

    return text.strip()


# ==========================================
# 5. CREATE CHUNKS
# ==========================================

def create_chunks(text, chunk_size=1500, overlap=200):

    if not text:
        return []

    chunks = []

    start = 0
    text_length = len(text)

    while start < text_length:

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = end - overlap

    return chunks


# ==========================================
# 6. EXTRACT ALL PDF DOCUMENTS
# ==========================================

pdf_files = sorted(DOCUMENTS_DIR.glob("*.pdf"))

print("=" * 60)
print("MULTI-DOCUMENT PREPROCESSING")
print("=" * 60)

print(f"\nDocuments folder:")
print(DOCUMENTS_DIR)

print(f"\nPDF files found: {len(pdf_files)}")


if not pdf_files:

    print("\nERROR: No PDF files were found.")

    print("\nPlease put your annual reports inside:")
    print(DOCUMENTS_DIR)

    print("\nExample:")
    print("annual_report_2024.pdf")
    print("annual_report_2023.pdf")

    raise SystemExit


records = []


for pdf_path in pdf_files:

    print("\n" + "-" * 60)
    print(f"Processing: {pdf_path.name}")

    year = detect_year(pdf_path.name)

    print(f"Detected year: {year}")

    try:
        reader = PdfReader(str(pdf_path))

    except Exception as e:

        print(f"ERROR reading PDF: {e}")
        continue

    print(f"Pages: {len(reader.pages)}")

    document_chunk_count = 0

    for page_number, page in enumerate(reader.pages, start=1):

        try:
            text = page.extract_text() or ""

        except Exception as e:

            print(
                f"Warning: Could not extract page "
                f"{page_number}: {e}"
            )

            continue

        text = clean_text(text)

        if not text:
            continue

        chunks = create_chunks(
            text,
            chunk_size=CHUNK_SIZE,
            overlap=OVERLAP
        )

        for chunk_number, chunk in enumerate(chunks, start=1):

            records.append({
                "document": pdf_path.name,
                "year": year,
                "page": page_number,
                "chunk": chunk_number,
                "text": chunk
            })

            document_chunk_count += 1

    print(
        f"Chunks created from {pdf_path.name}: "
        f"{document_chunk_count}"
    )


# ==========================================
# 7. CREATE DATAFRAME
# ==========================================

df = pd.DataFrame(records)


if df.empty:

    print("\nERROR: No text chunks were created.")

    raise SystemExit


print("\n" + "=" * 60)
print("PREPROCESSING SUMMARY")
print("=" * 60)

print(f"Documents: {df['document'].nunique()}")
print(f"Total chunks: {len(df)}")


# ==========================================
# 8. SAVE CHUNKS
# ==========================================

chunks_path = (
    PROCESSED_DIR /
    "annual_reports_chunks.csv"
)

df.to_csv(
    chunks_path,
    index=False,
    encoding="utf-8-sig"
)

print(f"\nChunks saved to:")
print(chunks_path)


# ==========================================
# 9. CREATE EMBEDDINGS
# ==========================================

print("\nLoading embedding model...")

model = SentenceTransformer(EMBEDDING_MODEL)

print("Creating embeddings...")

embeddings = model.encode(
    df["text"].tolist(),
    show_progress_bar=True,
    convert_to_numpy=True
)


# ==========================================
# 10. SAVE EMBEDDINGS
# ==========================================

embeddings_path = (
    PROCESSED_DIR /
    "annual_reports_embeddings.npy"
)

np.save(
    embeddings_path,
    embeddings
)

print("\nEmbeddings saved to:")
print(embeddings_path)


# ==========================================
# 11. FINAL CHECK
# ==========================================

print("\n" + "=" * 60)
print("DONE")
print("=" * 60)

print(f"Documents processed : {df['document'].nunique()}")
print(f"Total chunks        : {len(df)}")
print(f"Embedding shape     : {embeddings.shape}")

print("\nGenerated files:")

print(chunks_path)
print(embeddings_path)
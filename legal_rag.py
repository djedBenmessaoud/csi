#!/usr/bin/env python3
"""
Legal Compliance RAG Pipeline
Fetches GDPR and CCPA legal text, builds retrievable chunks, embeds them,
and stores in ChromaDB for semantic search.
"""

import requests
from bs4 import BeautifulSoup
import re
import json
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import spacy
from pathlib import Path

# =============================================================================
# Configuration
# =============================================================================

# GitHub mirror of GDPR - EU regulations as Markdown
GDPR_URL = "https://raw.githubusercontent.com/legalize-dev/legalize-eu/main/regulations/gdpr.md"
CCPA_LEGINFO_BASE = "https://leginfo.legislature.ca.gov/faces/codes_displaySection.xhtml"

DB_PATH = "./compliance_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
MAX_CHUNK_TOKENS = 200

# =============================================================================
# Step 1 — Fetch GDPR from GitHub mirror (GDPRtEXT JSON)
# =============================================================================

GDPR_JSON_URL = "https://raw.githubusercontent.com/coolharsh55/GDPRtEXT/master/gdpr.json"


def fetch_gdpr() -> list[dict]:
    """Fetch GDPR articles from GDPRtEXT JSON."""
    articles = []

    print("Fetching GDPR from GDPRtEXT JSON...")
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(GDPR_JSON_URL, headers=headers, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print(f"Error fetching GDPR JSON: {e}")
        return []

    def extract_articles(contents: list, parent_type: str = ""):
        """Recursively extract articles from nested contents."""
        extracted = []
        for item in contents:
            item_type = item.get("type", "")

            if item_type == "article":
                art_num = item.get("number", "")
                art_title = item.get("title", "")

                # Collect all text from article points
                text_parts = []
                for point in item.get("contents", []):
                    if "text" in point:
                        text_parts.append(point["text"])
                    # Handle subpoints (a), (b), etc.
                    for sp in point.get("subpoints", []):
                        if "text" in sp:
                            text_parts.append(f"({sp.get('number', '')}) {sp['text']}")

                full_text = " ".join(text_parts)
                full_text = re.sub(r'\s+', ' ', full_text).strip()

                if len(full_text) >= 50:
                    title = f"GDPR Article {art_num}"
                    if art_title:
                        title += f" - {art_title}"

                    extracted.append({
                        "source": "gdpr",
                        "article_id": f"gdpr_art_{art_num}",
                        "article_number": int(art_num),
                        "title": title,
                        "text": full_text,
                        "text_for_embedding": f"{title}: {full_text}"
                    })

            # Recurse into sections/chapters
            elif item_type in ("section", "chapter") and "contents" in item:
                extracted.extend(extract_articles(item["contents"], item_type))

        return extracted

    # Traverse all chapters and their nested contents
    for chapter in data.get("chapters", []):
        articles.extend(extract_articles(chapter.get("contents", [])))

    return articles


def parse_gdpr(_) -> list[dict]:
    """Identity function for GDPR - already parsed in fetch."""
    return []


# =============================================================================
# Step 2 — Fetch CCPA from California leginfo
# =============================================================================

def fetch_ccpa_leginfo() -> list[dict]:
    """Scrape CCPA sections from California leginfo."""
    sections = []
    section_numbers = [
        "1798.100", "1798.105", "1798.106", "1798.110", "1798.115",
        "1798.120", "1798.121", "1798.125", "1798.130", "1798.135",
        "1798.140", "1798.145", "1798.150", "1798.155", "1798.160",
        "1798.185", "1798.190", "1798.195", "1798.199"
    ]

    headers = {"User-Agent": "Mozilla/5.0"}
    print("Fetching CCPA sections from leginfo...")

    for sec in section_numbers:
        params = {"lawCode": "CIV", "sectionNum": f"{sec}."}
        try:
            resp = requests.get(CCPA_LEGINFO_BASE, params=params, headers=headers, timeout=15)
            if resp.status_code != 200:
                print(f"  Skipping {sec}: HTTP {resp.status_code}")
                continue

            soup = BeautifulSoup(resp.text, "html.parser")
            content_div = soup.find("div", id="codeLawSectionNoHead")
            if not content_div:
                continue

            text = content_div.get_text(separator=" ", strip=True)
            text = re.sub(r"\s+", " ", text)

            if len(text) < 50:
                continue

            sections.append({
                "source": "ccpa",
                "article_id": f"ccpa_sec_{sec.replace('.', '_')}",
                "section_number": sec,
                "title": f"CCPA Section {sec}",
                "text": text,
                "text_for_embedding": f"CCPA Section {sec}: {text}"
            })
            print(f"  Fetched CCPA § {sec} ({len(text)} chars)")
        except Exception as e:
            print(f"  Error fetching {sec}: {e}")

    return sections


# =============================================================================
# Step 3 — Build Parent-Child Chunks
# =============================================================================

def load_spacy():
    """Load spaCy model for sentence splitting."""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        from spacy.cli import download
        download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 2_000_000
    return nlp


def build_chunks(articles: list[dict], nlp, max_tokens: int = MAX_CHUNK_TOKENS) -> list[dict]:
    """Split articles into paragraph/sentence chunks with parent references."""
    all_chunks = []

    for article in articles:
        text = article["text"]
        source = article["source"]
        art_id = article["article_id"]

        para_splits = re.split(r"(?<!\w)(\d+\.\s|\([a-z]\)\s|\([ivx]+\)\s)", text)

        paragraphs = []
        i = 0
        while i < len(para_splits):
            chunk = para_splits[i]
            if re.match(r"(\d+\.\s|\([a-z]\)\s|\([ivx]+\)\s)", chunk):
                if i + 1 < len(para_splits):
                    chunk = chunk + para_splits[i + 1]
                    i += 2
                else:
                    i += 1
            else:
                i += 1
            chunk = chunk.strip()
            if len(chunk) > 30:
                paragraphs.append(chunk)

        if len(paragraphs) <= 1:
            doc = nlp(text)
            paragraphs = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 30]

        for idx, para_text in enumerate(paragraphs):
            token_estimate = len(para_text) // 4

            if token_estimate > max_tokens:
                doc = nlp(para_text)
                sub_chunks = [s.text.strip() for s in doc.sents if len(s.text.strip()) > 20]
            else:
                sub_chunks = [para_text]

            for sub_idx, sub_text in enumerate(sub_chunks):
                chunk_id = f"{art_id}_p{idx}_s{sub_idx}"

                all_chunks.append({
                    "chunk_id": chunk_id,
                    "parent_id": art_id,
                    "source": source,
                    "article_number": article.get("article_number") or article.get("section_number"),
                    "title": article["title"],
                    "text": sub_text,
                    "text_for_embedding": f"{article['title']}: {sub_text}"
                })

    return all_chunks


# =============================================================================
# Step 4 — Embed with Sentence-Transformers
# =============================================================================

def embed_chunks(chunks: list[dict], model: SentenceTransformer, batch_size: int = 64) -> np.ndarray:
    """Embed all chunks using the BGE model."""
    texts = [c["text_for_embedding"] for c in chunks]

    print(f"Embedding {len(texts)} chunks...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )
    return embeddings


# =============================================================================
# Step 5 — Store in ChromaDB
# =============================================================================

def store_chunks(client, chunks: list[dict], embeddings: np.ndarray):
    """Store chunks in ChromaDB with metadata."""
    try:
        client.delete_collection("legal_chunks")
    except:
        pass

    collection = client.create_collection(
        name="legal_chunks",
        metadata={"hnsw:space": "cosine"}
    )

    ids = [c["chunk_id"] for c in chunks]
    documents = [c["text"] for c in chunks]
    metadatas = [
        {
            "source": c["source"],
            "parent_id": c["parent_id"],
            "title": c["title"],
            "article_number": str(c["article_number"])
        }
        for c in chunks
    ]
    embs = embeddings.tolist()

    batch_size = 100
    for i in range(0, len(ids), batch_size):
        collection.add(
            ids=ids[i:i+batch_size],
            embeddings=embs[i:i+batch_size],
            documents=documents[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size]
        )
        print(f"  Stored {min(i+batch_size, len(ids))}/{len(ids)}")

    return collection


# =============================================================================
# Step 6 — Build Parent Document Store
# =============================================================================

def build_parent_store(articles: list[dict], db_path: str):
    """Build and save parent article lookup store."""
    parent_store = {}
    for article in articles:
        parent_store[article["article_id"]] = {
            "title": article["title"],
            "text": article["text"],
            "source": article["source"]
        }

    Path(db_path).mkdir(parents=True, exist_ok=True)
    with open(f"{db_path}/parent_store.json", "w") as f:
        json.dump(parent_store, f, indent=2)

    print(f"Parent store: {len(parent_store)} full articles saved")
    return parent_store


# =============================================================================
# Step 7 — Verify Retrieval
# =============================================================================

def verify_retrieval(collection, model: SentenceTransformer):
    """Test retrieval with sample queries."""
    query = "What are the conditions for a data subject to request deletion of personal data?"

    query_embedding = model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        normalize_embeddings=True
    ).tolist()

    print("\n=== GDPR top matches ===")
    results_gdpr = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"source": "gdpr"},
        include=["documents", "metadatas", "distances"]
    )

    for doc, meta, dist in zip(
        results_gdpr["documents"][0],
        results_gdpr["metadatas"][0],
        results_gdpr["distances"][0]
    ):
        print(f"  [{meta['title']}] score={1-dist:.3f}")
        print(f"  {doc[:150]}...\n")

    print("=== CCPA top matches ===")
    results_ccpa = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        where={"source": "ccpa"},
        include=["documents", "metadatas", "distances"]
    )

    for doc, meta, dist in zip(
        results_ccpa["documents"][0],
        results_ccpa["metadatas"][0],
        results_ccpa["distances"][0]
    ):
        print(f"  [{meta['title']}] score={1-dist:.3f}")
        print(f"  {doc[:150]}...\n")


# =============================================================================
# Main Pipeline
# =============================================================================

def main():
    print("=" * 60)
    print("Legal Compliance RAG Pipeline")
    print("=" * 60)

    # Step 1: Fetch GDPR
    print("\n[Step 1] Fetching GDPR...")
    gdpr_articles = fetch_gdpr()
    print(f"Parsed {len(gdpr_articles)} GDPR articles")

    # Step 2: Fetch CCPA
    print("\n[Step 2] Fetching CCPA...")
    ccpa_articles = fetch_ccpa_leginfo()
    print(f"Parsed {len(ccpa_articles)} CCPA sections")

    if not gdpr_articles and not ccpa_articles:
        print("ERROR: No articles fetched. Check network/URLs.")
        return

    all_articles = gdpr_articles + ccpa_articles

    # Step 3: Build chunks
    print("\n[Step 3] Building parent-child chunks...")
    nlp = load_spacy()
    gdpr_chunks = build_chunks(gdpr_articles, nlp)
    ccpa_chunks = build_chunks(ccpa_articles, nlp)
    all_chunks = gdpr_chunks + ccpa_chunks

    print(f"GDPR: {len(gdpr_articles)} articles → {len(gdpr_chunks)} chunks")
    print(f"CCPA: {len(ccpa_articles)} sections → {len(ccpa_chunks)} chunks")
    print(f"Total chunks: {len(all_chunks)}")

    # Step 4: Embed
    print("\n[Step 4] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    embeddings = embed_chunks(all_chunks, model)
    print(f"Embedding matrix shape: {embeddings.shape}")

    # Step 5: Store in ChromaDB
    print("\n[Step 5] Storing in ChromaDB...")
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = store_chunks(client, all_chunks, embeddings)
    print(f"Collection size: {collection.count()} chunks")

    # Step 6: Build parent store
    print("\n[Step 6] Building parent document store...")
    build_parent_store(all_articles, DB_PATH)

    # Step 7: Verify
    print("\n[Step 7] Verifying retrieval...")
    verify_retrieval(collection, model)

    print("\n" + "=" * 60)
    print("Pipeline complete! Database stored in:", DB_PATH)
    print("=" * 60)


if __name__ == "__main__":
    main()

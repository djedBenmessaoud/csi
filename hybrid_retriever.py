#!/usr/bin/env python3
"""
Hybrid Retriever - Combines BM25 (lexical) + Dense (semantic) retrieval with Reciprocal Rank Fusion.
"""

import json
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import chromadb

# ── Load everything ──────────────────────────────────────────────────────────
client = chromadb.PersistentClient(path="./compliance_db")
collection = client.get_collection("legal_chunks")
model = SentenceTransformer("BAAI/bge-small-en-v1.5")

with open("./compliance_db/parent_store.json") as f:
    parent_store = json.load(f)

# ── Build BM25 index over all chunks ─────────────────────────────────────────
print("Building BM25 index...")
all_results = collection.get(include=["documents", "metadatas"])
all_docs = all_results["documents"]
all_ids = all_results["ids"]
all_metas = all_results["metadatas"]

# BM25 expects pre-tokenized text
tokenized_corpus = [doc.lower().split() for doc in all_docs]
bm25 = BM25Okapi(tokenized_corpus)
print(f"BM25 index built over {len(all_docs)} chunks")


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────
def reciprocal_rank_fusion(
    ranked_lists: list[list[str]],
    k: int = 60
) -> list[tuple[str, float]]:
    """
    Merge multiple ranked lists of chunk IDs into one score.
    k=60 is the standard constant — dampens the effect of very high ranks.
    """
    scores = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] = scores.get(doc_id, 0) + 1 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── Core hybrid retrieval function ───────────────────────────────────────────
def hybrid_retrieve(
    query: str,
    source: str,          # "gdpr" or "ccpa"
    top_k: int = 5,
    dense_k: int = 10,    # fetch more than needed before RRF merge
    bm25_k: int = 10
) -> list[dict]:
    """
    Returns top_k chunks from a specific source (gdpr/ccpa),
    fused from dense + BM25 retrieval.
    """

    # ── Dense retrieval ───────────────────────────────────────────────────────
    query_emb = model.encode(
        f"Represent this sentence for searching relevant passages: {query}",
        normalize_embeddings=True
    ).tolist()

    dense_results = collection.query(
        query_embeddings=[query_emb],
        n_results=dense_k,
        where={"source": source},
        include=["documents", "metadatas", "distances"]
    )
    dense_ranked = dense_results["ids"][0]

    # ── BM25 retrieval ────────────────────────────────────────────────────────
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Filter to only chunks from the target source
    source_mask = [
        i for i, meta in enumerate(all_metas)
        if meta["source"] == source
    ]
    source_scores = [(i, bm25_scores[i]) for i in source_mask]
    source_scores.sort(key=lambda x: x[1], reverse=True)
    bm25_ranked = [all_ids[i] for i, _ in source_scores[:bm25_k]]

    # ── RRF fusion ────────────────────────────────────────────────────────────
    fused = reciprocal_rank_fusion([dense_ranked, bm25_ranked])
    top_ids = [doc_id for doc_id, _ in fused[:top_k]]

    # ── Resolve to parent documents ───────────────────────────────────────────
    seen_parents = set()
    results = []

    for chunk_id in top_ids:
        if chunk_id not in all_ids:
            continue
        idx = all_ids.index(chunk_id)
        meta = all_metas[idx]
        parent_id = meta["parent_id"]

        if parent_id in seen_parents:
            continue  # deduplicate — multiple chunks from same article
        seen_parents.add(parent_id)

        parent = parent_store.get(parent_id, {})
        results.append({
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "source": source,
            "title": meta["title"],
            "full_text": parent.get("text", all_docs[idx]),  # full article
            "chunk_text": all_docs[idx]                       # matched fragment
        })

    return results


if __name__ == "__main__":
    # Test the hybrid retriever
    query = "What are the conditions for a data subject to request deletion?"

    print(f"\nQuery: {query}\n")

    gdpr_results = hybrid_retrieve(query, source="gdpr", top_k=3)
    ccpa_results = hybrid_retrieve(query, source="ccpa", top_k=3)

    print("=== GDPR Results ===")
    for r in gdpr_results:
        print(f"  {r['title']}")
        print(f"  {r['chunk_text'][:120]}...\n")

    print("=== CCPA Results ===")
    for r in ccpa_results:
        print(f"  {r['title']}")
        print(f"  {r['chunk_text'][:120]}...\n")

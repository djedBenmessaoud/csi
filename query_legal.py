#!/usr/bin/env python3
"""
Query utility for the Legal Compliance RAG database.
Supports querying GDPR and/or CCPA with semantic search.
"""

import argparse
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

DB_PATH = "./compliance_db"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"


def load_resources():
    """Load the ChromaDB collection and parent store."""
    client = chromadb.PersistentClient(path=DB_PATH)
    collection = client.get_collection("legal_chunks")

    with open(f"{DB_PATH}/parent_store.json", "r") as f:
        parent_store = json.load(f)

    model = SentenceTransformer(EMBEDDING_MODEL)

    return collection, parent_store, model


def query(query_text: str, source: str = None, n_results: int = 3, model=None, collection=None, parent_store=None):
    """
    Query the legal database.

    Args:
        query_text: The search query
        source: Filter by source ("gdpr", "ccpa", or None for both)
        n_results: Number of results to return
        model: Pre-loaded embedding model
        collection: Pre-loaded ChromaDB collection
        parent_store: Pre-loaded parent document store

    Returns:
        List of (parent_article, chunk_text, score) tuples
    """
    # Embed query with BGE prefix
    query_embedding = model.encode(
        f"Represent this sentence for searching relevant passages: {query_text}",
        normalize_embeddings=True
    ).tolist()

    # Build where clause
    where_clause = {"source": source} if source else None

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where_clause,
        include=["documents", "metadatas", "distances"]
    )

    # Reconstruct full parent articles for each match
    seen_parents = set()
    output = []

    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        parent_id = meta["parent_id"]
        score = 1 - dist

        if parent_id not in seen_parents:
            seen_parents.add(parent_id)
            parent = parent_store.get(parent_id, {})
            output.append({
                "title": meta.get("title", "Unknown"),
                "source": meta.get("source", "unknown"),
                "article_number": meta.get("article_number", ""),
                "score": round(score, 3),
                "matched_chunk": doc,
                "full_text": parent.get("text", "")
            })

    return output


def main():
    parser = argparse.ArgumentParser(description="Query legal compliance database")
    parser.add_argument("query", nargs="?", help="Search query")
    parser.add_argument("--source", "-s", choices=["gdpr", "ccpa", "both"], default="both",
                        help="Filter by source (default: both)")
    parser.add_argument("--n", "-n", type=int, default=3, help="Number of results (default: 3)")
    parser.add_argument("--full", "-f", action="store_true", help="Show full article text")
    parser.add_argument("--json", "-j", action="store_true", help="Output as JSON")

    args = parser.parse_args()

    if not args.query:
        parser.print_help()
        return

    collection, parent_store, model = load_resources()

    source_filter = None if args.source == "both" else args.source

    results = query(
        args.query,
        source=source_filter,
        n_results=args.n,
        model=model,
        collection=collection,
        parent_store=parent_store
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"\nQuery: {args.query}")
        print(f"Source: {args.source} | Results: {len(results)}\n")
        print("=" * 70)

        for r in results:
            print(f"\n[{r['source'].upper()}] {r['title']}")
            print(f"Score: {r['score']}")
            print("-" * 70)

            if args.full:
                print(r["full_text"])
            else:
                print(f"Match: {r['matched_chunk'][:200]}...")

        print("\n" + "=" * 70)


if __name__ == "__main__":
    main()

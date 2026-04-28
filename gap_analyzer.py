#!/usr/bin/env python3
"""
Gap Analyzer - Uses LLM (via OpenAI API) to analyze compliance gaps between GDPR and CCPA.
"""

import os
import json
from openai import OpenAI
from hybrid_retriever import hybrid_retrieve
from compliance_queries import COMPLIANCE_AXES

# ── Configure OpenAI client ──────────────────────────────────────────────────
# Reads OPENAI_API_KEY from environment
client = OpenAI()

SYSTEM_PROMPT = """You are a legal compliance analyst specializing in data privacy law.
You will be given relevant excerpts from GDPR and CCPA regulations and a specific compliance question.

Your task is to analyze both regulations and identify:
1. What each regulation requires on this topic
2. The key differences (the compliance gap)
3. Concrete actionable recommendations for a company that must comply with BOTH

Respond ONLY with valid JSON matching this exact schema — no preamble, no markdown fences:
{
  "axis": "<compliance topic>",
  "gdpr_position": "<what GDPR requires, 2-4 sentences>",
  "ccpa_position": "<what CCPA requires, 2-4 sentences>",
  "gap_severity": "low | medium | high",
  "key_differences": ["<difference 1>", "<difference 2>", ...],
  "recommendations": ["<action 1>", "<action 2>", "<action 3>"],
  "stricter_law": "gdpr | ccpa | equivalent",
  "gdpr_articles": ["Article X", ...],
  "ccpa_sections": ["§ 1798.XXX", ...]
}"""


def analyze_axis(axis: dict) -> dict:
    """Run gap analysis for a single compliance axis."""

    # Retrieve relevant chunks from both laws
    gdpr_chunks = hybrid_retrieve(axis["query"], source="gdpr", top_k=3)
    ccpa_chunks = hybrid_retrieve(axis["query"], source="ccpa", top_k=3)

    # Build context block — full parent articles, clearly labeled
    gdpr_context = "\n\n".join([
        f"[GDPR — {c['title']}]\n{c['full_text']}"
        for c in gdpr_chunks
    ])
    ccpa_context = "\n\n".join([
        f"[CCPA — {c['title']}]\n{c['full_text']}"
        for c in ccpa_chunks
    ])

    user_prompt = f"""Compliance question: {axis['query']}

--- GDPR EXCERPTS ---
{gdpr_context}

--- CCPA EXCERPTS ---
{ccpa_context}

Analyze the gap between these two regulations on this specific question."""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.1,
        response_format={"type": "json_object"},  # native JSON mode
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ],
        max_tokens=600
    )

    raw = response.choices[0].message.content
    result = json.loads(raw)  # safe — json_object mode guarantees valid JSON
    result["axis_id"] = axis["id"]
    result["axis_label"] = axis["label"]

    # Attach the source chunks for traceability
    result["sources"] = {
        "gdpr": [c["title"] for c in gdpr_chunks],
        "ccpa": [c["title"] for c in ccpa_chunks]
    }

    return result


def run_full_gap_analysis() -> list[dict]:
    results = []
    for axis in COMPLIANCE_AXES:
        print(f"  Analyzing: {axis['label']}...")
        try:
            result = analyze_axis(axis)
            results.append(result)
            print(f"    Gap severity: {result['gap_severity']} | Stricter: {result['stricter_law']}")
        except Exception as e:
            print(f"    Failed: {e}")
            results.append({"axis_id": axis["id"], "error": str(e)})

    return results


if __name__ == "__main__":
    print("Running compliance gap analysis...\n")
    results = run_full_gap_analysis()

    with open("gap_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved {len(results)} axis results to gap_analysis_results.json")

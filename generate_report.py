#!/usr/bin/env python3
"""
Generate Report - Converts gap analysis JSON results into a clean markdown report.
"""

import json
from datetime import date

with open("gap_analysis_results.json") as f:
    results = json.load(f)

SEVERITY_EMOJI = {"high": "🔴", "medium": "🟡", "low": "🟢"}
STRICTER_LABEL = {
    "gdpr": "GDPR is stricter",
    "ccpa": "CCPA is stricter",
    "equivalent": "Roughly equivalent"
}


def generate_report(results: list[dict]) -> str:
    lines = [
        f"# GDPR vs CCPA Compliance Gap Analysis Report",
        f"Generated: {date.today().isoformat()}",
        f"Total axes analyzed: {len(results)}",
        ""
    ]

    # Summary table
    lines += ["## Executive Summary\n", "| Topic | Severity | Stricter Law |", "|-------|----------|--------------|"]
    for r in results:
        if "error" in r:
            continue
        sev = r.get("gap_severity", "?")
        lines.append(
            f"| {r['axis_label']} "
            f"| {SEVERITY_EMOJI.get(sev, '')} {sev.capitalize()} "
            f"| {STRICTER_LABEL.get(r.get('stricter_law', ''), '?')} |"
        )

    lines.append("")

    # Detail sections
    for r in results:
        if "error" in r:
            lines.append(f"## {r['axis_id']}\n⚠️ Analysis failed: {r['error']}\n")
            continue

        sev = r.get("gap_severity", "?")
        lines += [
            f"---",
            f"## {SEVERITY_EMOJI.get(sev, '')} {r['axis_label']}",
            "",
            f"**GDPR position** *(Articles: {', '.join(r.get('gdpr_articles', []))})*",
            f"> {r.get('gdpr_position', 'N/A')}",
            "",
            f"**CCPA position** *(Sections: {', '.join(r.get('ccpa_sections', []))})*",
            f"> {r.get('ccpa_position', 'N/A')}",
            "",
            "**Key differences**",
        ]
        for diff in r.get("key_differences", []):
            lines.append(f"- {diff}")

        lines += ["", "**Recommendations**"]
        for i, rec in enumerate(r.get("recommendations", []), 1):
            lines.append(f"{i}. {rec}")
        lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    report = generate_report(results)
    with open("gap_analysis_report.md", "w") as f:
        f.write(report)

    print(report)
    print("\nReport saved to gap_analysis_report.md")

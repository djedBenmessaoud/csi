# GDPR vs CCPA Compliance Gap Analyzer

A Retrieval-Augmented Generation (RAG) system that performs automated compliance gap analysis between GDPR and CCPA regulations using hybrid search (BM25 + dense embeddings) and LLM-powered synthesis.

![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Hybrid Retrieval**: Combines BM25 (lexical) + BGE embeddings (semantic) with Reciprocal Rank Fusion
- **Parent-Child Chunking**: Retrieves precise chunks while returning full article context
- **Automated Gap Analysis**: 8 predefined compliance axes analyzed by LLM
- **Structured Output**: JSON results with traceability to source articles
- **Markdown Reports**: Executive summary with severity ratings and recommendations

## Quick Start

```bash
# 1. Clone the repo
git clone <your-repo-url>
cd csi

# 2. Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or: venv\Scripts\activate  # Windows

# 3. Install dependencies
make setup
# or: pip install -r requirements.txt
#     python -m spacy download en_core_web_sm

# 4. Build the legal database
make db

# 5. Set your OpenAI API key
export OPENAI_API_KEY=sk-...

# 6. Run gap analysis and generate report
make all

# 7. View results
cat gap_analysis_report.md
```

## Project Structure

| File | Description |
|------|-------------|
| `legal_rag.py` | Pipeline: fetch, parse, chunk, embed, store GDPR/CCPA |
| `query_legal.py` | CLI for querying the legal database |
| `hybrid_retriever.py` | BM25 + dense retrieval with RRF fusion |
| `compliance_queries.py` | 8 compliance analysis axes |
| `gap_analyzer.py` | LLM-powered gap analysis engine |
| `generate_report.py` | Markdown report generator |
| `compliance_db/` | ChromaDB database (generated) |

## Compliance Axes Analyzed

1. **Right to deletion/erasure** - Conditions and exceptions
2. **Data breach notification** - Timelines and procedures  
3. **Consent requirements** - Valid consent for processing
4. **Data portability** - Rights to transfer data
5. **Right to access/know** - Disclosure requirements
6. **Opt-out of data sale** - Third-party sharing rights
7. **Children's data protections** - Minor-specific protections
8. **Penalties and enforcement** - Financial penalties

## Example Output

```markdown
## 🔴 Right to deletion / erasure

**GDPR position** *(Articles: Article 17)*
> The data subject shall have the right to obtain from the controller 
> the erasure of personal data concerning him or her without undue 
> delay where one of the following grounds applies...

**CCPA position** *(Sections: § 1798.105)*
> A consumer shall have the right to request that a business delete 
> any personal information about the consumer which the business has 
> collected from the consumer...

**Key differences**
- GDPR requires "undue delay" while CCPA allows 45 days
- GDPR has 6 specific grounds; CCPA is more general
- CCPA has more exceptions for business operations

**Recommendations**
1. Implement unified deletion request workflow
2. Default to GDPR's stricter timeline (30 days)
3. Document all exceptions applied
```

## Data Sources

- **GDPR**: [GDPRtEXT](https://github.com/coolharsh55/GDPRtEXT) - Linked data format
- **CCPA**: [California LegInfo](https://leginfo.legislature.ca.gov/) - Official text

## Requirements

- Python 3.10+
- OpenAI API key (for gap analysis)
- ~500MB disk space for database

## License

MIT License - see [LICENSE](LICENSE) for details.

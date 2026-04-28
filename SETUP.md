# Setup Guide

This guide walks you through setting up the GDPR vs CCPA Compliance Gap Analyzer on a fresh machine.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git (optional, for cloning the repo)
- OpenAI API key (for gap analysis)

## Step 1: Clone or Download

```bash
# Option A: Clone from GitHub
git clone <your-repo-url>
cd csi

# Option B: Download and extract ZIP
# Then cd into the extracted directory
```

## Step 2: Create Virtual Environment

**Important:** Dependencies are installed in a local virtual environment. The `venv/` folder is not committed to git—each user creates their own.

```bash
# Create virtual environment
python -m venv venv

# Activate on Linux/macOS
source venv/bin/activate

# Activate on Windows
venv\Scripts\activate
```

## Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

## Step 4: Download spaCy Model

```bash
python -m spacy download en_core_web_sm
```

## Step 5: Build the Legal Database

```bash
python legal_rag.py
```

This will:
- Download all 99 GDPR articles
- Download 18 key CCPA sections
- Create ~1,338 searchable chunks
- Generate embeddings and store in `compliance_db/`

Expected output:
```
Parsed 99 GDPR articles
Parsed 18 CCPA sections
Total chunks: 1338
Embedding matrix shape: (1338, 384)
Collection size: 1338 chunks
```

## Step 6: Configure API Key

```bash
# Linux/macOS
export OPENAI_API_KEY=sk-your-key-here

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-your-key-here"

# Windows (Command Prompt)
set OPENAI_API_KEY=sk-your-key-here
```

To get an API key:
1. Visit https://platform.openai.com
2. Sign in or create account
3. Go to API Keys section
4. Create new secret key

## Step 7: Run Gap Analysis

```bash
# Run the analysis (8 LLM calls, ~1-2 minutes)
python gap_analyzer.py

# Generate markdown report
python generate_report.py
```

## Step 8: View Results

```bash
# View in terminal
cat gap_analysis_report.md

# Or open in browser
open gap_analysis_report.md  # macOS
xdg-open gap_analysis_report.md  # Linux
start gap_analysis_report.md  # Windows
```

## Verification

### Test the Database

```bash
python query_legal.py "right to erasure" --source gdpr -n 2
```

Expected: Should return GDPR Article 17 with score > 0.7

### Test Hybrid Retriever

```bash
python hybrid_retriever.py
```

Expected: Should show BM25 index built over 1338 chunks and test retrieval results.

## Troubleshooting

### "No module named 'chromadb'"

```bash
pip install chromadb
```

### "spaCy model not found"

```bash
python -m spacy download en_core_web_sm
```

### "OPENAI_API_KEY not set"

Make sure you exported the key in your current shell session. Add to `~/.bashrc` or `~/.zshrc` for persistence.

### "HTTP 429 Rate Limit"

OpenAI API rate limits vary by tier. Add delays between requests or upgrade your plan.

### "ChromaDB connection error"

Delete and rebuild the database:
```bash
rm -rf compliance_db
python legal_rag.py
```

## Optional: Query the Database

```bash
# Simple queries
python query_legal.py "consent requirements"
python query_legal.py "data breach notification" --source gdpr

# Full article text
python query_legal.py "administrative fines" --source gdpr --full

# JSON output
python query_legal.py "consumer rights" -j

# More results
python query_legal.py "legal basis" -n 5
```

## File Sizes

After setup, expect:
- `compliance_db/`: ~5MB
- `requirements.txt`: <1KB
- Python scripts: ~50KB total

## Next Steps

1. Review `gap_analysis_report.md` for compliance insights
2. Customize `compliance_queries.py` for your use case
3. Integrate results into your compliance documentation
4. Re-run analysis when regulations change

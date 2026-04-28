.PHONY: setup db query analyze report clean help

# Default target
help:
	@echo "GDPR vs CCPA Compliance Gap Analyzer"
	@echo ""
	@echo "Available targets:"
	@echo "  setup    - Install dependencies and download spaCy model"
	@echo "  db       - Build the legal database (GDPR + CCPA)"
	@echo "  query    - Query the database (use QUERY= arg)"
	@echo "  analyze  - Run gap analysis (requires OPENAI_API_KEY)"
	@echo "  report   - Generate markdown report"
	@echo "  clean    - Remove generated files"
	@echo "  all      - Run full pipeline: db -> analyze -> report"
	@echo ""
	@echo "Examples:"
	@echo "  make setup"
	@echo "  make db"
	@echo "  make query QUERY=\"right to erasure\""
	@echo "  export OPENAI_API_KEY=sk-... && make all"

# Install dependencies
setup:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm
	@echo "Setup complete!"

# Build database
db:
	python legal_rag.py

# Query database
query:
ifndef QUERY
	@echo "Usage: make query QUERY=\"your search query\""
else
	python query_legal.py "$(QUERY)"
endif

# Run gap analysis
analyze:
	python gap_analyzer.py

# Generate report
report:
	python generate_report.py

# Run full pipeline
all: db analyze report
	@echo ""
	@echo "Pipeline complete! View results:"
	@echo "  cat gap_analysis_report.md"

# Clean generated files
clean:
	rm -rf compliance_db/
	rm -f gap_analysis_results.json
	rm -f gap_analysis_report.md
	rm -rf __pycache__/
	rm -rf *.egg-info
	@echo "Cleaned generated files"

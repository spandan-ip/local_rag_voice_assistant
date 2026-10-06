# Edit this list to match the documents you configured in app/config.py's SOURCES.
# "relevant_sources" must match the "source" metadata value (filename or URL)
# of documents that genuinely contain the answer to "question".
#
# Example: if SOURCES includes {"source_type": "pdf", "source": "source/company_policy.pdf"},
# then "relevant_sources" for a question about that PDF should be ["company_policy.pdf"].

EVALUATION_QUESTIONS = [
    {
        "question": "What is this document about?",
        "relevant_sources": ["sample.pdf"],
    },
]

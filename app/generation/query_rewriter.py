import logging

from langchain_core.prompts import PromptTemplate

from app.generation.llm import get_llm

logger = logging.getLogger(__name__)

_REWRITE_PROMPT = PromptTemplate.from_template(
    """Rewrite the question below into a short, keyword-rich search query,
optimized for finding relevant passages in a document search index.

- Expand abbreviations and vague phrasing where it helps matching.
- If the question compares two or more things, include ALL of the
  things being compared in the query, not just one of them.
- Do not answer the question. Output ONLY the rewritten query, nothing else.

Question: {question}

Search query:"""
)


def rewrite_query(question: str) -> str:
    """Return a search-optimized version of the question, using the LLM.

    Retrieval (FAISS/BM25) matches on wording, so a vague or narrowly
    phrased question can miss chunks that use different words for the
    same idea, or miss one side of a comparison question entirely. This
    asks the LLM to restate the question in a way more likely to match
    relevant text -- used ONLY for searching; the original question is
    still what gets shown to the user and answered.

    Falls back to the original question, unchanged, if the LLM call
    fails for any reason -- a failed rewrite should never block search.
    """
    if not question or not question.strip():
        return question

    try:
        prompt = _REWRITE_PROMPT.format(question=question)
        rewritten = get_llm().invoke(prompt).strip()
    except Exception as exc:
        logger.warning("Query rewrite failed, using original question: %s", exc)
        return question

    if not rewritten:
        return question

    logger.info("Rewrote query: %r -> %r", question, rewritten)
    return rewritten

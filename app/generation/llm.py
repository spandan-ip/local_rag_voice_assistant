import logging

from langchain_ollama import OllamaLLM

from app.config import LLM_MODEL, OLLAMA_BASE_URL, TEMPERATURE
from app.generation.prompts import RAG_PROMPT_TEMPLATE, format_context

logger = logging.getLogger(__name__)

_llm_cache: OllamaLLM | None = None


class LLMError(Exception):
    pass


def _get_llm() -> OllamaLLM:
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, temperature=TEMPERATURE)
    return _llm_cache


def get_llm() -> OllamaLLM:
    """Public accessor for the shared Ollama LLM instance -- used by
    other modules (e.g. query_rewriter.py) that need to make their own,
    differently-prompted call to the same model.
    """
    return _get_llm()


def generate_answer(question: str, context_chunks, history: str = "") -> str:
    context = format_context(context_chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(
        question=question, context=context, history=history or "(no previous conversation)"
    )

    try:
        response = _get_llm().invoke(prompt)
    except Exception as exc:
        message = str(exc).lower()
        if "connection" in message or "connect" in message:
            raise LLMError(
                f"Could not connect to Ollama at {OLLAMA_BASE_URL}. Run `ollama serve` and try again."
            ) from exc
        if "not found" in message or "model" in message:
            raise LLMError(f"Model '{LLM_MODEL}' not found. Run `ollama pull {LLM_MODEL}`.") from exc
        raise LLMError(f"LLM generation failed: {exc}") from exc

    if not response or not response.strip():
        raise LLMError("The LLM returned an empty response.")

    return response.strip()


def stream_answer(question: str, context_chunks, history: str = ""):
    """Same prompt as generate_answer(), but yields the answer piece by
    piece as Ollama generates it, instead of waiting for the full answer.

    This is what makes a "Stop generating" button possible: the caller
    (see api.py) can simply stop iterating this generator partway
    through, and no more tokens get produced or sent.
    """
    context = format_context(context_chunks)
    prompt = RAG_PROMPT_TEMPLATE.format(
        question=question, context=context, history=history or "(no previous conversation)"
    )

    try:
        for token in _get_llm().stream(prompt):
            yield token
    except Exception as exc:
        raise LLMError(f"LLM streaming failed: {exc}") from exc

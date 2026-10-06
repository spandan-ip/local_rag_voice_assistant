def build_sources(retrieved_docs) -> list[dict]:
    seen = set()
    sources = []

    for doc in retrieved_docs:
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page")
        key = (source, page)
        if key in seen:
            continue
        seen.add(key)
        sources.append({"source": source, "page": page})

    return sources

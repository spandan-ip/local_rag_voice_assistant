import json
import os

from app.config import SOURCES as DEFAULT_SOURCES, VECTOR_STORE_DIR

_REGISTRY_FILENAME = "dynamic_sources.json"


def _registry_path(registry_dir: str = VECTOR_STORE_DIR) -> str:
    return os.path.join(registry_dir, _REGISTRY_FILENAME)


def _load_dynamic_sources(registry_dir: str = VECTOR_STORE_DIR) -> list[dict]:
    path = _registry_path(registry_dir)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _save_dynamic_sources(sources: list[dict], registry_dir: str = VECTOR_STORE_DIR) -> None:
    os.makedirs(registry_dir, exist_ok=True)
    with open(_registry_path(registry_dir), "w") as f:
        json.dump(sources, f, indent=2)


def add_source(source_type: str, source: str, registry_dir: str = VECTOR_STORE_DIR) -> None:
    sources = _load_dynamic_sources(registry_dir)
    entry = {"source_type": source_type, "source": source}
    if entry not in sources:
        sources.append(entry)
        _save_dynamic_sources(sources, registry_dir)


def get_all_sources(registry_dir: str = VECTOR_STORE_DIR) -> list[dict]:
    return DEFAULT_SOURCES + _load_dynamic_sources(registry_dir)

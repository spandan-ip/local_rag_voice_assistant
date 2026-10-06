from app.ingestion.source_registry import add_source, get_all_sources


def test_add_source_appears_in_get_all_sources(tmp_path):
    registry_dir = str(tmp_path)
    add_source("txt", "source/notes.txt", registry_dir=registry_dir)

    sources = get_all_sources(registry_dir=registry_dir)
    assert {"source_type": "txt", "source": "source/notes.txt"} in sources


def test_add_source_does_not_duplicate(tmp_path):
    registry_dir = str(tmp_path)
    add_source("web", "https://example.com", registry_dir=registry_dir)
    add_source("web", "https://example.com", registry_dir=registry_dir)

    sources = get_all_sources(registry_dir=registry_dir)
    matching = [s for s in sources if s == {"source_type": "web", "source": "https://example.com"}]
    assert len(matching) == 1


def test_dynamic_sources_persist_across_calls(tmp_path):
    registry_dir = str(tmp_path)
    add_source("csv", "source/data.csv", registry_dir=registry_dir)

    # Simulate a fresh call (e.g. after an app restart) by just calling
    # get_all_sources again with the same registry_dir -- it re-reads from disk.
    sources = get_all_sources(registry_dir=registry_dir)
    assert {"source_type": "csv", "source": "source/data.csv"} in sources


def test_get_all_sources_includes_default_sources_from_config(tmp_path):
    from app.config import SOURCES as DEFAULT_SOURCES

    sources = get_all_sources(registry_dir=str(tmp_path))
    for default in DEFAULT_SOURCES:
        assert default in sources

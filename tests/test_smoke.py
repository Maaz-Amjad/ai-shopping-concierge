"""Smoke test: config loads and core libraries import."""

from src import load_config


def test_config_loads_with_required_keys():
    cfg = load_config()
    assert cfg["seed"] == 42
    assert cfg["llm"]["provider"] == "gemini"
    assert cfg["llm"]["model"] != cfg["llm"]["judge_model"]  # spec 10.3: judge differs from generator
    assert cfg["embedding"]["model"] == "BAAI/bge-small-en-v1.5"


def test_core_libraries_import():
    import faiss  # noqa: F401
    import pandas  # noqa: F401
    import sentence_transformers  # noqa: F401
    import sklearn  # noqa: F401
    from google import genai  # noqa: F401

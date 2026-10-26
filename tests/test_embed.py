"""Tests for src/embed.py with a fake model (no model download)."""

import numpy as np
import pytest

from src.embed import (QUERY_INSTRUCTION, encode_products, encode_query, load_embeddings,
                       save_embeddings, use_query_instruction)


class FakeModel:
    """Mimics SentenceTransformer.encode: deterministic vectors, records prompts."""

    def __init__(self) -> None:
        self.prompts: list = []

    def encode(self, inputs, prompt=None, normalize_embeddings=False, **kw):
        self.prompts.append(prompt)
        vecs = np.array([[len(t), t.count("a") + 1, 1.0] for t in inputs], dtype=np.float64)
        if normalize_embeddings:
            vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


def test_encode_products_float32_and_normalized():
    emb = encode_products(["abc", "aaaa"], FakeModel())
    assert emb.dtype == np.float32 and emb.shape == (2, 3)
    assert np.allclose(np.linalg.norm(emb, axis=1), 1.0)


def test_encode_query_instruction_toggle():
    model = FakeModel()
    vec = encode_query("ice maker", model, use_instruction=True)
    encode_query("ice maker", model, use_instruction=False)
    assert model.prompts == [QUERY_INSTRUCTION, None]
    assert vec.shape == (3,) and vec.dtype == np.float32


@pytest.mark.parametrize("setting, expected", [("auto", True), (True, True), (False, False)])
def test_use_query_instruction_reads_config(setting, expected):
    assert use_query_instruction({"embedding": {"query_instruction": setting}}) is expected


def test_use_query_instruction_rejects_unknown_value():
    with pytest.raises(ValueError):
        use_query_instruction({"embedding": {"query_instruction": "sometimes"}})


def test_save_load_roundtrip(tmp_path):
    emb = np.random.default_rng(42).random((4, 3), dtype=np.float32)
    save_embeddings(emb, ["a", "b", "c", "d"], tmp_path)
    loaded, ids = load_embeddings(tmp_path)
    assert np.array_equal(loaded, emb) and ids.tolist() == ["a", "b", "c", "d"]


def test_save_rejects_mismatched_ids(tmp_path):
    with pytest.raises(ValueError):
        save_embeddings(np.zeros((2, 3), dtype=np.float32), ["only-one"], tmp_path)

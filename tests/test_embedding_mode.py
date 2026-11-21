import sys
from types import SimpleNamespace

import pytest

from stampmem import DefaultEmbedder


def test_semantic_model_load_failure_is_explicit(monkeypatch):
    def unavailable(_):
        raise OSError("model unavailable")
    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=unavailable))
    with pytest.raises(RuntimeError, match="use_model=False") as err:
        DefaultEmbedder(model_name="unavailable")
    assert isinstance(err.value.__cause__, OSError)


def test_hash_mode_is_explicit():
    model = DefaultEmbedder(use_model=False)
    assert model.model is None
    assert len(model.embed("测试")) == 128
    with pytest.raises(ValueError, match="model_name"):
        DefaultEmbedder(use_model=False, model_name="unused")

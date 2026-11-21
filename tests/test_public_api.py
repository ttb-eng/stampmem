# -*- coding: utf-8 -*-
"""公开 API 的端到端测试：像用户一样装库后能用。"""
from datetime import datetime, timedelta

import pytest

import stampmem as tm


# ---------------------------------------------------------------- 导出面
def test_public_api_complete():
    for name in tm.__all__:
        assert hasattr(tm, name), f"__all__ 里声明了 {name} 但实际没有"


def test_version_is_string():
    assert isinstance(tm.__version__, str) and tm.__version__.count(".") >= 1


# ---------------------------------------------------------------- 历史文档打时间戳
def test_stamp_documents_batch():
    docs = [{"id": f"d{i}", "text": f"第{i}份会议记录，讨论了项目进度"} for i in range(1, 31)]
    periods = {f"d{i}": f"2026-{min(10, (i - 1) // 3 + 1):02d}" for i in range(1, 31)}
    stamped = tm.stamp_documents(docs, periods=periods)

    assert len(stamped) == len(docs)
    assert all(s.get("timestamp") for s in stamped)
    ts = [s["timestamp"] for s in stamped]
    assert ts == sorted(ts), "时间不应倒退"

    payload = tm.to_memory_payload(stamped)
    assert isinstance(payload, list) and len(payload) == len(docs)


# ---------------------------------------------------------------- 写入 + 检索
@pytest.fixture()
def mem(db_dir):
    m = tm.Memory(collection_name="api_t", embedder=tm.DefaultEmbedder(use_model=False),
                  persist_dir=db_dir)
    base = datetime(2026, 1, 1)
    rows = [
        ("我养了一只狸花猫，叫大黄", 0, "L1"),
        ("大黄送回老家了", 120, "L1"),
        ("后来接了一只英短回家，叫豆豆", 240, "L1"),
        ("楼下新开了家咖啡店", 60, "L2"),
    ]
    for text, days, lid in rows:
        m.add(text, metadata={"line_id": lid},
              created_at=(base + timedelta(days=days)).isoformat())
    return m


def test_retrieve_returns_contract(mem):
    r = mem.retrieve("我现在养的是什么猫？", n_results=4)
    assert isinstance(r, dict) and "hits" in r and "metrics" in r
    assert all("created_at" in (h.get("metadata") or {}) for h in r["hits"])


def test_lightweight_retrieval_preserves_order_and_result_fields(mem):
    query = "我现在养的是什么猫？"
    full = mem.retrieve(query, n_results=3)["hits"]
    light = mem.retrieve(query, n_results=3, include_embeddings=False)["hits"]
    assert full and all(row["embedding"] is not None for row in full)
    assert light == [{**row, "embedding": None} for row in full]


def test_retrieve_finds_latest_state(mem):
    r = mem.retrieve("我现在养的是什么猫？", n_results=4)
    joined = " ".join(h["text"] for h in r["hits"])
    assert "英短" in joined, f"没召回最新状态: {joined}"


def test_retrieve_chronological_same_evidence(mem):
    sim = mem.retrieve("我现在养的是什么猫？", n_results=4)
    chro = mem.retrieve("我现在养的是什么猫？", n_results=4, order="chronological")
    assert sorted(h["text"] for h in chro["hits"]) == sorted(h["text"] for h in sim["hits"]), \
        "时间序只能改顺序，不能改证据集合"
    ts = [h["metadata"]["created_at"] for h in chro["hits"]]
    assert ts == sorted(ts)


def test_retrieve_rejects_bad_order(mem):
    with pytest.raises(ValueError):
        mem.retrieve("随便问问", order="bogus")


def test_count_and_forget(mem):
    before = mem.count()
    assert before == 4
    mid = mem.retrieve("咖啡店", n_results=1)["hits"][0]["id"]
    mem.forget(mid)
    assert mem.count() == before - 1


# ---------------------------------------------------------------- 不硬依赖 openai
def test_llm_module_importable():
    import stampmem.llm  # noqa: F401

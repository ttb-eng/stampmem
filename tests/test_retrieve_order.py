# -*- coding: utf-8 -*-
"""
Memory.retrieve(order="chronological") 的行为测试。

行为契约：
  chronological **不增删任何候选**，只改呈现顺序，因此它只需要时间戳、不需要人工标注。
"""
from datetime import datetime, timedelta

import pytest

from stampmem import DefaultEmbedder, Memory


@pytest.fixture()
def mem(db_dir):
    m = Memory(collection_name="order_t", embedder=DefaultEmbedder(use_model=False),
               persist_dir=db_dir)
    base = datetime(2026, 1, 1)
    rows = [
        ("我养了一只狸花猫，叫大黄", 0),
        ("我把大黄送回老家了", 120),
        ("后来接了一只英短回家，叫豆豆", 240),
        ("楼下新开了家咖啡店", 60),
        ("我在用一台旧笔记本", 300),
    ]
    for text, days in rows:
        m.add(text, created_at=(base + timedelta(days=days)).isoformat())
    return m


QUERY = "我现在养的是什么猫？"


def test_same_count(mem):
    sim = mem.retrieve(QUERY, n_results=5)
    chro = mem.retrieve(QUERY, n_results=5, order="chronological")
    assert len(sim["hits"]) == len(chro["hits"])


def test_same_evidence_set(mem):
    """这是"只改顺序"的严格断言：集合必须完全一致。"""
    sim = mem.retrieve(QUERY, n_results=5)
    chro = mem.retrieve(QUERY, n_results=5, order="chronological")
    assert sorted(h["text"] for h in sim["hits"]) == sorted(h["text"] for h in chro["hits"])


def test_chronological_is_ascending(mem):
    chro = mem.retrieve(QUERY, n_results=5, order="chronological")
    ts = [h["metadata"]["created_at"] for h in chro["hits"]]
    assert ts == sorted(ts)


def test_order_actually_differs(mem):
    """此示例的相关性顺序与时间顺序应不同。"""
    sim = mem.retrieve(QUERY, n_results=5)
    chro = mem.retrieve(QUERY, n_results=5, order="chronological")
    assert [h["text"] for h in sim["hits"]] != [h["text"] for h in chro["hits"]]


def test_default_is_similarity(mem):
    """默认必须保持向后兼容。"""
    a = mem.retrieve(QUERY, n_results=5)
    b = mem.retrieve(QUERY, n_results=5, order="similarity")
    assert [h["text"] for h in a["hits"]] == [h["text"] for h in b["hits"]]


def test_invalid_order_raises(mem):
    with pytest.raises(ValueError):
        mem.retrieve(QUERY, order="bogus")

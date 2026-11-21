# -*- coding: utf-8 -*-
"""
README「快速上手」示例的逐字执行测试。

为什么要有这个：README 里写着
    # d1 2026-02-26 extrapolated 0.2
如果实际跑出来不是这样，README 就在误导用户。文档与行为必须一起改。
"""
from stampmem import Memory, parse_period, stamp_documents, to_memory_payload

DOCS = [
    {"id": "d1", "text": "今天去体检了，医生说要少熬夜"},
    {"id": "d2", "text": "2026年3月5日 复查结果正常"},
    {"id": "d3", "text": "三天前换了新药"},
]


def test_readme_example_1_no_periods():
    out = stamp_documents(DOCS, trust_text_dates=True)
    got = {o["id"]: (o["timestamp"], o["method"]) for o in out}
    # README 里承诺的输出
    assert got["d2"] == ("2026-03-05", "absolute")
    assert got["d1"] == ("2026-02-26", "extrapolated")
    assert got["d3"] == ("2026-03-12", "extrapolated")


def test_readme_example_2_with_periods():
    out = stamp_documents(DOCS, periods={
        "d1": "2026-03",
        "d2": "2026-03-05",
        "d3": ("2026-03-01", "2026-03-31"),
    })
    ts = [o["timestamp"] for o in out]
    assert len(ts) == 3
    assert all(t.startswith("2026-03") for t in ts)
    assert ts == sorted(ts)


def test_readme_output_fields():
    for o in stamp_documents(DOCS):
        assert {"id", "timestamp", "method", "confidence"} <= set(o)


def test_readme_parse_period_forms():
    assert tuple(map(str, parse_period("2026-03"))) == ("2026-03-01", "2026-03-31")
    assert tuple(map(str, parse_period("2026-03-05"))) == ("2026-03-05", "2026-03-05")
    assert tuple(map(str, parse_period("2026"))) == ("2026-01-01", "2026-12-31")


def test_readme_payload_to_memory(db_dir):
    """README 里 payload -> Memory.add 的串联要真的能跑。"""
    payload = to_memory_payload(stamp_documents(DOCS, periods={i:"2026-03" for i in range(len(DOCS))}))
    assert isinstance(payload, list) and len(payload) == 3

    from stampmem import DefaultEmbedder
    mem = Memory(collection_name="readme_t", persist_dir=db_dir, embedder=DefaultEmbedder(use_model=False))
    for item in payload:
        mem.add(**item)
    assert mem.count() == len(payload)

    r = mem.retrieve("复查结果怎么样？", n_results=3)
    assert len(r["hits"]) > 0
    from stampmem import render_context
    assert '"ts_period_end": "2026-03-31"' in render_context(r["hits"])
    assert '"observed_at"' not in render_context(r["hits"])
    assert '"observed_at"' in render_context(r["hits"], include_estimated_day=True)
    assert "ts_period_end" not in render_context(r["hits"], include_time=False)

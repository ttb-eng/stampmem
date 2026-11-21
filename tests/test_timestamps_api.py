# -*- coding: utf-8 -*-
"""时间戳推断与解析的单元测试（不联网、不建库）。"""
from datetime import date

import pytest

from stampmem import parse_period, stamp_documents, to_memory_payload


# ---------------------------------------------------------------- parse_period
def test_parse_period_month():
    assert parse_period("2026-03") == (date(2026, 3, 1), date(2026, 3, 31))


def test_parse_period_year():
    assert parse_period("2026") == (date(2026, 1, 1), date(2026, 12, 31))


def test_parse_period_december_crosses_year():
    assert parse_period("2026-12")[1] == date(2026, 12, 31)


def test_parse_period_single_day():
    assert parse_period("2026-03-05") == (date(2026, 3, 5), date(2026, 3, 5))


def test_parse_period_tuple_passthrough():
    a, b = parse_period(("2026-03-01", "2026-03-31"))
    assert (str(a), str(b)) == ("2026-03-01", "2026-03-31")


# ---------------------------------------------------------------- 正文绝对日期
def test_absolute_dates_are_exact():
    docs = [
        {"id": "a", "text": "2026年3月1日 开始记日记"},
        {"id": "b", "text": "今天又写了一篇"},
        {"id": "c", "text": "2026年3月11日 满十篇了"},
    ]
    out = stamp_documents(docs, trust_text_dates=True)
    ts = {o["id"]: o["timestamp"] for o in out}
    assert ts["a"] == "2026-03-01"
    assert ts["c"] == "2026-03-11"
    assert "2026-03-01" <= ts["b"] <= "2026-03-11"
    assert {o["id"]: o["method"] for o in out}["b"] == "interpolated"


# ---------------------------------------------------------------- 月份粗锚点
def test_month_anchors_stay_in_month():
    docs = [{"id": f"d{i}", "text": f"随便聊聊第{i}条"} for i in range(10)]
    periods = {f"d{i}": "2026-06" for i in range(10)}
    out = stamp_documents(docs, periods=periods)
    ts = [o["timestamp"] for o in out]
    assert all(s.startswith("2026-06") for s in ts)
    assert ts == sorted(ts)
    assert all(o["method"] == "file_anchor" for o in out)


# ---------------------------------------------------------------- 无锚点
def test_no_anchor_keeps_order_and_low_confidence():
    out = stamp_documents([{"id": i, "text": f"第{i}条"} for i in range(5)])
    assert all(o["method"] == "extrapolated" for o in out)
    assert all(o["confidence"] <= 0.2 for o in out)


# ---------------------------------------------------------------- 回归：硬锚点不可被移动
def test_hard_anchor_is_immutable():
    docs = [
        {"id": "x", "text": "今天心情不好"},
        {"id": "y", "text": "昨天也是"},
        {"id": "z", "text": "2026年1月2日 好转了"},
        {"id": "w", "text": "今天继续好转"},
    ]
    out = stamp_documents(docs, trust_text_dates=True)
    z = next(o for o in out if o["id"] == "z")
    assert z["timestamp"] == "2026-01-02"
    ts = [o["timestamp"] for o in out]
    assert ts == sorted(ts)


# ---------------------------------------------------------------- 边界
def test_empty_input():
    assert stamp_documents([]) == []


def test_output_contract():
    """输出的字段名是对外承诺，不能随手改名。"""
    out = stamp_documents([{"id": "a", "text": "2026年5月1日 一条"}])
    assert {"id", "timestamp", "method", "confidence"} <= set(out[0])


@pytest.mark.parametrize("trust_text", [False, True])
def test_explicit_archive_month_overrides_event_text_and_neighbor_drift(trust_text):
    docs = [
        {"id":"old","text":"1990年1月5日 市场开业"},
        {"id":"new","text":"今天回顾往事"},
        {"id":"future","text":"2090年1月1日 建设规划"},
    ]
    out = stamp_documents(docs, periods={"old":"2023-02","new":"2023-03","future":"2023-04"},
                          trust_text_dates=trust_text)
    assert [o["timestamp"] for o in out] == ["2023-02-15","2023-03-16","2023-04-15"]
    assert all(o["method"] == "file_anchor" for o in out)
    assert to_memory_payload(out)[0]["metadata"]["ts_period_end"] == "2023-02-28"


def test_plain_document_date_is_not_an_observation_anchor_by_default():
    out = stamp_documents([{"text":"2010年4月10日 我搬了家"}])
    assert out[0]["time_source"] == "synthetic_unanchored"
    with pytest.raises(ValueError, match="不可辨识"):
        to_memory_payload(out)


def test_first_exact_anchor_must_agree_with_archive_period():
    docs = [{"text":"1990年3月1日 有一件旧事"}]
    with pytest.raises(ValueError, match="冲突"):
        stamp_documents(docs, periods={0:"2024-02"}, first_anchor=date(2024,3,1))
    out = stamp_documents(docs, periods={0:"2024-02"}, first_anchor=date(2024,2,29), trust_text_dates=True)
    assert out[0]["timestamp"] == "2024-02-29"
    assert out[0]["observation_bounds"] == ("2024-02-01","2024-02-29")


def test_reversed_period_is_rejected():
    with pytest.raises(ValueError,match="不可晚于"):
        parse_period(("2024-03-01","2024-02-29"))


def test_conflicting_archive_order_does_not_move_dates_across_periods():
    out = stamp_documents([{"text":"later"},{"text":"earlier"}],
                          periods={0:"2024-03",1:"2024-02"})
    assert [o["timestamp"][:7] for o in out] == ["2024-03","2024-02"]

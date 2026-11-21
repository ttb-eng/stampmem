# -*- coding: utf-8 -*-
"""
公共时间戳接口：把一批"没有时间字段"的历史文档批量打上时间戳。

用法
----
>>> from stampmem import stamp_documents
>>> docs = [{"id": "a", "text": "今天去体检了"},
...         {"id": "b", "text": "2026年3月5日 复查结果正常"},
...         {"id": "c", "text": "三天前换了药"}]
>>> out = stamp_documents(docs)
>>> [(o["id"], o["timestamp"], o["method"]) for o in out]

重要前提
--------
1. **传入顺序必须是原始顺序**（目录 / 文件名 / 导出批次提供的先后）。
   这是本算法唯一免费且必需的信息。
2. ``periods`` 指调用方提供的可信观察日期区间。点估计不会越界，也不等于真实日期。
3. 正文日期默认不作观察锚点。无可信锚点时只生成不可默认导入的顺序坐标。
"""

import re
from calendar import monthrange
from datetime import date, datetime, timedelta

from .timestamp_infer_v3 import infer_timestamps_v3

__all__ = ["stamp_documents", "parse_period"]

_DATE_RE = re.compile(r"^\s*(\d{4})(?:-(\d{1,2}))?(?:-(\d{1,2}))?\s*$")


def _parse_date_groups(spec):
    m = _DATE_RE.match(str(spec).strip())
    if not m:
        raise ValueError(f"无法解析时间区间: {spec!r}")
    y = int(m.group(1))
    mo = int(m.group(2)) if m.group(2) else None
    d = int(m.group(3)) if m.group(3) else None
    return y, mo, d


def parse_period(spec):
    """把 '2026-03' / '2026-03-01' / '2026' / (date, date) 解析成日期区间"""
    if isinstance(spec, (tuple, list)) and len(spec) == 2:
        a, b = (x.date() if isinstance(x, datetime) else x if isinstance(x, date) else _to_date(str(x))
                for x in spec)
        if a > b:
            raise ValueError("时间区间的开始日期不可晚于结束日期")
        return a, b
    y, mo, d = _parse_date_groups(spec)
    if d is not None:
        if mo is None:
            raise ValueError(f"给了日却没给月: {spec!r}")
        one = date(y, mo, d)
        return one, one
    if mo is not None:
        first = date(y, mo, 1)
        return first, date(y, mo, monthrange(y, mo)[1])
    return date(y, 1, 1), date(y, 12, 31)


def _to_date(s):
    y, mo, d = _parse_date_groups(s)
    if mo is None:
        raise ValueError(f"需要具体日期，得到: {s!r}")
    return date(y, mo, d if d is not None else 1)


def _order_stat_anchors(idx_by_period):
    """周期内按顺序统计量给出点估计（均匀先验下的最小期望误差）"""
    extra = {}
    for (first, last), idxs in idx_by_period.items():
        span = (last - first).days
        n = len(idxs)
        for k, i in enumerate(idxs):
            extra[i] = first + timedelta(days=round((k + 1) / (n + 1) * span))
    return extra


def _project_archive_order(preferred, bounds):
    """Preserve input order when consistent with all known intervals."""
    ordered = sorted(bounds)
    latest = date.max
    ceiling = {}
    for i in reversed(ordered):
        lo, hi = bounds[i]
        latest = min(latest, hi)
        if lo > latest:
            # Input order cannot be correct under all provided dates; keep
            # each bound rather than silently moving a record to another period.
            return preferred
        ceiling[i] = latest
    result = {}
    last = date.min
    for i in ordered:
        last = max(last, bounds[i][0], min(preferred[i], ceiling[i]))
        result[i] = last
    return result


def stamp_documents(docs, periods=None, first_anchor=None,
                    default_interval_days=7, use_self_refine=False, *, trust_text_dates=False):
    """
    给一批历史文档批量推断时间戳。

    参数
    ----
    docs : list[dict]
        每条至少含 ``text``；可含 ``id``。**列表顺序即原始时间顺序。**
    periods : dict | None
        可信观察日期区间，{文档 id: 周期}，周期写法见 :func:`parse_period`。
        也可以传 {下标: 周期}。
    first_anchor : date | None
        已知首条记录的确切观察日期时传入；若同时有区间，必须在区间内。
    trust_text_dates : bool
        默认关闭。仅当语料保证正文日期就是记录的观察日期时才打开。
    default_interval_days : int
        完全没有锚点时外推用的默认间隔。
    use_self_refine : bool
        是否启用"相邻记录自指差分"细化。实测增益≈0（触发率 <1%），默认关闭。

    返回
    ----
    list[dict]：每条含
        ``id`` / ``text`` / ``timestamp``(ISO 日期) / ``confidence`` /
        ``method`` / ``evidence``
    """
    docs = list(docs)
    if not docs:
        return []

    records = [{"text": d.get("text", ""), "id": d.get("id", i)}
               for i, d in enumerate(docs)]

    extra = None
    bounds = {}
    if periods:
        idx_by_period = {}
        for i, d in enumerate(docs):
            key = d.get("id", i)
            spec = periods.get(key, periods.get(i))
            if spec is None:
                continue
            bounds[i] = parse_period(spec)
            idx_by_period.setdefault(bounds[i], []).append(i)
        if idx_by_period:
            extra = _order_stat_anchors(idx_by_period)

    if isinstance(first_anchor, datetime):
        first_anchor = first_anchor.date()
    if first_anchor is not None and not isinstance(first_anchor, date):
        raise TypeError("first_anchor 必须是 date")
    if first_anchor is not None and 0 in bounds and not bounds[0][0] <= first_anchor <= bounds[0][1]:
        raise ValueError("首条观察日期与已知观察区间冲突")
    if extra:
        order_bounds = dict(bounds)
        preferred = dict(extra)
        if first_anchor is not None:
            order_bounds[0] = (first_anchor, first_anchor)
            preferred[0] = first_anchor
        extra = _project_archive_order(preferred, order_bounds)

    out = infer_timestamps_v3(records, first_anchor=first_anchor,
                              default_interval_days=default_interval_days,
                              use_self_refine=use_self_refine,
                              extra_anchors=extra, trust_text_dates=trust_text_dates,
                              lock_extra_anchors=True)

    identifiable = any(o["method"] in ("absolute", "file_anchor") for o in out)
    for i, o in enumerate(out):
        o["timestamp"] = o["utter_est"].isoformat()
        if i in bounds:
            first, last = bounds[i]
            if not first <= o["utter_est"] <= last:
                raise RuntimeError("推断结果越出已知观察区间")
            o["observation_bounds"] = (first.isoformat(), last.isoformat())
        o["time_identifiable"] = identifiable
        o["time_source"] = "inferred" if identifiable else "synthetic_unanchored"
    return out


def to_memory_payload(stamped, *, allow_unanchored=False):
    """转换为 Memory.add(**payload)；无锚点的合成日期默认禁止作为真实时间导入。

    兼容旧格式：没有 time_identifiable 的结果必须逐批显式审核后才能使用。
    allow_unanchored=True 是调用方主动接受合成坐标，不代表恢复出真实日期。
    """
    result = []
    for o in stamped:
        identifiable = o.get("time_identifiable", False)
        if not identifiable and not allow_unanchored:
            raise ValueError("历史时间不可辨识或来源未标记；请提供时间锚点，勿导入人为日期")
        meta = {"ts_confidence": o["confidence"], "ts_method": o["method"]}
        if "observation_bounds" in o:
            first, last = o["observation_bounds"]
            meta.update(ts_period_start=first, ts_period_end=last)
        result.append({"text": o["text"], "created_at": o["timestamp"],
                       "time_source": o.get("time_source", "unverified_legacy"),
                       "metadata": meta})
    return result

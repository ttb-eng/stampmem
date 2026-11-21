# -*- coding: utf-8 -*-
"""
时间戳推断 v2：区分「说话时间（utterance）」与「事件时间（event）」，并用时间单调性约束。

核心洞察
--------
真实对话里，时间表达描述的是**事件**，而我们要推断的是**这条记录被说出的时间**。
两者不同：
    "去年冬天去哈尔滨玩了"   ← 事件在过去，但这句话可能说于今年任何时刻
    "我昨天去了医院"         ← "昨天" 直接给出说话时间

因此算法分两步：
    ① 抽取时间表达，并分类为「自指」或「事件」
    ② 以时间单调性（对话按时间顺序，说话时间非递减）为约束，
       自指表达直接定位，事件表达只提供上下界，空缺处用锚点插值

输出为**区间** [earliest, latest] 而非单点，便于下游做保守的时序推理。
"""

import re
from datetime import date, timedelta

from .timestamp_infer import (
    extract_time_expressions, cn_to_int, _add_months,
)

# 自指：直接给出说话时间
SELF_KINDS = {
    "REL_DAY_SELF",     # 今天/昨天/前天/大前天
}
# 未来指向：说话时间 ≈ 事件时间 - 偏移
FUTURE_KINDS = {"REL_DAY_FUTURE", "OFFSET_FUTURE"}
# 事件指向：只给事件时间，为说话时间提供下界
EVENT_KINDS = {"ABS", "OFFSET_PAST", "REL_WEEK", "REL_MONTH", "REL_YEAR"}

SELF_DAY = {"大前天": -3, "前天": -2, "昨天": -1, "今天": 0}
FUTURE_DAY = {"明天": 1, "后天": 2, "大后天": 3}


def classify_expressions(text: str) -> dict:
    """
    把文本中的时间表达分为三类：
        self   : 自指，给出说话时间
        future : 未来指向
        event  : 事件指向（含绝对日期）
    """
    raw = extract_time_expressions(text)
    out = {"self": [], "future": [], "event": []}

    for e in raw:
        kind = e["kind"]
        v = e["value"]
        if kind == "REL_DAY":
            word = e["raw"]
            if word in SELF_DAY:
                out["self"].append({**e, "offset": SELF_DAY[word]})
            elif word in FUTURE_DAY:
                out["future"].append({**e, "offset": FUTURE_DAY[word]})
            continue
        if kind == "ABS_FULL" or isinstance(v, date):
            out["event"].append({**e, "abs": v})
            continue
        if kind == "ABS_MD":
            out["event"].append({**e})
            continue
        if kind == "OFFSET":
            _, n, unit = v
            if n > 0:
                out["future"].append({**e, "n": n, "unit": unit})
            else:
                out["event"].append({**e, "n": n, "unit": unit})
            continue
        if kind in ("REL_WEEK", "REL_MONTH", "REL_YEAR"):
            out["event"].append({**e})
            continue
        # VAGUE 等：忽略
    return out


def _offset_days(n: float, unit: str) -> int:
    if unit == "day":
        return int(round(n))
    if unit == "week":
        return int(round(n * 7))
    if unit == "month":
        return int(round(n * 30))
    if unit == "year":
        return int(round(n * 365))
    return int(round(n))


def infer_timestamps_v2(records: list[dict],
                        default_start: date | None = None,
                        default_interval_days: int = 1,
                        max_gap_days: int = 365) -> list[dict]:
    """
    推断「说话时间」区间。

    返回每条记录新增：
        utter_from / utter_to : 说话时间区间
        utter_est            : 点估计（区间中点）
        event_est            : 事件时间点估计（若可解析）
        confidence           : 置信度
        method               : self / future / anchor / sequence
        evidence             : 依据原文
    """
    n = len(records)
    recs = [dict(r) for r in records]
    start = default_start or date(2026, 1, 1)

    # ---------- ① 分类 ----------
    for r in recs:
        r["_cls"] = classify_expressions(r.get("text", ""))

    # ---------- ② 正反向传播约束 ----------
    # lower[i] : 说话时间的下界；upper[i] : 上界
    lower = [None] * n
    upper = [None] * n
    point = [None] * n      # 由自指直接确定的时间点
    method = ["sequence"] * n
    evidence = [""] * n
    conf = [0.15] * n

    for i, r in enumerate(recs):
        c = r["_cls"]
        # 自指：直接确定说话时间（相对"当前"未知，用相对关系记录）
        if c["self"]:
            e = max(c["self"], key=lambda x: x["confidence"])
            r["_self_offset"] = e["offset"]
            r["_self_evidence"] = e["raw"]
            method[i] = "self"
            evidence[i] = e["raw"]
            conf[i] = 0.9
        # 未来指向：说话时间 = 事件时间 - 偏移（给上界）
        if c["future"]:
            e = c["future"][0]
            if "n" in e:
                days = _offset_days(e["n"], e["unit"])
            else:
                days = e["offset"]
            r["_future_days"] = days
            r["_future_evidence"] = e["raw"]
            if method[i] == "sequence":
                method[i] = "future"
                evidence[i] = e["raw"]
                conf[i] = 0.8

    # ---------- ③ 以自指为强锚点，向两侧推进 ----------
    # 找到所有自指位置，用它们把时间线切成段
    self_idx = [i for i in range(n) if recs[i].get("_self_offset") is not None]

    est = [None] * n
    if self_idx:
        # 第一个自指点作为初始锚
        first = self_idx[0]
        est[first] = start
        for i in range(first - 1, -1, -1):
            est[i] = est[i + 1] - timedelta(days=default_interval_days)
        for k, i in enumerate(self_idx):
            if k > 0:
                prev = self_idx[k - 1]
                prev_t = est[prev]
                # 用自指偏移差推算
                delta = recs[i]["_self_offset"] - recs[prev]["_self_offset"]
                if delta > 0:
                    est[i] = prev_t + timedelta(days=delta)
                else:
                    # 偏移相同或倒退：按位置间隔保证单调
                    est[i] = prev_t + timedelta(days=max(1, i - prev))
            # 段内插值
            if k > 0:
                prev = self_idx[k - 1]
                span = i - prev
                for j in range(prev + 1, i):
                    frac = (j - prev) / span
                    est[j] = prev_t + timedelta(
                        days=int(round((est[i] - prev_t).days * frac)))
        # 最后一段向后延展
        last = self_idx[-1]
        for i in range(last + 1, n):
            est[i] = est[i - 1] + timedelta(days=default_interval_days)
    else:
        # 无自指：纯序列推进
        est[0] = start
        for i in range(1, n):
            est[i] = est[i - 1] + timedelta(days=default_interval_days)

    # ---------- ④ 事件表达给出下界，修正说话时间 ----------
    for i, r in enumerate(recs):
        c = r["_cls"]
        event_est = None
        if c["event"]:
            e = c["event"][0]
            if e.get("abs"):
                event_est = e["abs"]
            elif "n" in e:
                days = _offset_days(e["n"], e["unit"])
                event_est = est[i] + timedelta(days=days)
            elif e["kind"] == "REL_YEAR":
                from .timestamp_infer import REL_YEAR_OFFSET
                y = est[i].year + REL_YEAR_OFFSET.get(e["raw"], 0)
                try:
                    event_est = date(y, est[i].month, min(est[i].day, 28))
                except ValueError:
                    event_est = None
            elif e["kind"] == "REL_MONTH":
                from .timestamp_infer import REL_MONTH_OFFSET
                event_est = _add_months(est[i], REL_MONTH_OFFSET.get(e["raw"], 0))
            r["event_est"] = event_est
            if event_est:
                # 说话时间不可能早于所提及的事件时间
                if est[i] < event_est:
                    est[i] = event_est
                    if method[i] in ("sequence",):
                        method[i] = "anchor"
                        evidence[i] = e["raw"]
                        conf[i] = 0.6
        else:
            r["event_est"] = None

    # ---------- ⑤ 强制单调 ----------
    for i in range(1, n):
        if est[i] < est[i - 1]:
            est[i] = est[i - 1] + timedelta(days=max(1, default_interval_days))

    # ---------- ⑥ 输出区间 ----------
    for i, r in enumerate(recs):
        r["utter_est"] = est[i]
        r["utter_from"] = est[i] - timedelta(days=2)
        r["utter_to"] = est[i] + timedelta(days=2)
        r["confidence"] = conf[i]
        r["method"] = method[i]
        r["evidence"] = evidence[i]
    return recs

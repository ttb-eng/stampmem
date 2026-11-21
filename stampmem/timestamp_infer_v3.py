# -*- coding: utf-8 -*-
"""
时间戳推断 v3：以「绝对时间锚点」为核心，锚点之间用序列插值 + 相对表达细化。

设计依据（v2 测试得出的结论）
------------------------------
相对表达（今天/昨天/三天前）相对于「该句自身」而言，不含绝对信息；
若无任何绝对置信锚点，绝对时间存在整体平移的不可辨识性。
因此 v3 的策略是：
    ① 绝对日期记录 = 强锚点（误差≈0），作为**硬约束**，后续步骤不得修改
    ② 锚点之间：按记录位置线性插值
    ③ 自指偏移细化：仅在「前后两个锚点」构成的闭区间内生效，越界即丢弃
       （相邻两句都含"今天/昨天"只给出下界，不能给出确切差值；不加区间约束
        会形成链式累加，实测每 200 条记录漂移约 100 天）
    ④ 锚点之外：按默认间隔外推，并标记低置信度
    ⑤ 单调：对话按时间顺序非递减，但只移动非锚点记录
"""

import re
from datetime import date, timedelta

from .timestamp_infer import extract_time_expressions

SELF_DAY = {"大前天": -3, "前天": -2, "昨天": -1, "今天": 0}
FUTURE_DAY = {"明天": 1, "后天": 2, "大后天": 3}


def _offset_days(n: float, unit: str) -> int:
    return int(round(n * {"day": 1, "week": 7, "month": 30, "year": 365}.get(unit, 1)))


def _estimate_rate(est, spine) -> float | None:
    """
    从锚点之间的实际间距估计「每条记录相当于多少天」。

    用相邻锚点的 (天差 / 条数差) 的中位数，因此对个别异常锚点稳健。
    估不出来（锚点少于 2 个，或所有锚点同日）时返回 None。
    """
    gaps = []
    for k in range(len(spine) - 1):
        a, b = spine[k], spine[k + 1]
        di = b - a
        if di <= 0:
            continue
        dd = (est[b] - est[a]).days
        if dd < 0:
            continue
        gaps.append(dd / di)
    if not gaps:
        return None
    gaps.sort()
    mid = len(gaps) // 2
    return gaps[mid] if len(gaps) % 2 else (gaps[mid - 1] + gaps[mid]) / 2


def _parse(rec: dict) -> dict:
    """解析一条记录的时间线索"""
    raw = extract_time_expressions(rec.get("text", ""))
    out = {"abs": None, "self": None, "event_past": None, "future": None, "raw": raw}
    for e in raw:
        if isinstance(e["value"], date):
            out["abs"] = e["value"]
            break
    for e in raw:
        if e["kind"] == "REL_DAY":
            w = e["raw"]
            if w in SELF_DAY:
                out["self"] = SELF_DAY[w]
            elif w in FUTURE_DAY:
                out["future"] = FUTURE_DAY[w]
    for e in raw:
        if e["kind"] == "OFFSET":
            _, n, unit = e["value"]
            d = _offset_days(n, unit)
            if d < 0:
                out["event_past"] = d
            elif d > 0:
                out["future"] = out["future"] or d
    return out


def infer_timestamps_v3(records: list[dict],
                        first_anchor: date | None = None,
                        default_interval_days: int = 7,
                        use_self_refine: bool = True,
                        extra_anchors: dict[int, date] | None = None,
                        *,
                        trust_text_dates: bool = True,
                        lock_extra_anchors: bool = False) -> list[dict]:
    """
    推断说话时间。

    返回每条记录新增：
        utter_est   : 点估计
        confidence  : 0~1
        method      : absolute / file_anchor / interpolated / self_refined / extrapolated
        evidence    : 依据原文

    参数
    ----
    extra_anchors : 外部粗锚点（如文件 mtime / 目录所属月份），{记录下标: 日期}。
                    lock_extra_anchors=True 时为锁定锚点，优先于正文绝对日期。
    trust_text_dates : 是否将正文识别到的绝对日期作为观察锚点。

    关键约束：绝对锚点是**硬约束**，后续任何步骤都不得移动它。
    """
    n = len(records)
    if n == 0:
        return []
    recs = [dict(r) for r in records]
    info = [_parse(r) for r in recs]

    est = [None] * n
    method = ["none"] * n
    conf = [0.0] * n
    evid = [""] * n
    is_anchor = [False] * n
    hard = [False] * n          # 硬锚点：精确绝对日期，任何步骤不得移动
                                # 软锚点（文件级粗时间）：可被硬锚点和单调约束修正

    # ---------- ① 绝对锚点（硬约束，后续不可改） ----------
    for i in range(n):
        if (trust_text_dates and info[i]["abs"] is not None
                and not (lock_extra_anchors and extra_anchors and i in extra_anchors)
                and not (lock_extra_anchors and i == 0 and first_anchor)):
            est[i] = info[i]["abs"]
            method[i] = "absolute"
            conf[i] = 1.0
            is_anchor[i] = True
            hard[i] = True
            evid[i] = next(e["raw"] for e in info[i]["raw"] if isinstance(e["value"], date))

    # ①b 外部粗锚点：只在正文没有绝对日期时生效，置信度较低，且可被单调约束修正
    if extra_anchors:
        for i, d in extra_anchors.items():
            if 0 <= i < n and not is_anchor[i]:
                est[i] = d
                method[i] = "file_anchor"
                conf[i] = 0.6
                is_anchor[i] = True
                evid[i] = "file_meta"
                hard[i] = lock_extra_anchors

    # 若给了 first_anchor 且首条无锚，则视首条为锚（模拟"文件时间已知"）
    if first_anchor and (lock_extra_anchors or not is_anchor[0]):
        est[0] = first_anchor
        method[0] = "absolute"
        conf[0] = 0.8
        is_anchor[0] = True
        hard[0] = True
        evid[0] = "first_record"

    hard_idx = [i for i in range(n) if hard[i]]
    known = [i for i in range(n) if is_anchor[i]]

    # ---------- ② 锚点之间：位置插值 ----------
    # 有精确锚点时，插值**只用精确锚点**；文件级粗锚点只负责补首尾外推区域，
    # 否则粗锚点（如只知道年份）会把精确锚点顶住，实测反而更差。
    if not known:
        # 完全无锚：只能按固定节拍推进（只保留相对顺序，绝对时间不可辨识）
        est[0] = date(2026, 1, 1)
        method[0] = "extrapolated"
        conf[0] = 0.1
        for i in range(1, n):
            est[i] = est[i - 1] + timedelta(days=default_interval_days)
            method[i] = "extrapolated"
            conf[i] = 0.1
    else:
        spine = hard_idx if hard_idx else known

        # 首尾外推用的「每天对应多少条记录」速率：优先从锚点间实际间距估计，
        # 估不出来才退回 default_interval_days。
        # 手调这个值很容易错一个量级（实测节拍取 7、真实间距 0.057 时，
        # 首尾各十几条记录的误差可达上百天，占全部误差的绝大部分）。
        rate = _estimate_rate(est, spine)
        if rate is None:
            rate = float(default_interval_days)

        for k in range(len(spine) - 1):
            a, b = spine[k], spine[k + 1]
            span = b - a
            for j in range(a + 1, b):
                frac = (j - a) / span
                est[j] = est[a] + timedelta(days=int(round((est[b] - est[a]).days * frac)))
                method[j] = "interpolated"
                conf[j] = 0.4
        # 首个锚点之前
        for j in range(spine[0] - 1, -1, -1):
            if is_anchor[j]:
                continue                      # 保留软锚点自身取值
            est[j] = est[spine[0]] - timedelta(days=int(round(rate * (spine[0] - j))))
            method[j] = "extrapolated"
            conf[j] = 0.2
        # 最后锚点之后
        for j in range(spine[-1] + 1, n):
            if is_anchor[j]:
                continue
            est[j] = est[spine[-1]] + timedelta(days=int(round(rate * (j - spine[-1]))))
            method[j] = "extrapolated"
            conf[j] = 0.2

        # ---------- ③ 自指偏移细化（仅在锚点区间内，作为软提示） ----------
        # 注意：相邻两句都是"今天/昨天"只能给出下界，不能给出确切差值，
        # 因此只在 [前锚点, 后锚点] 区间内生效，越界即丢弃，避免链式漂移。
        if use_self_refine:
            for k in range(len(known) - 1):
                a, b = known[k], known[k + 1]
                lo, hi = est[a], est[b]
                for i in range(a + 1, b):
                    if info[i]["self"] is not None and info[i - 1]["self"] is not None:
                        delta = info[i]["self"] - info[i - 1]["self"]
                        if delta <= 0:
                            continue
                        cand = est[i - 1] + timedelta(days=delta)
                        if lo <= cand <= hi:
                            est[i] = cand
                            method[i] = "self_refined"
                            conf[i] = min(0.75, conf[i] + 0.3)

    # ---------- ④ 单调约束（硬锚点绝不移动；软锚点可被修正） ----------
    # 前向：非硬锚点不得早于前一条
    prev = None
    for i in range(n):
        if not hard[i] and prev is not None and est[i] < prev:
            est[i] = prev
        prev = est[i]
    # 后向：非硬锚点不得晚于后一条（修正"硬锚点回退"造成的逆向）
    nxt = None
    for i in range(n - 1, -1, -1):
        if not hard[i] and nxt is not None and est[i] > nxt:
            est[i] = nxt
        nxt = est[i]

    for i, r in enumerate(recs):
        r["utter_est"] = est[i]
        r["confidence"] = round(conf[i], 2)
        r["method"] = method[i]
        r["evidence"] = evid[i]
    return recs

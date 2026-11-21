# -*- coding: utf-8 -*-
"""
时间戳推断模块：从无时间戳的历史对话中，仅依据文本中的时间线索，推断每条记录的时间。

设计原则
--------
1. **只用文本线索**：不依赖文件修改时间、不依赖外部元数据（可作为 L0 兜底）
2. **零 API 成本**：规则法完成，不需要调 LLM
3. **可变精度**：输出估计日期 + 置信度 + 依据，便于用户事后调整

支持的时间表达
--------------
绝对：2026年3月15日 / 2026-03-15 / 3月15日 / 3/15
相对日：今天 昨天 明天 前天 后天 大前天 大后天
相对周：这周 上周 下周 + 周一~周日/星期X
相对月/年：这个月 上个月 下个月 今年 去年 前年 明年
时段偏移：N天前/后 N个月前/后 N年前/后 N周前/后，半个月前，三天前
模糊：最近 前段时间 不久前 前几天（低置信度）
"""

import re
from datetime import date, timedelta

# ============================================================
# 中文数字解析
# ============================================================
_CN_DIGIT = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9,
}


def cn_to_int(s: str):
    """把中文数字（含"十""十一""二十三"）转成整数；失败返回 None"""
    if not s:
        return None
    s = s.strip()
    if s.isdigit():
        return int(s)
    if s == "十":
        return 10
    if "十" in s:
        left, _, right = s.partition("十")
        tens = _CN_DIGIT.get(left, 1) if left else 1
        ones = _CN_DIGIT.get(right, 0) if right else 0
        return tens * 10 + ones
    if all(c in _CN_DIGIT for c in s):
        v = 0
        for c in s:
            v = v * 10 + _CN_DIGIT[c]
        return v
    return None


# ============================================================
# 时间表达抽取
# ============================================================
# (正则, 类型, 处理说明)
PATTERNS = [
    # ---- 绝对日期 ----
    ("ABS_FULL", re.compile(r"(20\d{2})\s*[-/年]\s*(\d{1,2})\s*[-/月]\s*(\d{1,2})\s*[日号]?")),
    ("ABS_MD", re.compile(r"(?<!\d)(\d{1,2})\s*月\s*(\d{1,2})\s*[日号]")),
    ("ABS_YMD_CN", re.compile(r"(20\d{2})\s*年\s*([一二三四五六七八九十]{1,3})\s*月\s*([一二三四五六七八九十]{1,3})\s*[日号]")),
    # ---- 相对日 ----
    ("REL_DAY", re.compile(r"(大前天|前天|昨天|今天|明天|后天|大后天)")),
    # ---- 相对周 ----
    ("REL_WEEK", re.compile(r"(上周|这周|本周|下周|上个星期|这个星期|下个星期)\s*([一二三四五六日天]|周[一二三四五六日天])?")),
    # ---- 相对月/年 ----
    ("REL_MONTH", re.compile(r"(上个月|这个月|本月|下个月|上月|下月)")),
    ("REL_YEAR", re.compile(r"(前年|去年|今年|明年)")),
    # ---- 时段偏移 ----
    ("OFFSET", re.compile(r"([0-9一二三四五六七八九十两半]{1,3})\s*(天|日|周|个?星期|个?月|年)\s*(前|后|以前|以后|之前|之后)")),
    # ---- 模糊 ----
    ("VAGUE", re.compile(r"(最近|前段时间|不久前|前几天|前阵子|这两天)")),
]

REL_DAY_MAP = {
    "大前天": -3, "前天": -2, "昨天": -1, "今天": 0, "明天": 1, "后天": 2, "大后天": 3,
}
WEEKDAY = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6}
REL_WEEK_OFFSET = {"上周": -1, "上个星期": -1, "这周": 0, "本周": 0, "这个星期": 0,
                   "下周": 1, "下个星期": 1}
REL_MONTH_OFFSET = {"上个月": -1, "上月": -1, "这个月": 0, "本月": 0, "下个月": 1, "下月": 1}
REL_YEAR_OFFSET = {"前年": -2, "去年": -1, "今年": 0, "明年": 1}


def extract_time_expressions(text: str) -> list[dict]:
    """
    从一段文本里抽取所有时间线索。

    返回: [{"kind": 类型, "raw": 原文, "value": 结构化值, "confidence": 0~1}, ...]
    """
    out = []
    for kind, pat in PATTERNS:
        for m in pat.finditer(text):
            raw = m.group(0)
            conf = 0.9
            val = None

            if kind == "ABS_FULL":
                try:
                    val = date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
                except ValueError:
                    continue
                conf = 1.0
            elif kind == "ABS_YMD_CN":
                y, mo, d = cn_to_int(m.group(1)), cn_to_int(m.group(2)), cn_to_int(m.group(3))
                if None in (y, mo, d):
                    continue
                try:
                    val = date(y, mo, d)
                except ValueError:
                    continue
                conf = 1.0
            elif kind == "ABS_MD":
                val = ("MD", int(m.group(1)), int(m.group(2)))   # 缺年份，需靠锚点补
                conf = 0.7
            elif kind == "REL_DAY":
                val = ("DAY_OFFSET", REL_DAY_MAP[m.group(1)])
                conf = 0.85
            elif kind == "REL_WEEK":
                base = REL_WEEK_OFFSET.get(m.group(1), 0)
                wd = None
                if m.group(2):
                    wd = WEEKDAY.get(m.group(2).replace("周", "")[-1])
                val = ("WEEK", base, wd)
                conf = 0.7
            elif kind == "REL_MONTH":
                val = ("MONTH_OFFSET", REL_MONTH_OFFSET[m.group(1)])
                conf = 0.65
            elif kind == "REL_YEAR":
                val = ("YEAR_OFFSET", REL_YEAR_OFFSET[m.group(1)])
                conf = 0.6
            elif kind == "OFFSET":
                n_raw, unit, direction = m.group(1), m.group(2), m.group(3)
                n = 0.5 if n_raw == "半" else cn_to_int(n_raw)
                if n is None:
                    continue
                sign = -1 if direction in ("前", "以前", "之前") else 1
                base_unit = "week" if "周" in unit or "星期" in unit else (
                    "month" if "月" in unit else ("year" if unit == "年" else "day"))
                val = ("OFFSET", sign * n, base_unit)
                conf = 0.8
            elif kind == "VAGUE":
                val = ("VAGUE", m.group(1))
                conf = 0.3

            if val is not None:
                out.append({"kind": kind, "raw": raw, "value": val, "confidence": conf})
    return out


# ============================================================
# 时间线重建
# ============================================================
def _add_months(d: date, months: int) -> date:
    """加/减月份，自动处理月末溢出"""
    y = d.year + (d.month - 1 + months) // 12
    m = (d.month - 1 + months) % 12 + 1
    # 处理 2 月 30 日之类
    day = min(d.day, [31, 29 if (y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)) else 28,
                      31, 30, 31, 30, 31, 31, 30, 31, 30, 31][m - 1])
    return date(y, m, day)


def _resolve(expr: dict, anchor: date) -> date | None:
    """把一个时间表达解析成具体日期（相对 anchor）"""
    v = expr["value"]
    if isinstance(v, date):
        return v
    tag = v[0]
    if tag == "DAY_OFFSET":
        return anchor + timedelta(days=v[1])
    if tag == "MD":
        _, mo, dd = v
        try:
            return date(anchor.year, mo, dd)
        except ValueError:
            return None
    if tag == "WEEK":
        _, base, wd = v
        # 以 anchor 所在周的周一为基准
        monday = anchor - timedelta(days=anchor.weekday())
        target = monday + timedelta(weeks=base, days=(wd if wd is not None else 0))
        return target
    if tag == "MONTH_OFFSET":
        return _add_months(anchor, v[1])
    if tag == "YEAR_OFFSET":
        try:
            return date(anchor.year + v[1], anchor.month, min(anchor.day, 28))
        except ValueError:
            return None
    if tag == "OFFSET":
        _, n, unit = v
        if unit == "day":
            return anchor + timedelta(days=int(round(n)))
        if unit == "week":
            return anchor + timedelta(weeks=int(round(n)))
        if unit == "month":
            return _add_months(anchor, int(round(n)))
        if unit == "year":
            return _add_months(anchor, int(round(n * 12)))
    return None


# ============================================================
# 语义规则：区分「自指时间」与「未来指向」
# ============================================================
# 自指：描述这条记忆本身发生/被说出的时间，应当推进时间线锚点
SELF_REFERENTIAL = {"大前天", "前天", "昨天", "今天"}

# 未来指向：描述尚未发生的事件，记录时间≈锚点，不推进（事件时间另记）
FUTURE_POINTING = {"明天", "后天", "大后天"}


def infer_timestamps(records: list[dict],
                     default_start: date | None = None,
                     default_interval_days: int = 1) -> list[dict]:
    """
    对一串**无时间戳**的记录推断时间。

    参数:
        records: [{"text": "...", ...}, ...]，按原始顺序（视为时间顺序）
        default_start: 无任何锚点时的起点日期（默认 2026-01-01）
        default_interval_days: 无锚点时相邻记录的默认间隔

    返回: 每条记录追加
        infer_date : 推断的「记录时间」（这条记忆被说出的时间）
        event_date : 若文本描述的是未来事件，这里给出事件时间（否则为 None）
        confidence : 置信度
        evidence   : 依据的时间表达原文
        method     : absolute / relative / sequence
    """
    records = [dict(r) for r in records]
    anchor = default_start or date(2026, 1, 1)

    for r in records:
        r["_exprs"] = extract_time_expressions(r.get("text", ""))

    for r in records:
        exprs = r["_exprs"]
        r["event_date"] = None

        # ① 绝对日期：最强线索，直接推进锚点
        abs_dates = [e for e in exprs if isinstance(e["value"], date)]
        if abs_dates:
            d = abs_dates[0]["value"]
            r["infer_date"] = d
            r["confidence"] = 1.0
            r["evidence"] = abs_dates[0]["raw"]
            r["method"] = "absolute"
            anchor = d
            continue

        # ② 未来指向：记录时间≈锚点，事件时间=锚点+偏移
        def _future_offset(e):
            """判断是否为"未来指向"表达，返回天数偏移或 None"""
            if e["kind"] == "REL_DAY" and e["value"][0] == "DAY_OFFSET":
                return e["value"][1] if e["value"][1] > 0 else None
            if e["kind"] == "OFFSET":
                _, n, unit = e["value"]
                if n > 0 and unit == "day":
                    return int(round(n))
                if n > 0:      # 周/月/年的未来指向，按天粗估
                    return int(round(n * (7 if unit == "week" else (30 if unit == "month" else 365))))
            return None

        fut = None
        fut_days = None
        for e in exprs:
            d = _future_offset(e)
            if d:
                fut, fut_days = e, d
                break

        if fut:
            r["infer_date"] = anchor
            r["event_date"] = anchor + timedelta(days=fut_days)
            r["confidence"] = 0.85
            r["evidence"] = fut["raw"]
            r["method"] = "relative"
            continue

        # ③ 自指 / 过去偏移：解析后推进锚点
        resolved, best = None, None
        for e in sorted(exprs, key=lambda x: -x["confidence"]):
            if e["kind"] == "VAGUE":
                continue
            d = _resolve(e, anchor)
            if d:
                resolved, best = d, e
                break

        if resolved:
            r["infer_date"] = resolved
            r["confidence"] = best["confidence"] * 0.9
            r["evidence"] = best["raw"]
            r["method"] = "relative"
            # 只有"过去/当下"的表达推进时间线
            if not (best["kind"] == "OFFSET" and best["value"][1] > 0):
                anchor = resolved
        else:
            anchor = anchor + timedelta(days=default_interval_days)
            r["infer_date"] = anchor
            r["confidence"] = 0.15
            r["evidence"] = ""
            r["method"] = "sequence"
    return records

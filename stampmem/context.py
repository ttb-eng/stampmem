"""Render retrieved evidence without internal identifiers or arbitrary metadata.

This adapter supports metadata availability comparisons; it does not implement or
claim to reproduce the paper's frozen benchmark pipeline.
"""
import hashlib
import json
import secrets


def render_context(hits, *, include_time=True, include_estimated_day=False):
    """Return JSON evidence in input order, preserving all text and candidates.

    No truncation, retrieval, sorting or automatic event-time inference occurs here.
    Hiding time removes structured fields only; dates in the text remain intact.
    Inferred observation days are hidden by default. Known archive bounds and
    observed dates from non-inferred sources remain visible.
    Opt in to the inferred point with include_estimated_day=True.
    Never forwards raw ids, distances, embeddings, line_id or answer-status labels.
    Text is evidence data and can itself contain untrusted instructions.
    """
    evidence = []
    for hit in hits:
        item = {"text": hit["text"]}
        meta = hit.get("metadata") or {}
        if include_time:
            source = meta.get("time_source", "unverified_legacy")
            temporal = {"source": source}
            observed = meta.get("observed_at") or meta.get("created_at")
            exact_bound = (observed == meta.get("ts_period_start") == meta.get("ts_period_end"))
            if (observed and source != "synthetic_unanchored"
                    and (source != "inferred" or include_estimated_day or exact_bound)):
                temporal["observed_at"] = observed
            if meta.get("event_time"):
                temporal["event_time"] = meta["event_time"]
            for key in ("ts_confidence", "ts_method", "ts_period_start", "ts_period_end"):
                if key in meta:
                    temporal[key] = meta[key]
            item["time"] = temporal
        evidence.append(item)
    return json.dumps(evidence, ensure_ascii=False)


def _text_digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def render_context_with_sources(hits, *, excerpts=None, include_time=True, include_estimated_day=False):
    """Return (context JSON, private source map) for a single retrieval pass.

    Pass original hits. Optional excerpts is a server-side {hit id: text}
    mapping; each excerpt must describe only its own source hit. This function
    does not generate or verify summaries. Never pass the private map to a model.
    """
    hits = list(hits)
    ids = [hit.get("id") for hit in hits]
    if any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != len(ids):
        raise ValueError("出处视图需要每条原文都有唯一的记忆 ID")
    excerpts = {} if excerpts is None else excerpts
    if not set(excerpts) <= set(ids) or any(not isinstance(value, str) or not value.strip()
                                            for value in excerpts.values()):
        raise ValueError("摘录必须对应本轮命中的原文，且不能为空")
    out = json.loads(render_context(hits, include_time=include_time, include_estimated_day=include_estimated_day))
    mapping = {}
    for row,hit,mid in zip(out,hits,ids,strict=True):
        ref = "src_" + secrets.token_urlsafe(12)
        while ref in mapping:
            ref = "src_" + secrets.token_urlsafe(12)
        mapping[ref] = (mid, _text_digest(hit["text"]))
        row["source_ref"] = ref
        row["text"] = excerpts.get(mid, hit["text"])
    return json.dumps(out, ensure_ascii=False), mapping

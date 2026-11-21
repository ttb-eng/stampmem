# -*- coding: utf-8 -*-
"""
stampmem 核心：统一记忆模型与时间序检索。

本版只实现两个核心能力：
  1. add()    —— 存一条记忆，自动附带时间戳
  2. search() —— 按语义检索记忆（后续版本会加"时间过滤 + 失效判断"）

设计思路：
  - 每条记忆 = 文本 + 向量 + 时间戳(元数据)
  - 用 ChromaDB 存向量，用 metadata 存时间戳
"""

from datetime import datetime, timezone
import json
import time
import uuid

import chromadb

from .context import _text_digest, render_context


def parse_ts(s) -> datetime:
    """
    解析时间字符串为带时区的 datetime。
    支持：'2026-01-15'、'2026-01-15 09:30'、ISO 字符串。
    """
    if isinstance(s, datetime):
        dt = s
    else:
        s = str(s).strip()
        dt = None
        for fmt in ("%Y-%m-%d %H:%M", "%Y-%m-%d", "%Y-%m-%dT%H:%M:%S"):
            try:
                dt = datetime.strptime(s, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


class Memory:
    """带时间戳的长期记忆（HNSW 向量索引 + 时间戳元数据）"""

    def __init__(self, collection_name: str = "agent_memory", embedder=None,
                 persist_dir: str = "./memory_db",
                 hnsw_space: str = "cosine",
                 hnsw_M: int = 16,
                 hnsw_ef_construction: int = 200,
                 hnsw_ef_search: int = 100):
        """
        HNSW 参数（ChromaDB 底层即 HNSW 图索引）:
            hnsw_M                : 每个节点的邻居数，越大越准越占内存（默认16）
            hnsw_ef_construction  : 建索引时的候选队列长度，越大建得越慢但图更好
            hnsw_ef_search        : 查询时的候选队列长度，越大召回越高但更慢
            hnsw_space            : 距离度量，cosine / l2 / ip
        """
        self.embedder = embedder if embedder is not None else DefaultEmbedder()
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.hnsw_config = {
            "hnsw:space": hnsw_space,
            "hnsw:M": hnsw_M,
            "hnsw:construction_ef": hnsw_ef_construction,
            "hnsw:search_ef": hnsw_ef_search,
        }
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata=self.hnsw_config,
        )

    def add(self, text: str, mem_type: str = "fact", status: str = "active",
            confidence: float = 1.0, metadata: dict | None = None,
            created_at: str | None = None, *,
            observed_at: str | None = None, recorded_at: str | None = None,
            event_time: str | None = None, time_source: str | None = None) -> str:
        """
        存一条记忆（统一记忆模型），自动打时间戳，返回记忆 id。

        参数:
            text:       记忆内容（只进 embedding，不含时间戳）
            mem_type:   "event"(一次性事件，会过时) / "fact"(长期事实，永不过时)
            status:     "active"(有效) / "done"(已完成) / "stale"(已失效) / "conflict"(冲突)
            confidence: 置信度 0~1（多次提及可提高）
            metadata:   额外自定义元数据
            created_at: 指定创建时间（评测时用于"回填历史事件"，格式 YYYY-MM-DD 或 ISO）
            observed_at: 信息被观察/说出的时间；created_at 的明确语义别名
            recorded_at: 实际入库时间，默认当前 UTC，与历史观察时间分开
            event_time: 可选的事件发生时间点；不从观察时间自动填充
            time_source: captured / provided / inferred 等来源，不代表真实性保证
        """
        if created_at and observed_at and parse_ts(created_at) != parse_ts(observed_at):
            raise ValueError("created_at 与 observed_at 不一致")
        reserved = {"created_at", "updated_at", "observed_at", "recorded_at", "event_time", "time_source"}
        if metadata and reserved.intersection(metadata):
            raise ValueError("时间核心字段请通过显式参数传入，不能用 metadata 覆盖")
        mem_id = str(uuid.uuid4())
        recorded = parse_ts(recorded_at).isoformat() if recorded_at else datetime.now(timezone.utc).isoformat()
        ts = observed_at or created_at or recorded
        parse_ts(ts)  # Reject malformed timestamps before embedding or writing.
        if event_time is not None:
            parse_ts(event_time)

        # 完整字段：时间戳独立存元数据，不混入 embedding 文本
        meta = {
            "created_at": ts,
            "observed_at": ts,
            "recorded_at": recorded,
            "updated_at": recorded,
            "time_source": time_source or ("provided" if observed_at or created_at else "captured"),
            "type": mem_type,
            "status": status,
            "confidence": confidence,
        }
        if event_time is not None:
            meta["event_time"] = event_time
        if metadata:
            meta.update(metadata)

        self.collection.add(
            ids=[mem_id],
            documents=[text],                      # 只有内容进 embedding
            metadatas=[meta],                      # 时间戳/类型/状态 独立存
            embeddings=[self.embedder.embed(text)],
        )
        return mem_id

    def search(self, query: str, n_results: int = 5):
        """按语义检索最相关的记忆，返回 (文本, 元数据) 列表"""
        q_emb = self.embedder.embed(query)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
        )
        out = []
        # ChromaDB query 返回的是两层嵌套：res["ids"] = [[id1, id2, ...]]
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        for mem_id, doc, meta in zip(ids, docs, metas):
            out.append({"id": mem_id, "text": doc, "metadata": meta})
        return out

    def retrieve(self, query: str, n_results: int = 10,
                 now: str | None = None,
                 exclude_status: tuple = ("stale",),
                 return_metrics: bool = True,
                 order: str = "similarity", *, include_embeddings: bool = True) -> dict:
        """
        HNSW 向量检索 + 时间戳/状态过滤，并记录效率指标。

        参数:
            query          : 查询文本
            n_results      : 向量检索召回的候选数（Top-K）
            now            : 当前时间（用于判断记忆是否过期），默认取系统当前时间
            exclude_status : 检索后要排除的记忆状态（默认排除 stale）
            return_metrics : 是否返回耗时等指标
            include_embeddings : 默认保留候选向量。若只消耗文本/元数据，传 False，
                                 不从向量库取回候选向量，hits[].embedding 为 None。
            order          : 结果的呈现顺序
                             "similarity"（默认）—— 按向量相似度，从最像到最不像
                             "chronological"    —— **同一批候选**改成按 created_at 从早到晚

        时间排序只改变呈现，不承诺回答增益。当前公共基准的一条路径上未检出
        排序增量；旧自建数据结果不能作为所有用户场景的效果保证。

        返回:
            {"hits": [...], "metrics": {"embed_ms","search_ms","filter_ms","total_ms",
                                        "recalled","kept","library_size"}}
        """
        if order not in ("similarity", "chronological"):
            raise ValueError("order 只能是 'similarity' 或 'chronological'")
        if n_results < 1:
            raise ValueError("n_results 必须大于0")
        t0 = time.perf_counter()

        # ① embedding（把查询变成向量）
        t_emb = time.perf_counter()
        q_emb = self.embedder.embed(query)
        embed_ms = (time.perf_counter() - t_emb) * 1000

        # ② HNSW 近似最近邻检索（库内部走 HNSW 图索引，非全量线性扫描）
        t_search = time.perf_counter()
        fields = ["documents", "metadatas", "distances"]
        if include_embeddings:
            fields.append("embeddings")
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=n_results,
            include=fields,
        )
        search_ms = (time.perf_counter() - t_search) * 1000

        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        ids = (res.get("ids") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        raw_embs = res.get("embeddings") if include_embeddings else None
        embs = raw_embs[0] if raw_embs is not None and len(raw_embs) else []

        recalled = []
        for i, (mem_id, doc, meta) in enumerate(zip(ids, docs, metas)):
            emb = None
            try:
                if embs is not None and i < len(embs) and embs[i] is not None:
                    emb = [float(x) for x in embs[i]]
            except Exception:
                emb = None
            recalled.append({
                "id": mem_id,
                "text": doc,
                "metadata": meta,
                "distance": dists[i] if i < len(dists) else None,
                "embedding": emb,
            })

        # ③ 时间戳/状态过滤（只对召回的少量候选做，不扫全库）
        t_filter = time.perf_counter()
        now_dt = parse_ts(now) if now else datetime.now(timezone.utc)
        kept = []
        for item in recalled:
            meta = item["metadata"] or {}
            if meta.get("status") in exclude_status:
                continue
            # 事件型记忆：若设置了 valid_until 且已过期，则过滤
            vu = meta.get("valid_until")
            if vu:
                try:
                    if now_dt > parse_ts(vu):
                        continue
                except Exception:
                    pass
            kept.append(item)
        filter_ms = (time.perf_counter() - t_filter) * 1000

        # ④ 呈现顺序（只改顺序，不增删候选）
        if order == "chronological":
            def temporal_key(hit):
                meta = hit.get("metadata") or {}
                value = meta.get("observed_at") or meta.get("created_at")
                return (value is None or value == "", parse_ts(value) if value else datetime.max.replace(tzinfo=timezone.utc))
            kept.sort(key=temporal_key)
        elif order != "similarity":
            raise ValueError(
                f"order 只能是 'similarity' 或 'chronological'，收到 {order!r}")

        total_ms = (time.perf_counter() - t0) * 1000
        metrics = {
            "embed_ms": round(embed_ms, 3),
            "search_ms": round(search_ms, 3),
            "filter_ms": round(filter_ms, 3),
            "total_ms": round(total_ms, 3),
            "recalled": len(recalled),
            "kept": len(kept),
            "library_size": self.count(),
        }
        return {"hits": kept, "metrics": metrics if return_metrics else {}}

    def read_source(self, reference: str, *, source_refs, include_time: bool = True,
                    include_estimated_day: bool = False, now: str | None = None) -> dict | None:
        """Read one original record under a server-provided, request-local map.

        Return None for out-of-scope, deleted, changed, stale or expired sources.
        This does not do vector search or call an LLM. Applications must scope the
        mapping to the authorized user, collection and request, and cap calls.
        """
        if not isinstance(reference, str) or reference not in source_refs:
            return None
        mid,digest = source_refs[reference]
        result = self.collection.get(ids=[mid],include=["documents","metadatas"])
        if result.get("ids") != [mid]:
            return None
        original = (result.get("documents") or [None])[0]
        meta = (result.get("metadatas") or [{}])[0] or {}
        if not isinstance(original,str) or _text_digest(original)!=digest or meta.get("status")=="stale":
            return None
        if meta.get("valid_until"):
            try:
                expires = parse_ts(meta["valid_until"])
            except (TypeError,ValueError,OverflowError):
                return None
            if (parse_ts(now) if now is not None else datetime.now(timezone.utc))>expires:
                return None
        row = json.loads(render_context([{"text":original,"metadata":meta}],include_time=include_time,
                                         include_estimated_day=include_estimated_day))[0]
        row["source_ref"] = reference
        return row

    def count(self) -> int:
        """库中记忆总数"""
        try:
            return self.collection.count()
        except Exception:
            return -1

    def check(self, query: str, llm_judge=None, n_results: int = 3):
        """
        检索相关记忆，判断状态，返回自然语言确认。

        返回:
            dict: {
                "memory": 最相关的记忆文本,
                "metadata": 该记忆的元数据(含时间戳/类型/状态),
                "judgment": {"status": ..., "reply": ...}  # LLM 判断结果
            }
        """
        candidates = self.search(query, n_results=n_results)
        if not candidates:
            return {"memory": None, "metadata": None,
                    "judgment": {"status": "none", "reply": ""}}

        top = candidates[0]  # 最相关的记忆
        meta = top["metadata"]

        # 情况1：事实(fact) 永不过时，直接返回，不需要判断
        if meta.get("type") == "fact":
            return {"memory": top["text"], "metadata": meta,
                    "judgment": {"status": "active", "reply": ""}}

        # 情况2：事件(event) 且已完成(done)，直接返回
        if meta.get("status") == "done":
            return {"memory": top["text"], "metadata": meta,
                    "judgment": {"status": "done", "reply": ""}}

        # 情况3：事件且 active，用 LLM 判断是否过时 + 生成自然确认
        if llm_judge is not None:
            judgment = llm_judge.judge(
                memory_text=top["text"],
                created_at=meta.get("created_at", ""),
                current_text=query,
            )
        else:
            judgment = {"status": "active", "reply": ""}

        return {"memory": top["text"], "metadata": meta, "judgment": judgment}

    def update(self, mem_id: str, status: str = None, confidence: float = None,
               new_text: str = None) -> None:
        """更新一条记忆的状态/置信度/内容，并刷新 updated_at"""
        ts = datetime.now(timezone.utc).isoformat()
        update_kwargs = {}
        if new_text is not None:
            update_kwargs["documents"] = [new_text]
            update_kwargs["embeddings"] = [self.embedder.embed(new_text)]
        if status is not None or confidence is not None or new_text is not None:
            meta = self.collection.get(ids=[mem_id])["metadatas"][0] or {}
            if status is not None:
                meta["status"] = status
            if confidence is not None:
                meta["confidence"] = confidence
            meta["updated_at"] = ts
            update_kwargs["metadatas"] = [meta]
        if update_kwargs:
            self.collection.update(ids=[mem_id], **update_kwargs)

    def add_with_conflict_check(self, text: str, mem_type: str = "fact",
                                llm_judge=None, threshold: float = 0.5) -> dict:
        """
        存入新记忆前，先检测是否与已有记忆冲突。

        流程：
          1. 语义检索最相似的旧记忆
          2. 若相似度足够高且类型相同，用 LLM 判断是否冲突
          3. 根据判断：更新旧记忆 / 保留旧记忆 / 标记冲突

        返回：dict { 'action': 'added'/'updated'/'kept'/'conflict',
                     'conflict': {...} }
        """
        candidates = self.search(text, n_results=1)
        if not candidates:
            # 没有相似旧记忆，直接新增
            self.add(text, mem_type=mem_type)
            return {"action": "added"}

        old = candidates[0]

        # 没有 LLM 判断器，直接新增（不做冲突检测）
        if llm_judge is None:
            self.add(text, mem_type=mem_type)
            return {"action": "added"}

        # 用 LLM 判断是否冲突
        result = llm_judge.detect_conflict(
            old_memory=old["text"],
            new_memory=text,
        )

        action = result.get("action", "ask")
        is_conflict = result.get("is_conflict", False)

        if not is_conflict:
            # 不冲突，直接新增
            self.add(text, mem_type=mem_type)
            return {"action": "added", "conflict": result}

        # 冲突了，按 action 处理
        if action == "update":
            # 新替旧：旧记忆标记 stale，新增新记忆
            self.update(old["id"], status="stale")
            self.add(text, mem_type=mem_type, status="active")
            return {"action": "updated", "conflict": result,
                    "old": old["text"]}
        elif action == "keep":
            # 新记忆是气话：保留旧记忆，新记忆降低置信度
            self.add(text, mem_type=mem_type, status="active", confidence=0.3)
            return {"action": "kept", "conflict": result,
                    "old": old["text"]}
        else:
            # ask：两条都保留，新记忆标记 conflict
            self.add(text, mem_type=mem_type, status="conflict")
            return {"action": "conflict", "conflict": result,
                    "old": old["text"]}

    def forget(self, mem_id: str) -> None:
        """删除一条记忆（主动遗忘）"""
        self.collection.delete(ids=[mem_id])

    def cleanup(self, statuses: tuple = ("stale",)) -> int:
        """
        自动清理"过时/无效"记忆（默认清理 stale）。

        返回：清理掉的记忆数量。
        """
        # 取出所有记忆的元数据，找出 status 在 statuses 里的
        all_meta = self.collection.get()
        ids_to_delete = []
        for mem_id, meta in zip(all_meta.get("ids", []),
                                all_meta.get("metadatas", [])):
            if meta and meta.get("status") in statuses:
                ids_to_delete.append(mem_id)

        if ids_to_delete:
            self.collection.delete(ids=ids_to_delete)
        return len(ids_to_delete)


class DefaultEmbedder:
    """默认使用中文语义模型。仅显式设置 use_model=False 时使用测试哈希。"""

    MODEL_NAME = "BAAI/bge-small-zh-v1.5"

    @staticmethod
    def _enable_offline_if_cached(model_name: str) -> None:
        """
        模型已在本地 HF 缓存里时，强制走离线模式。

        否则 huggingface_hub 每次都会先发 HEAD 请求探测是否有更新；在无网环境下
        这会导致 5 次重试、每次超时 10 秒以上，表现为"脚本卡住几十秒甚至几分钟"，
        最后仍然回落到同一份缓存。既然缓存已在，就别再问了。
        用户若显式设置过 HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE，则尊重用户的设置。
        """
        import os
        if os.environ.get("HF_HUB_OFFLINE") or os.environ.get("TRANSFORMERS_OFFLINE"):
            return
        try:
            cache_root = os.environ.get("HF_HOME") or os.path.join(
                os.path.expanduser("~"), ".cache", "huggingface")
            d = os.path.join(cache_root, "hub",
                             "models--" + model_name.replace("/", "--"))
            if os.path.isdir(d):
                os.environ["HF_HUB_OFFLINE"] = "1"
                os.environ["TRANSFORMERS_OFFLINE"] = "1"
        except Exception:
            pass  # 探测失败就按默认行为走

    def __init__(self, use_model: bool = True, *, model_name: str | None = None):
        self.model = None
        if model_name is not None and not use_model:
            raise ValueError("model_name 需要启用语义模型")
        if use_model:
            model_name = self.MODEL_NAME if model_name is None else model_name
            try:
                # 必须在 import sentence_transformers 之前设置：
                # huggingface_hub 在 import 时就把 HF_HUB_OFFLINE 读成模块常量，
                # 之后再改环境变量对它已经无效。
                self._enable_offline_if_cached(model_name)
                from sentence_transformers import SentenceTransformer
                # 中文小模型，本地运行，不花钱
                self.model = SentenceTransformer(model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"无法加载语义模型 {model_name!r}。请检查依赖、网络或提供本地模型目录；"
                    "use_model=False 仅适合测试，不能用于正式检索。"
                ) from exc

    def embed(self, text: str):
        """把一段文本编码成向量（list[float]）。

        默认模型为 bge-small-zh-v1.5；只有显式测试模式才使用无语义哈希向量。
        """
        if self.model is not None:
            return self.model.encode(text).tolist()
        # 降级：用字符哈希生成一个固定长度向量（仅测试用，无语义）
        import hashlib
        h = int(hashlib.md5(text.encode("utf-8")).hexdigest(), 16)
        return [float((h >> i) & 1) for i in range(128)]

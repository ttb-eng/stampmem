# stampmem

为 Agent 存储、导入和检索带时间信息的记忆。旧记忆可从已知日期或档案时间范围估计观察时间；新记忆可自动记录观察与写入时间；检索后可向模型展示这些字段。

## 安装

需要 Python 3.10 或更高版本。可在虚拟环境中从 [PyPI](https://pypi.org/project/stampmem/) 安装：

```bash
python -m pip install stampmem
```

也可从 [GitHub Releases](https://github.com/ttb-eng/stampmem/releases) 下载 wheel，再用 `python -m pip install <wheel文件>` 安装。

建议用项目自己的 Python 虚拟环境。Windows / Python 3.13.5 的全新环境已从 PyPI 安装 0.4.4 及其依赖，`pip check` 和两个离线示例均通过。这个验证使用测试向量，未覆盖其他平台或语义模型首次下载。

从源码安装则在本目录运行 `python -m pip install .`。默认在本地运行 [BAAI/bge-small-zh-v1.5](https://huggingface.co/BAAI/bge-small-zh-v1.5)；首次使用需取得模型文件，不需要 LLM API 密钥。也可以传入已下载的 Sentence Transformers 模型目录：

```python
from stampmem import DefaultEmbedder, Memory

mem = Memory(
    persist_dir="./my_memory",
    embedder=DefaultEmbedder(model_name="./models/bge-small-zh-v1.5"),
)
```

模型加载失败会明确报错，并保留原始异常。只有开发测试才显式使用 `DefaultEmbedder(use_model=False)`，它生成无语义的哈希向量。已有库换向量模型时，请创建新集合并重新写入原文；不要把哈希、不同模型或不同维度的向量混在同一个集合里。

## 从旧记忆到新对话

```python
from stampmem import Memory, stamp_documents, to_memory_payload, render_context

mem = Memory(persist_dir="./my_memory")
history = stamp_documents(
    [{"id": "old", "text": "用户以前住在北京。"}],
    periods={"old": "2020-03"},
)
for payload in to_memory_payload(history):
    mem.add(**payload)

mem.add("用户说自己已搬到杭州。", event_time="2025-04-01")

hits = mem.retrieve("用户的居住历史", n_results=2, include_embeddings=False)["hits"]
with_time = render_context(hits, include_time=True)
without_time = render_context(hits, include_time=False)
```

- **观察时间** `observed_at` 是信息被说出或观察的时间。新输入没传该字段时默认用当前写入时间；导入历史时应明确传入已知时间或走上面的估计与转换接口。旧参数 `created_at` 仍可用。
- **写入时间** `recorded_at` 默认是实际入库时间。**事件时间** `event_time` 是可选的、由调用者提供的事件日期。本版不自动解析事件日期或区间；无时区的值按 UTC 解释。
- 历史估计带来源与可辨识标记。若 `periods` 提供可信观察区间，估计不得越界，渲染会保留上下界；估出来的日仍不是已确认日期。若输入顺序与区间矛盾，保留区间，不能保证输出日期按输入排序。正文日期默认不作观察锚点，因为它可能只是事件发生日。只有确认正文日期就是记录时间时才传 `trust_text_dates=True`。无可信锚点时 `to_memory_payload()` 默认拒绝导入；不要绕过它把估计值直接当作可靠观察日期。升级不会改写已存的历史估计，请从原档案重算。

默认按相关性呈现。传 `order="chronological"` 可将选出的候选按观察时间由早到晚呈现。它不等同于优先召回最新记忆。`render_context` 只展示允许的字段，不改正文或顺序；它不会代替内容脱敏或提示注入防护。

默认展示可信的时间范围与已知的确切时间，不把推断日写成模型看到的具体日期。确需展示推断日时使用 `render_context(hits, include_estimated_day=True)`。只需要正文和元数据时，`include_embeddings=False` 不取回候选向量，返回项中的 `embedding` 为 `None`；仍会用查询向量检索。这个开关不减少发送给模型的正文，降低上下文成本仍需要按场景控制候选数或正文长度。

跨进程重新打开相同的 `persist_dir` 和 `collection_name`，即可查询已写入的记忆，前提是使用同一个向量模型。更多示例见 [examples](examples/)；其中 `demo_temporal_workflow.py` 不需要网络或下载模型，使用测试哈希。

## 按出处补读原文（可选）

原文仍由 `Memory` 保存。只有调用方已经有可靠摘录时，才通过 `excerpts` 缩短给模型看的文本；不传时原样呈现。本库不自动生成摘录，也不会根据关键词触发补读。

```python
from stampmem import render_context_with_sources

# 原始 hits 来自 mem.retrieve(...)。这个例子不假定检索一定命中。
context, source_refs = render_context_with_sources(hits)
# 给模型的是 context；source_refs 留在应用侧。
# Agent 的工具收到模型请求的 source_ref 后，可以读回对应的一条。
original = mem.read_source(requested_ref, source_refs=source_refs)
```

如果有调用方制作的摘录，传 `excerpts={hit_id: excerpt_text}`：每条摘录只对应一个原始命中，保留同一条记录的时间。引用不能跨请求复用；私有映射由应用绑定到当前用户与集合、控制生命周期和回查次数。未知引用、内容被改写、删除或失效时返回 `None`。出处引用不是鉴权凭据；应用必须先做授权，内容也仍需防提示注入。

## 验证范围

运行 `python -m pip install -e ".[dev]"` 和 `python -m pytest`。自动化测试使用显式哈希或测试替身来验证 API 和数据契约；语义模型的实际加载和跨进程检索应另测。

论文的公开基准来自独立的冻结实验框架。库实现了时间记录、来源标记、可选排序和元数据显隐接口；论文的效果量不等于本库的端到端收益。历史估计精度、真实数据召回、延迟和答题增益需要在部署条件下测量。版本变更见 [CHANGELOG](CHANGELOG.md)。

## 在不同 Agent 中使用技能

[下载技能 ZIP](https://github.com/ttb-eng/stampmem/releases/download/v0.4.4/stampmem-skill.zip) 并解压，或从仓库复制完整的 `stampmem` 技能目录，保留 `SKILL.md` 和 `references/`。它采用 [Agent Skills 格式](https://agentskills.io/specification)，按客户端的项目级路径放置：

| Agent | 目标项目中的位置 |
| --- | --- |
| [Codex](https://developers.openai.com/codex/skills) | `.agents/skills/stampmem/` |
| [Claude Code](https://code.claude.com/docs/en/skills) | `.claude/skills/stampmem/` |
| [Gemini CLI](https://geminicli.com/docs/cli/skills/) | `.agents/skills/stampmem/` 或 `.gemini/skills/stampmem/` |
| 自建 Agent | 由宿主读取 `SKILL.md`，按需加载 `references/integration.md` |

技能提供操作说明。要执行记忆读写，宿主还需要能运行 Python，或把库接成受授权的工具。安装 Python 包后，按客户端的说明加载技能。上述目录来自各客户端文档，实际加载和调用尚未逐客户端验证。

## 离线示例

```bash
python examples/demo_temporal_workflow.py
python examples/demo_source_readback.py
```

示例使用测试向量和临时数据库，可在装好依赖后离线运行。源代码采用 [MIT 许可证](LICENSE)。

# Integration with stampmem 0.4.4

Install the supplied local wheel into the application's virtual environment. Use `DefaultEmbedder(model_name=...)` with a compatible local Sentence Transformers model directory if the default model is not available. An existing Chroma collection must keep the same embedding model; rebuild into a separate collection when changing models. `use_model=False` is only for offline smoke tests.

```python
from stampmem import Memory, render_context

mem = Memory(persist_dir="./authorized_memory", collection_name="user_memory")
hits = mem.retrieve("Where did the user previously live?", include_embeddings=False)["hits"]
context = render_context(hits)
```

For an authorized live write, call `mem.add(text)`; pass `observed_at` for a known historical observation and `event_time` only when independently known. For archive month ranges, use `stamp_documents(docs, periods=...)` → `to_memory_payload(...)` → `mem.add(**payload)`. If time is unknown, retain the source separately pending review. Only opt in to using dates from the body when they are confirmed to be observation dates. The default renderer hides inferred days; exact native/provided dates and trusted ranges remain available.

To wire a readback tool, start with original retrieval hits. Optionally provide server-side excerpts keyed by hit ID, with one excerpt per original record.

```python
from stampmem import render_context_with_sources

context, private_refs = render_context_with_sources(hits, excerpts=approved_excerpts)
# Send only context to the reader. In the authorized request's tool handler:
source = mem.read_source(requested_ref, source_refs=private_refs)
```

Pass no `excerpts` when the application has no reliable excerpt. A stale, deleted, rewritten, or unknown reference returns `None`. Keep any source text local unless the user's data policy and task authorize the chosen model provider. The references control lookup scope, not authentication. The library does not guarantee that excerpts keep all evidence or improve answer accuracy.

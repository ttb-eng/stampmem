"""Offline demo of one source read; no API or model download."""
import json
from tempfile import TemporaryDirectory

from stampmem import DefaultEmbedder, Memory, render_context_with_sources


def main():
    with TemporaryDirectory(prefix='stampmem_source_',ignore_cleanup_errors=True) as db:
        mem = Memory(collection_name='source_demo',persist_dir=db,embedder=DefaultEmbedder(use_model=False))
        mid = mem.add('我 5 月 12 日搬到杭州，先住在西湖区。',observed_at='2025-06-01',event_time='2025-05-12')
        hits = mem.retrieve('杭州搬家',n_results=1,include_embeddings=False)['hits']
        # This example withholds the structured time to illustrate a missing detail.
        compact, private = render_context_with_sources(hits,excerpts={mid:'用户搬到了杭州。'},include_time=False)
        ref = json.loads(compact)[0]['source_ref']  # In an Agent, supplied by a tool call.
        source = mem.read_source(ref,source_refs=private,include_time=False)
        assert source and source['text']=='我 5 月 12 日搬到杭州，先住在西湖区。'
        print(compact)
        print(json.dumps(source,ensure_ascii=False))
        assert mem.read_source('src_not_in_this_request',source_refs=private) is None


if __name__=='__main__':
    main()

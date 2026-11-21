"""Offline workflow demo; no API, embedding download, or accuracy claim."""
from tempfile import TemporaryDirectory

from stampmem import DefaultEmbedder, Memory, render_context, stamp_documents, to_memory_payload


def main():
    # Chroma may hold mmap files until process exit on Windows. Cleanup is best
    # effort for this disposable directory; never delete a user's persistent DB.
    with TemporaryDirectory(prefix='stampmem_demo_', ignore_cleanup_errors=True) as directory:
        mem = Memory(collection_name='temporal_demo', persist_dir=directory,
                     embedder=DefaultEmbedder(use_model=False))
        history = stamp_documents([{'id': 'old', 'text': '用户当时住在北京'}],
                                  periods={'old': '2020-03'})
        for payload in to_memory_payload(history):
            mem.add(**payload)
        mem.add('用户刚告知去年四月搬到杭州', event_time='2025-04-01')
        hits = mem.retrieve('用户的居住历史', n_results=2)['hits']
        print('With stored time metadata:')
        print(render_context(hits))
        print('Same evidence without structured time:')
        print(render_context(hits, include_time=False))
        try:
            to_memory_payload(stamp_documents([{'text': '完全没有时间锚点'}]))
        except ValueError as exc:
            print('Expected refusal to invent historical time:', exc)


if __name__ == '__main__':
    main()

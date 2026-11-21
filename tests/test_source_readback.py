import json

import pytest

from stampmem import DefaultEmbedder, Memory, render_context, render_context_with_sources


@pytest.fixture()
def mem(db_dir):
    return Memory(persist_dir=db_dir,embedder=DefaultEmbedder(use_model=False))


def original_hit(mem, text='I moved to Hangzhou on May 12.'):
    mid = mem.add(text,observed_at='2025-06-01',event_time='2025-05-12',metadata={'private_label':'not for model'})
    hit = mem.retrieve('where do I live?',n_results=1,include_embeddings=False)['hits'][0]
    assert mid == hit['id']
    return hit


def test_excerpt_readback_preserves_source_time_without_embedding(mem,monkeypatch):
    hit = original_hit(mem)
    context,refs = render_context_with_sources([hit],excerpts={hit['id']:'I moved to Hangzhou.'})
    compact = json.loads(context)[0]
    ref = compact['source_ref']
    assert compact['text']=='I moved to Hangzhou.'
    assert hit['id'] not in context and 'private_label' not in context
    assert 'source_ref' not in render_context([hit])
    def no_embedding(*args,**kwargs):
        raise AssertionError('source reads must not embed or search')
    monkeypatch.setattr(mem.embedder,'embed',no_embedding)
    original = mem.read_source(ref,source_refs=refs)
    assert original['text']==hit['text']
    assert original['time']==compact['time']
    assert original['source_ref']==ref
    assert set(original)=={'text','time','source_ref'}
    assert 'time' not in mem.read_source(ref,source_refs=refs,include_time=False)


def test_references_are_request_local_and_unknown_ref_performs_no_database_read(mem,monkeypatch):
    hit = original_hit(mem)
    _,first = render_context_with_sources([hit])
    _,other = render_context_with_sources([hit])
    ref = next(iter(first))
    assert ref not in other
    def no_read(*args,**kwargs):
        raise AssertionError('unknown ref must fail before any database read')
    monkeypatch.setattr(mem.collection.__class__,'get',no_read)
    assert mem.read_source(ref,source_refs=other) is None
    assert mem.read_source(hit['id'],source_refs=first) is None


@pytest.mark.parametrize('mutate',['changed','stale','deleted'])
def test_old_refs_do_not_show_changed_stale_or_deleted_memory(mem,mutate):
    hit = original_hit(mem)
    _,refs = render_context_with_sources([hit])
    if mutate=='changed':
        mem.update(hit['id'],new_text='New address')
    elif mutate=='stale':
        mem.update(hit['id'],status='stale')
    else:
        mem.forget(hit['id'])
    assert mem.read_source(next(iter(refs)),source_refs=refs) is None


def test_source_expiry_and_out_of_scope_excerpts(mem):
    mid = mem.add('Event',metadata={'valid_until':'2026-01-02'})
    hits = mem.retrieve('Event',now='2026-01-01')['hits']
    assert [h['id'] for h in hits]==[mid]
    with pytest.raises(ValueError,match='摘录'):
        render_context_with_sources(hits,excerpts={'not-in-results':'surprise'})
    _,refs = render_context_with_sources(hits)
    ref = next(iter(refs))
    assert mem.read_source(ref,source_refs=refs,now='2026-01-01')['text']=='Event'
    assert mem.read_source(ref,source_refs=refs,now='2026-01-03') is None

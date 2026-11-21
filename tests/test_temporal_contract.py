"""Temporal semantics, identifier isolation and UTC ordering without model downloads."""
import json
from datetime import datetime

import numpy as np
import pytest

from stampmem import Memory, render_context, stamp_documents, to_memory_payload


class Embedder:
    def embed(self, _):
        return [1.0, 0.0]


class Collection:
    def __init__(self):
        self.records = []

    def add(self, **kwargs):
        self.records.append(kwargs)

    def count(self):
        return len(self.records)

    def query(self, **kwargs):
        records = self.records[:kwargs['n_results']]
        return {'ids': [[r['ids'][0] for r in records]],
                'documents': [[r['documents'][0] for r in records]],
                'metadatas': [[r['metadatas'][0] for r in records]],
                'distances': [[0.0] * len(records)],
                'embeddings': np.array([[[1.0, 0.0] for r in records]])}


@pytest.fixture
def memory():
    value = Memory.__new__(Memory)
    value.collection = Collection()
    value.embedder = Embedder()
    return value


def test_live_input_separates_event_and_ingestion(memory):
    memory.add('Last year I moved.', event_time='2025-04-01')
    meta = memory.collection.records[0]['metadatas'][0]
    assert meta['observed_at'] == meta['recorded_at'] == meta['created_at']
    assert meta['event_time'] == '2025-04-01'
    assert meta['time_source'] == 'captured'


def test_import_preserves_past_observation_and_current_write(memory):
    memory.add('Historical message', observed_at='2020-01-01', recorded_at='2026-11-15')
    meta = memory.collection.records[0]['metadatas'][0]
    assert meta['created_at'] == meta['observed_at'] == '2020-01-01'
    assert meta['updated_at'].startswith('2026-11-15')
    assert meta['time_source'] == 'provided'
    assert 'event_time' not in meta


@pytest.mark.parametrize('kwargs', [
    {'created_at': '2020-01-01', 'observed_at': '2021-01-01'},
    {'observed_at': 'not-a-date'},
    {'event_time': 'not-a-date'},
    {'metadata': {'created_at': '2020-01-01'}},
])
def test_bad_time_never_writes(memory, kwargs):
    with pytest.raises(ValueError):
        memory.add('text', **kwargs)
    assert memory.collection.count() == 0


def test_order_compares_instants_and_accepts_array_embeddings(memory):
    memory.add('later UTC', observed_at='2026-01-01T01:00:00+00:00')
    memory.add('earlier UTC', observed_at='2026-01-01T08:00:00+08:00')
    similar = memory.retrieve('query', n_results=2)
    chronological = memory.retrieve('query', n_results=2, order='chronological')
    assert [h['text'] for h in chronological['hits']] == ['earlier UTC', 'later UTC']
    assert {h['id'] for h in similar['hits']} == {h['id'] for h in chronological['hits']}
    assert all(h['embedding'] == [1.0, 0.0] for h in chronological['hits'])
    assert memory.retrieve('query', return_metrics=False)['metrics'] == {}


def test_no_anchor_cannot_silently_become_absolute_date():
    stamped = stamp_documents([{'text': 'No date here'}])
    assert stamped[0]['time_identifiable'] is False
    with pytest.raises(ValueError):
        to_memory_payload(stamped)
    payload = to_memory_payload(stamped, allow_unanchored=True)
    assert payload[0]['time_source'] == 'synthetic_unanchored'
    rendered = render_context([{'text': payload[0]['text'], 'metadata': {
        'created_at': payload[0]['created_at'], 'time_source': payload[0]['time_source']}}])
    assert 'observed_at' not in json.loads(rendered)[0]['time']


def test_anchored_import_roundtrip(memory):
    stamped = stamp_documents([{'text': 'Undated note'}], periods={0: '2020-03'})
    memory.add(**to_memory_payload(stamped)[0])
    meta = memory.collection.records[0]['metadatas'][0]
    assert meta['observed_at'].startswith('2020-03')
    assert meta['time_source'] == 'inferred'
    assert meta['ts_method'] == 'file_anchor'


def test_renderer_preserves_text_and_hides_internal_labels():
    hits = [{'id': 'answer_secret', 'text': 'Moved in 2020.', 'metadata': {
        'created_at': '2021-01-01', 'line_id': 'oracle', 'has_answer': True}}]
    with_time = render_context(hits)
    without = render_context(hits, include_time=False)
    assert json.loads(with_time)[0]['text'] == json.loads(without)[0]['text']
    assert '2020' in without  # Dates in original text are intentionally preserved.
    assert json.loads(without) == [{'text': 'Moved in 2020.'}]
    assert all(secret not in with_time for secret in ['answer_secret', 'line_id', 'has_answer'])


def test_renderer_only_hides_uncertain_observation_points():
    hits = [
        {"text":"native","metadata":{"time_source":"native","observed_at":"2024-01-02"}},
        {"text":"estimated","metadata":{"time_source":"inferred","observed_at":"2024-02-14",
            "ts_period_start":"2024-02-01","ts_period_end":"2024-02-29","event_time":"2023-02-14"}},
        {"text":"archive exact","metadata":{"time_source":"inferred","observed_at":"2024-03-03",
            "ts_period_start":"2024-03-03","ts_period_end":"2024-03-03"}},
    ]
    a,b,c = json.loads(render_context(hits))
    assert a["time"]["observed_at"]=="2024-01-02"
    assert b["time"]=={"source":"inferred","event_time":"2023-02-14",
                      "ts_period_start":"2024-02-01","ts_period_end":"2024-02-29"}
    assert c["time"]["observed_at"]=="2024-03-03"
    assert json.loads(render_context(hits,include_estimated_day=True))[1]["time"]["observed_at"]=="2024-02-14"

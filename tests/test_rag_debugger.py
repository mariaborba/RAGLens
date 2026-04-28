"""
Testes unitários do RAGLens — cobertura da lógica local sem chamadas de API.
"""

import numpy as np
import pytest

from rag_debugger import ChunkResult, Document, RAGDebugger


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def debugger(monkeypatch):
    monkeypatch.setenv("SABIA_API_KEY", "fake-key")
    monkeypatch.setenv("DEEPINFRA_API_KEY", "fake-key")
    return RAGDebugger(top_k=2)


@pytest.fixture
def sample_docs():
    return [
        Document(text="O escore de Apgar avalia o recém-nascido.", source="a.pdf", chunk_idx=0),
        Document(text="Hipertensão gestacional ocorre após 20 semanas.", source="b.pdf", chunk_idx=0),
        Document(text="Aleitamento materno exclusivo até 6 meses.", source="c.pdf", chunk_idx=0),
    ]


# ── Chunking ──────────────────────────────────────────────────────────────────

def test_chunk_text_basic():
    text = " ".join(f"word{i}" for i in range(500))
    chunks = RAGDebugger._chunk_text(text, chunk_words=400, overlap_words=80)
    assert len(chunks) >= 2
    for c in chunks:
        assert len(c.split()) <= 400


def test_chunk_text_overlap():
    words = [f"w{i}" for i in range(100)]
    text = " ".join(words)
    chunks = RAGDebugger._chunk_text(text, chunk_words=50, overlap_words=10)
    # Second chunk should start 40 words after the first
    first_words = set(chunks[0].split())
    second_words = set(chunks[1].split())
    assert first_words & second_words  # there is overlap


def test_chunk_text_short_input():
    chunks = RAGDebugger._chunk_text("hello world", chunk_words=400, overlap_words=80)
    assert chunks == ["hello world"]


def test_chunk_text_empty():
    chunks = RAGDebugger._chunk_text("", chunk_words=400, overlap_words=80)
    assert chunks == []


# ── Normalização de docs ───────────────────────────────────────────────────────

def test_normalize_docs_strings():
    result = RAGDebugger._normalize_docs(["texto a", "texto b"])
    assert all(isinstance(d, Document) for d in result)
    assert result[0].text == "texto a"


def test_normalize_docs_dicts():
    result = RAGDebugger._normalize_docs([{"text": "x", "source": "src"}])
    assert result[0].source == "src"


def test_normalize_docs_filters_empty():
    result = RAGDebugger._normalize_docs(["", "   ", "válido"])
    assert len(result) == 1
    assert result[0].text == "válido"


def test_normalize_docs_mixed():
    doc = Document(text="doc obj", source="s")
    result = RAGDebugger._normalize_docs(["texto", {"text": "dict"}, doc])
    assert len(result) == 3


# ── Cosine similarity ─────────────────────────────────────────────────────────

def test_cosine_similarity_identical():
    v = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    matrix = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    scores = RAGDebugger._cosine_similarity(v, matrix)
    assert abs(scores[0] - 1.0) < 1e-5


def test_cosine_similarity_orthogonal():
    v = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([[0.0, 1.0]], dtype=np.float32)
    scores = RAGDebugger._cosine_similarity(v, matrix)
    assert abs(scores[0]) < 1e-5


def test_cosine_similarity_ranking():
    query = np.array([1.0, 0.0], dtype=np.float32)
    matrix = np.array([
        [0.0, 1.0],   # ortogonal → score ≈ 0
        [1.0, 0.0],   # idêntico  → score ≈ 1
        [0.7, 0.7],   # parcial   → score ≈ 0.7
    ], dtype=np.float32)
    scores = RAGDebugger._cosine_similarity(query, matrix)
    assert scores[1] > scores[2] > scores[0]


# ── ChunkResult ───────────────────────────────────────────────────────────────

def test_chunk_result_score_pct():
    c = ChunkResult(text="t", score=0.853, source="s", rank=1)
    assert c.score_pct == 85.3


@pytest.mark.parametrize("score,expected_keyword", [
    (0.95, "Altíssima"),
    (0.80, "Forte"),
    (0.65, "moderada"),
    (0.45, "Baixa"),
    (0.20, "muito baixa"),
])
def test_chunk_result_why(score, expected_keyword):
    c = ChunkResult(text="t", score=score, source="s", rank=1)
    assert expected_keyword.lower() in c.why().lower()


# ── RAGDebugger — gestão de chunks ────────────────────────────────────────────

def test_add_and_get_chunks(debugger, sample_docs):
    for d in sample_docs:
        debugger.add_chunk(d)
    assert len(debugger.get_chunks()) == 3


def test_clear_chunks(debugger, sample_docs):
    for d in sample_docs:
        debugger.add_chunk(d)
    debugger.clear_chunks()
    assert debugger.get_chunks() == []


def test_chunks_by_source(debugger, sample_docs):
    for d in sample_docs:
        debugger.add_chunk(d)
    by_source = debugger.chunks_by_source()
    assert by_source["a.pdf"] == 1
    assert by_source["b.pdf"] == 1


def test_load_from_folder_missing(debugger):
    stats = debugger.load_from_folder("/pasta/que/nao/existe")
    assert stats.files_found == 0
    assert len(stats.errors) == 1

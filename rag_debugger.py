"""
RAG Debugger — intercepta pipelines RAG e expõe scores, chunks e rastreabilidade.
"""

from __future__ import annotations

import asyncio
import hashlib
import io
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional, Union

import httpx
import numpy as np
import pypdf

logging.getLogger("pypdf").setLevel(logging.ERROR)

DEEPINFRA_BASE = "https://api.deepinfra.com/v1/openai"
MARITACA_BASE  = "https://chat.maritaca.ai/api"
DEFAULT_EMBEDDING_MODEL = "intfloat/multilingual-e5-large"
DEFAULT_LLM_MODEL       = "sabia-3"


# ── Tipos de dados ────────────────────────────────────────────────────────────

@dataclass
class Document:
    text: str
    source: str = ""
    chunk_idx: int = 0
    embedding: Optional[List[float]] = field(default=None, repr=False)


@dataclass
class ChunkResult:
    text: str
    score: float
    source: str
    rank: int

    @property
    def score_pct(self) -> float:
        return round(self.score * 100, 1)

    def why(self) -> str:
        if self.score >= 0.90:
            return "Altíssima similaridade semântica com a query."
        if self.score >= 0.75:
            return "Forte sobreposição semântica — tema central coincide."
        if self.score >= 0.60:
            return "Similaridade moderada — contexto parcialmente relevante."
        if self.score >= 0.40:
            return "Baixa similaridade — recuperado por falta de alternativas melhores."
        return "Similaridade muito baixa — revisar base de documentos."


@dataclass
class IngestStats:
    files_found: int = 0
    files_ok: int = 0
    files_skipped: int = 0
    files_error: int = 0
    chunks_added: int = 0
    errors: List[str] = field(default_factory=list)


@dataclass
class DebugResult:
    query: str
    chunks: List[ChunkResult]
    response: str
    timestamp: datetime
    embedding_model: str
    llm_model: str
    top_k: int
    elapsed_ms: float
    all_scores: List[float] = field(default_factory=list)
    total_chunks_in_base: int = 0

    def summary(self) -> str:
        lines = [
            f"Query   : {self.query}",
            f"Modelo  : {self.llm_model} | Embedding: {self.embedding_model}",
            f"Tempo   : {self.elapsed_ms:.0f} ms",
            f"Chunks  : {len(self.chunks)} recuperados de {self.total_chunks_in_base} disponíveis",
            "",
        ]
        for c in self.chunks:
            lines.append(f"  [{c.rank}] score={c.score:.3f} ({c.score_pct}%) | {c.source or 'sem fonte'}")
            lines.append(f"      {c.why()}")
            lines.append(f"      {c.text[:120]}{'...' if len(c.text) > 120 else ''}")
        lines += ["", f"Resposta: {self.response[:300]}{'...' if len(self.response) > 300 else ''}"]
        return "\n".join(lines)


# ── Core ──────────────────────────────────────────────────────────────────────

class RAGDebugger:
    """
    Envolve qualquer pipeline RAG adicionando visibilidade total sobre
    scores de similaridade, chunks selecionados e rastreabilidade da resposta.
    """

    def __init__(
        self,
        llm_api_key: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_model: str = DEFAULT_EMBEDDING_MODEL,
        llm_model: str = DEFAULT_LLM_MODEL,
        top_k: int = 3,
        llm_base_url: str = MARITACA_BASE,
        embedding_base_url: str = DEEPINFRA_BASE,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.llm_api_key       = llm_api_key       or os.environ["SABIA_API_KEY"]
        self.embedding_api_key = embedding_api_key or os.environ["DEEPINFRA_API_KEY"]
        self.embedding_model   = embedding_model
        self.llm_model         = llm_model
        self.top_k             = top_k
        self.llm_base_url       = llm_base_url.rstrip("/")
        self.embedding_base_url = embedding_base_url.rstrip("/")
        self.system_prompt = system_prompt or (
            "Você é um assistente preciso. Responda à pergunta baseando-se "
            "exclusivamente no contexto fornecido. Se a informação não estiver "
            "no contexto, diga explicitamente: 'Não encontrei esta informação no contexto.'"
        )
        self._chunks: List[Document] = []
        self._ingested_fps: set = set()
        self._history: List[DebugResult] = []

    # ── Ingestão de PDFs ──────────────────────────────────────────────────────

    @staticmethod
    def _extract_pdf_text(path: str) -> str:
        try:
            reader = pypdf.PdfReader(path, strict=False)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            return f"[ERRO ao ler {os.path.basename(path)}: {e}]"

    @staticmethod
    def _extract_pdf_bytes(data: bytes) -> str:
        try:
            reader = pypdf.PdfReader(io.BytesIO(data), strict=False)
            return "\n".join(p.extract_text() or "" for p in reader.pages)
        except Exception as e:
            return f"[ERRO: {e}]"

    @staticmethod
    def _chunk_text(text: str, chunk_words: int = 400, overlap_words: int = 80) -> List[str]:
        words = text.split()
        chunks, i = [], 0
        step = max(1, chunk_words - overlap_words)
        while i < len(words):
            seg = " ".join(words[i: i + chunk_words])
            if seg.strip():
                chunks.append(seg)
            i += step
        return chunks

    def load_from_folder(
        self,
        folder: str,
        chunk_words: int = 400,
        overlap_words: int = 80,
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
    ) -> IngestStats:
        """Lê todos os PDFs de `folder`, chunka e adiciona à base interna."""
        stats = IngestStats()

        if not os.path.isdir(folder):
            stats.errors.append(f"Pasta não encontrada: {folder}")
            return stats

        pdf_files = sorted(f for f in os.listdir(folder) if f.lower().endswith(".pdf"))
        stats.files_found = len(pdf_files)

        for i, fname in enumerate(pdf_files):
            fpath = os.path.join(folder, fname)
            if progress_callback:
                progress_callback(i, len(pdf_files), fname)
            try:
                fst = os.stat(fpath)
                fp = f"folder::{os.path.abspath(fpath)}::{fst.st_size}::{fst.st_mtime_ns}"
                if fp in self._ingested_fps:
                    stats.files_skipped += 1
                    continue

                raw = self._extract_pdf_text(fpath)
                if raw.startswith("[ERRO"):
                    stats.files_error += 1
                    stats.errors.append(raw)
                    continue

                clean = re.sub(r"\n{3,}", "\n\n", raw).strip()
                if len(clean) < 50:
                    stats.files_error += 1
                    stats.errors.append(f"{fname}: texto muito curto")
                    continue

                new_chunks = self._add_chunks(fname, self._chunk_text(clean, chunk_words, overlap_words))
                stats.chunks_added += new_chunks
                stats.files_ok += 1
                self._ingested_fps.add(fp)

            except Exception as e:
                stats.files_error += 1
                stats.errors.append(f"{fname}: {e}")

        return stats

    def load_from_pdf_bytes(
        self,
        name: str,
        data: bytes,
        chunk_words: int = 400,
        overlap_words: int = 80,
    ) -> int:
        """Processa PDF enviado como bytes (upload via browser). Retorna chunks adicionados."""
        digest = hashlib.sha1(data).hexdigest()
        fp = f"upload::{name}::{len(data)}::{digest}"
        if fp in self._ingested_fps:
            return 0

        raw = self._extract_pdf_bytes(data)
        if not raw.strip() or raw.startswith("[ERRO"):
            return 0

        clean = re.sub(r"\n{3,}", "\n\n", raw).strip()
        n = self._add_chunks(name, self._chunk_text(clean, chunk_words, overlap_words))
        self._ingested_fps.add(fp)
        return n

    def _add_chunks(self, source: str, texts: List[str]) -> int:
        for idx, text in enumerate(texts):
            self._chunks.append(Document(text=text, source=source, chunk_idx=idx))
        return len(texts)

    def add_chunk(self, doc: Document) -> None:
        """Adiciona um chunk já pronto (usado para restaurar do SQLite)."""
        self._chunks.append(doc)

    def get_chunks(self) -> List[Document]:
        return self._chunks

    def clear_chunks(self) -> None:
        self._chunks.clear()
        self._ingested_fps.clear()

    def chunks_by_source(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for c in self._chunks:
            counts[c.source] = counts.get(c.source, 0) + 1
        return counts

    # ── API calls ─────────────────────────────────────────────────────────────

    _MAX_EMBED_CHARS = 1800  # multilingual-e5-large: limite de 512 tokens
    _EMBED_BATCH     = 512   # máximo de itens por request na DeepInfra

    async def _embed_raw(self, texts: List[str], client: httpx.AsyncClient) -> np.ndarray:
        """Embeda uma lista de textos em batches de _EMBED_BATCH."""
        all_embeddings = []
        for i in range(0, len(texts), self._EMBED_BATCH):
            batch = [t[:self._MAX_EMBED_CHARS] for t in texts[i:i + self._EMBED_BATCH]]
            r = await client.post(
                f"{self.embedding_base_url}/embeddings",
                headers={"Authorization": f"Bearer {self.embedding_api_key}"},
                json={"model": self.embedding_model, "input": batch},
                timeout=60.0,
            )
            if not r.is_success:
                raise httpx.HTTPStatusError(
                    f"{r.status_code} — {r.text}", request=r.request, response=r
                )
            data = sorted(r.json()["data"], key=lambda x: x["index"])
            all_embeddings.extend([d["embedding"] for d in data])
        return np.array(all_embeddings, dtype=np.float32)

    async def embed_all_chunks(self) -> int:
        """Computa e armazena embeddings para todos os chunks sem embedding. Retorna total processado."""
        pending = [d for d in self._chunks if d.embedding is None]
        if not pending:
            return 0
        async with httpx.AsyncClient() as client:
            embs = await self._embed_raw([d.text for d in pending], client)
        for doc, emb in zip(pending, embs):
            doc.embedding = emb.tolist()
        return len(pending)

    async def _call_llm(self, query: str, chunks: List[ChunkResult], client: httpx.AsyncClient) -> str:
        context = "\n\n".join(
            f"[Chunk {c.rank} | score={c.score:.3f} | fonte: {c.source or 'N/A'}]\n{c.text}"
            for c in chunks
        )
        r = await client.post(
            f"{self.llm_base_url}/chat/completions",
            headers={"Authorization": f"Key {self.llm_api_key}"},
            json={
                "model": self.llm_model,
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": f"CONTEXTO:\n{context}\n\nPERGUNTA: {query}"},
                ],
                "temperature": 0.2,
                "max_tokens": 800,
            },
            timeout=60.0,
        )
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]

    # ── Similaridade ──────────────────────────────────────────────────────────

    @staticmethod
    def _cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
        q = query_vec / (np.linalg.norm(query_vec) + 1e-10)
        d = doc_matrix / (np.linalg.norm(doc_matrix, axis=1, keepdims=True) + 1e-10)
        return (d @ q).astype(float)

    # ── Interface principal ───────────────────────────────────────────────────

    async def query(
        self,
        text: str,
        docs: Optional[List[Union[str, dict, Document]]] = None,
    ) -> DebugResult:
        """
        Executa o pipeline RAG completo.
        Se `docs` for None, usa os chunks carregados via load_from_folder / load_from_pdf_bytes.
        """
        t0 = datetime.now()

        if docs is not None:
            normalized = self._normalize_docs(docs)
        else:
            normalized = self._chunks

        if not normalized:
            raise ValueError("Base de chunks vazia. Ingira documentos primeiro.")

        async with httpx.AsyncClient() as client:
            # Embeda só a query (1 item) — chunks já foram embedados na ingestão
            q_embs = await self._embed_raw([text], client)
            query_vec = q_embs[0]

            # Usa embeddings pré-computados; fallback inline se ausentes
            if all(d.embedding is not None for d in normalized):
                doc_matrix = np.array([d.embedding for d in normalized], dtype=np.float32)
            else:
                doc_embs = await self._embed_raw([d.text for d in normalized], client)
                doc_matrix = doc_embs

            scores = self._cosine_similarity(query_vec, doc_matrix)
            k = min(self.top_k, len(normalized))
            top_indices = np.argsort(scores)[::-1][:k]

            chunks = [
                ChunkResult(
                    text=normalized[i].text,
                    score=float(scores[i]),
                    source=normalized[i].source,
                    rank=rank + 1,
                )
                for rank, i in enumerate(top_indices)
            ]

            response = await self._call_llm(text, chunks, client)

        elapsed_ms = (datetime.now() - t0).total_seconds() * 1000
        result = DebugResult(
            query=text,
            chunks=chunks,
            response=response,
            timestamp=t0,
            embedding_model=self.embedding_model,
            llm_model=self.llm_model,
            top_k=k,
            elapsed_ms=elapsed_ms,
            all_scores=scores.tolist(),
            total_chunks_in_base=len(normalized),
        )
        self._history.append(result)
        return result

    def query_sync(self, text: str, docs=None) -> DebugResult:
        return asyncio.run(self.query(text, docs))

    def get_history(self, n: int = 10) -> List[DebugResult]:
        return self._history[-n:]

    @staticmethod
    def _normalize_docs(docs: List[Union[str, dict, Document]]) -> List[Document]:
        result = []
        for d in docs:
            if isinstance(d, str):
                result.append(Document(text=d))
            elif isinstance(d, dict):
                result.append(Document(text=d.get("text", ""), source=d.get("source", "")))
            else:
                result.append(d)
        return [d for d in result if d.text.strip()]

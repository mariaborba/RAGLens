"""
RAG Debugger — servidor FastAPI com histórico e base de chunks em SQLite.
"""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional

import aiosqlite
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, model_validator

from rag_debugger import Document, RAGDebugger

load_dotenv()

DB_PATH          = os.getenv("DB_PATH", "history.db")
TOP_K_DEFAULT    = int(os.getenv("TOP_K", "3"))
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
LLM_MODEL        = os.getenv("LLM_MODEL", "sabia-3")
RAG_FOLDER       = os.getenv("RAG_FOLDER", "rag")

_db: Optional[aiosqlite.Connection] = None
_debugger: Optional[RAGDebugger] = None


# ── Banco de dados ────────────────────────────────────────────────────────────

async def init_db(db: aiosqlite.Connection) -> None:
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS history (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT    NOT NULL,
            query           TEXT    NOT NULL,
            response        TEXT    NOT NULL,
            chunks          TEXT    NOT NULL,
            all_scores      TEXT    NOT NULL,
            embedding_model TEXT    NOT NULL,
            llm_model       TEXT    NOT NULL,
            top_k           INTEGER NOT NULL,
            elapsed_ms      REAL    NOT NULL,
            total_chunks    INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS chunks (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            source      TEXT    NOT NULL,
            chunk_idx   INTEGER NOT NULL,
            text        TEXT    NOT NULL,
            embedding   TEXT,
            ingested_at TEXT    NOT NULL
        );
    """)
    # Migração: adiciona coluna embedding se não existir (banco antigo)
    try:
        await db.execute("ALTER TABLE chunks ADD COLUMN embedding TEXT")
        await db.commit()
    except Exception:
        pass


async def save_result(db: aiosqlite.Connection, result) -> int:
    chunks_json = json.dumps([
        {
            "text":      c.text,
            "score":     c.score,
            "score_pct": c.score_pct,
            "source":    c.source,
            "rank":      c.rank,
            "why":       c.why(),
        }
        for c in result.chunks
    ], ensure_ascii=False)
    cursor = await db.execute(
        """INSERT INTO history
           (timestamp,query,response,chunks,all_scores,embedding_model,llm_model,top_k,elapsed_ms,total_chunks)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (result.timestamp.isoformat(), result.query, result.response, chunks_json,
         json.dumps(result.all_scores), result.embedding_model, result.llm_model,
         result.top_k, result.elapsed_ms, result.total_chunks_in_base),
    )
    await db.commit()
    return cursor.lastrowid


async def save_chunks(db: aiosqlite.Connection, docs: List[Document]) -> None:
    now = datetime.now().isoformat()
    await db.executemany(
        "INSERT INTO chunks (source, chunk_idx, text, embedding, ingested_at) VALUES (?,?,?,?,?)",
        [
            (
                d.source,
                d.chunk_idx,
                d.text,
                json.dumps(d.embedding) if d.embedding is not None else None,
                now,
            )
            for d in docs
        ],
    )
    await db.commit()


async def load_chunks_from_db(db: aiosqlite.Connection) -> List[Document]:
    db.row_factory = aiosqlite.Row
    async with db.execute(
        "SELECT source, chunk_idx, text, embedding FROM chunks ORDER BY source, chunk_idx"
    ) as cur:
        docs = []
        async for r in cur:
            emb = json.loads(r["embedding"]) if r["embedding"] else None
            docs.append(Document(
                text=r["text"],
                source=r["source"],
                chunk_idx=r["chunk_idx"],
                embedding=emb,
            ))
        return docs


async def clear_chunks_db(db: aiosqlite.Connection) -> int:
    cursor = await db.execute("SELECT COUNT(*) FROM chunks")
    row = await cursor.fetchone()
    n = row[0]
    await db.execute("DELETE FROM chunks")
    await db.commit()
    return n


async def fetch_history(db: aiosqlite.Connection, n: int) -> list:
    db.row_factory = aiosqlite.Row
    async with db.execute("SELECT * FROM history ORDER BY id DESC LIMIT ?", (n,)) as cur:
        return [dict(r) async for r in cur]


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _db, _debugger
    _db = await aiosqlite.connect(DB_PATH)
    await init_db(_db)

    _debugger = RAGDebugger(
        embedding_model=EMBEDDING_MODEL,
        llm_model=LLM_MODEL,
        top_k=TOP_K_DEFAULT,
    )

    # Restaura chunks do banco — embeddings já vêm salvos, startup instantâneo
    saved = await load_chunks_from_db(_db)
    for doc in saved:
        _debugger.add_chunk(doc)

    yield
    await _db.close()


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="RAG Debugger",
    description="Intercepta pipelines RAG e visualiza chunks, scores e rastreabilidade.",
    version="0.3.0",
    lifespan=lifespan,
)


# ── Schemas ───────────────────────────────────────────────────────────────────

class DocInput(BaseModel):
    text: str
    source: str = ""


class DebugRequest(BaseModel):
    query: str
    top_k: int = TOP_K_DEFAULT
    embedding_model: str = EMBEDDING_MODEL
    llm_model: str = LLM_MODEL


class ChunkOut(BaseModel):
    text: str
    score: float
    score_pct: Optional[float] = None
    source: str
    rank: int
    why: str

    @model_validator(mode="after")
    def fill_score_pct(self):
        if self.score_pct is None:
            self.score_pct = round(self.score * 100, 1)
        return self


class DebugResponse(BaseModel):
    id: int
    query: str
    chunks: List[ChunkOut]
    response: str
    timestamp: str
    embedding_model: str
    llm_model: str
    top_k: int
    elapsed_ms: float
    all_scores: List[float]
    total_chunks_in_base: int


class HistoryItem(BaseModel):
    id: int
    timestamp: str
    query: str
    response: str
    chunks: List[ChunkOut]
    all_scores: List[float]
    embedding_model: str
    llm_model: str
    top_k: int
    elapsed_ms: float
    total_chunks: int


class IngestFolderRequest(BaseModel):
    folder: str = RAG_FOLDER
    chunk_words: int = 400
    overlap_words: int = 80


class DocumentsResponse(BaseModel):
    total_chunks: int
    sources: Dict[str, int]


# ── Endpoints — documentos ────────────────────────────────────────────────────

@app.get("/documents", response_model=DocumentsResponse, summary="Lista documentos ingeridos")
async def list_documents():
    return DocumentsResponse(
        total_chunks=len(_debugger.get_chunks()),
        sources=_debugger.chunks_by_source(),
    )


@app.post("/ingest/folder", summary="Ingere PDFs de uma pasta no servidor")
async def ingest_folder(req: IngestFolderRequest):
    before = len(_debugger.get_chunks())
    stats = _debugger.load_from_folder(req.folder, req.chunk_words, req.overlap_words)

    new_chunks = _debugger.get_chunks()[before:]
    if new_chunks:
        await _debugger.embed_all_chunks()
        await save_chunks(_db, new_chunks)

    return {
        "files_found":   stats.files_found,
        "files_ok":      stats.files_ok,
        "files_skipped": stats.files_skipped,
        "files_error":   stats.files_error,
        "chunks_added":  stats.chunks_added,
        "total_chunks":  len(_debugger.get_chunks()),
        "errors":        stats.errors,
    }


@app.post("/ingest/upload", summary="Ingere PDFs enviados pelo browser")
async def ingest_upload(
    files: List[UploadFile] = File(...),
    chunk_words: int  = Form(400),
    overlap_words: int = Form(80),
):
    results = []
    before = len(_debugger.get_chunks())

    for f in files:
        if not f.filename.lower().endswith(".pdf"):
            results.append({"file": f.filename, "chunks": 0, "error": "Não é PDF"})
            continue
        data = await f.read()
        n = _debugger.load_from_pdf_bytes(f.filename, data, chunk_words, overlap_words)
        results.append({"file": f.filename, "chunks": n, "error": None})

    new_chunks = _debugger.get_chunks()[before:]
    if new_chunks:
        await _debugger.embed_all_chunks()
        await save_chunks(_db, new_chunks)

    return {
        "files": results,
        "chunks_added": sum(r["chunks"] for r in results),
        "total_chunks": len(_debugger.get_chunks()),
    }


@app.delete("/documents", summary="Limpa toda a base de chunks")
async def delete_documents():
    n = await clear_chunks_db(_db)
    _debugger.clear_chunks()
    return {"deleted_chunks": n}


# ── Endpoints — debug e histórico ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("index.html")


@app.post("/debug", response_model=DebugResponse, summary="Executa pipeline RAG com debug completo")
async def debug(req: DebugRequest):
    if not _debugger.get_chunks():
        raise HTTPException(status_code=422, detail="Base vazia. Ingira documentos primeiro.")

    _debugger.top_k = req.top_k
    _debugger.llm_model = req.llm_model
    _debugger.embedding_model = req.embedding_model

    try:
        result = await _debugger.query(text=req.query)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))

    row_id = await save_result(_db, result)

    return DebugResponse(
        id=row_id,
        query=result.query,
        chunks=[ChunkOut(text=c.text, score=c.score, score_pct=c.score_pct,
                         source=c.source, rank=c.rank, why=c.why()) for c in result.chunks],
        response=result.response,
        timestamp=result.timestamp.isoformat(),
        embedding_model=result.embedding_model,
        llm_model=result.llm_model,
        top_k=result.top_k,
        elapsed_ms=result.elapsed_ms,
        all_scores=result.all_scores,
        total_chunks_in_base=result.total_chunks_in_base,
    )


@app.get("/history", response_model=List[HistoryItem], summary="Últimas N queries debugadas")
async def history(n: int = Query(default=10, ge=1, le=200)):
    rows = await fetch_history(_db, n)
    return [
        HistoryItem(
            id=r["id"], timestamp=r["timestamp"], query=r["query"], response=r["response"],
            chunks=[ChunkOut(**c) for c in json.loads(r["chunks"])],
            all_scores=json.loads(r["all_scores"]),
            embedding_model=r["embedding_model"], llm_model=r["llm_model"],
            top_k=r["top_k"], elapsed_ms=r["elapsed_ms"],
            total_chunks=r.get("total_chunks", 0),
        )
        for r in rows
    ]


@app.delete("/history", summary="Limpa todo o histórico")
async def clear_history():
    await _db.execute("DELETE FROM history")
    await _db.commit()
    return {"deleted": True}

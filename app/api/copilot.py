"""Endpoints behind the upload-and-ask screen: ask, reports, chunk lookup, status."""

from __future__ import annotations

import logging
import os

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field

from app.copilot.demo_llm import DemoLLM
from app.copilot.qa import Answer, answer_question
from app.copilot.retrieve import QUESTION_K, get_retriever, pool_for_report
from app.generation import LLMOutputError, LLMRefusal, Report, ReportGenerator, ReportKind, StructuredLLM, get_llm
from app.generation.types import ChunkLike, Citation
from app.models import Chunk
from app.storage import ChunkStore, get_store

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["copilot"])


class Source(BaseModel):
    """A retrieved chunk as the UI shows it."""

    citation: Citation
    text: str


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    document_ids: list[str] | None = None


class AskResponse(BaseModel):
    answer: Answer
    sources: list[Source]
    retrieval: str


class ReportRequest(BaseModel):
    kind: ReportKind
    document_ids: list[str] | None = None
    company: str | None = None


class ReportResponse(BaseModel):
    report: Report
    sources: list[Source]
    retrieval: str


def _provider() -> str:
    return os.environ.get("LLM_PROVIDER", "gemini").lower()


def _llm_problem() -> str | None:
    provider = _provider()
    if provider == "gemini" and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        return "No GEMINI_API_KEY set. Add it to .env and restart, or run with LLM_PROVIDER=demo."
    if provider == "anthropic" and not os.environ.get("ANTHROPIC_API_KEY"):
        return "No ANTHROPIC_API_KEY set."
    return None


_llm: StructuredLLM | None = None


def get_ui_llm() -> StructuredLLM:
    global _llm
    if _llm is None:
        problem = _llm_problem()
        if problem:
            raise HTTPException(503, problem)
        try:
            _llm = DemoLLM() if _provider() == "demo" else get_llm()
        except ImportError as exc:
            raise HTTPException(503, f"LLM SDK not installed: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(503, str(exc)) from exc
    return _llm


def _sources(chunks: list[ChunkLike]) -> list[Source]:
    return [Source(citation=Citation.from_chunk(c), text=c.text) for c in chunks]


def _run_llm(fn):
    try:
        return fn()
    except LLMRefusal as exc:
        raise HTTPException(502, f"The model declined: {exc}") from exc
    except LLMOutputError as exc:
        raise HTTPException(502, f"The model returned unusable output: {exc}") from exc
    except HTTPException:
        raise
    except Exception as exc:  # provider SDK errors: bad key, quota, network
        log.exception("LLM call failed")
        raise HTTPException(502, f"LLM call failed: {type(exc).__name__}: {str(exc)[:300]}") from exc


def _company(store: ChunkStore, document_ids: list[str] | None, given: str | None) -> str:
    if given and given.strip():
        return given.strip()
    docs = [d for d in store.list_documents() if not document_ids or d.document_id in document_ids]
    names = list(dict.fromkeys(d.company for d in docs if d.company))
    if len(names) == 1:
        return names[0]
    if names:
        return ", ".join(names)
    return "the company in these documents"


@router.get("/status")
def status(store: ChunkStore = Depends(get_store)) -> dict:
    retriever = get_retriever(store)
    problem = _llm_problem()
    model = None
    if not problem:
        from app.generation.llm import DEFAULT_GEMINI_MODEL, DEFAULT_MODEL

        defaults = {"demo": DemoLLM.model, "gemini": DEFAULT_GEMINI_MODEL, "anthropic": DEFAULT_MODEL}
        model = DemoLLM.model if _provider() == "demo" else os.environ.get("GENERATION_MODEL") or defaults.get(_provider())
    return {
        "retrieval": retriever.name,
        "llm_provider": _provider(),
        "llm_model": model,
        "llm_ready": problem is None,
        "llm_problem": problem,
        "documents": len(store.list_documents()),
    }


@router.post("/ask", response_model=AskResponse)
def ask(
    req: AskRequest,
    store: ChunkStore = Depends(get_store),
    llm: StructuredLLM = Depends(get_ui_llm),
) -> AskResponse:
    retriever = get_retriever(store)
    retriever.prepare()
    chunks = retriever.search(req.question, req.document_ids or None, QUESTION_K)
    answer = _run_llm(lambda: answer_question(llm, req.question, chunks))
    return AskResponse(answer=answer, sources=_sources(chunks), retrieval=retriever.name)


@router.post("/reports", response_model=ReportResponse)
def report(
    req: ReportRequest,
    store: ChunkStore = Depends(get_store),
    llm: StructuredLLM = Depends(get_ui_llm),
) -> ReportResponse:
    retriever = get_retriever(store)
    retriever.prepare()
    chunks = pool_for_report(retriever, req.kind, req.document_ids or None)
    company = _company(store, req.document_ids, req.company)
    result = _run_llm(lambda: ReportGenerator(llm).generate(req.kind, company, chunks))
    return ReportResponse(report=result, sources=_sources(chunks), retrieval=retriever.name)


@router.get("/chunks/{chunk_id}", response_model=Chunk)
def get_chunk(chunk_id: str, store: ChunkStore = Depends(get_store)) -> Chunk:
    chunk = store.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(404, "Chunk not found")
    return chunk

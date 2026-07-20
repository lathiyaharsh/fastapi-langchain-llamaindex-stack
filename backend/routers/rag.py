"""
RAG routes (LlamaIndex + Supabase pgvector).

Endpoints:
  POST /rag            → LlamaIndex chat engine (CONDENSE_PLUS_CONTEXT)
  DELETE /rag/session  → clear conversation memory (vectors stay)
  POST /rag/rebuild    → re-embed backend/data/
  POST /rag/upload     → incremental insert of one .md / .txt

Hybrid RAG lives in rag_hybrid.py (separate learning module).
Helpers and Pydantic models stay in main.py; this module owns HTTP wiring.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile


def create_rag_router() -> APIRouter:
    """
    Build RAG routes.

    Import main inside the factory so helpers exist when main.py calls this
    after defining RagRequest / get_rag_index (avoids circular-import issues).
    """
    import main

    router = APIRouter(tags=["rag"])

    @router.post("/rag", response_model=main.RagResponse)
    def rag(request: main.RagRequest):
        """
        Document Q&A with chat memory (Ask My Docs).

        Per request: condense question → retrieve top-k chunks → Groq answers.
        sources[] is returned for API clients; the web UI may hide it.
        """
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=400, detail="Question cannot be empty.")

        try:
            # Rehydrate engine memory from client when provided (survives reload)
            chat_engine = main.sync_rag_session_history(
                request.session_id, request.history
            )
            # chat() → condense question → retrieve top-k → answer with docs + history
            result = chat_engine.chat(question)
            sources = main._format_rag_sources(result.source_nodes)
            return main.RagResponse(answer=result.response, sources=sources)
        except HTTPException:
            raise
        except Exception as exc:
            raise main._http_error_for_llm(exc, kind="RAG") from exc

    @router.delete("/rag/session/{session_id}")
    def clear_rag_session(session_id: str):
        """Forget RAG conversation for one session (does not delete Supabase vectors)."""
        main.rag_sessions.pop(session_id, None)
        return {"cleared": session_id}

    @router.post("/rag/rebuild")
    def rag_rebuild():
        """
        Re-embed everything in backend/data/ and write fresh vectors to Supabase.

        Call this after editing .md files (no server restart needed).
        View result in Supabase: schema vecs → table ai_chat_docs
        """
        try:
            main.get_rag_index(force_rebuild=True)
            file_count = (
                len(list(main.DATA_DIR.glob("*"))) if main.DATA_DIR.exists() else 0
            )
            return {
                "status": "rebuilt",
                "vector_store": "supabase_pgvector",
                "collection": main.SUPABASE_COLLECTION,
                "data_dir": str(main.DATA_DIR),
                "files_seen": file_count,
                "chunk_size": main.RAG_CHUNK_SIZE,
                "chunk_overlap": main.RAG_CHUNK_OVERLAP,
            }
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"Rebuild error: {exc}") from exc

    @router.post("/rag/upload", response_model=main.RagUploadResponse)
    def rag_upload(file: UploadFile = File(...)):
        """
        Upload a new .md or .txt file and insert only that document into RAG.

        Sync (not async): LlamaIndex insertion calls sync wrappers internally that
        conflict with FastAPI's running event loop in an async route.

        Existing filenames return 409 because replacing a document must first
        remove its old vectors. Edit it in backend/data and use /rag/rebuild.
        """
        if not file.filename:
            raise HTTPException(status_code=400, detail="Filename is required")

        saved: Path | None = None
        try:
            # Initialize/load the existing index before saving, so an empty index
            # does not ingest the new file once here and again during insert().
            main._safe_upload_filename(file.filename)
            index = main.get_rag_index()
            saved = main._save_upload_to_data_dir(file)
            main._insert_uploaded_document(index, saved)
            file_count = (
                len(list(main.DATA_DIR.glob("*"))) if main.DATA_DIR.exists() else 0
            )
            return main.RagUploadResponse(
                status="uploaded",
                filename=saved.name,
                files_seen=file_count,
                data_dir=str(main.DATA_DIR),
            )
        except HTTPException:
            if saved is not None:
                saved.unlink(missing_ok=True)
            raise
        except Exception as exc:
            # Do not leave a file on disk that failed to reach the vector index.
            if saved is not None:
                saved.unlink(missing_ok=True)
            raise HTTPException(status_code=502, detail=f"Upload error: {exc}") from exc

    return router

"""
FastAPI tutorial lab — Part 3: SQL databases (SQLModel + SQLite).

Official docs: https://fastapi.tiangolo.com/tutorial/sql-databases/

Run from backend/ (port 8003):

    .venv/bin/uvicorn labs.fastapi_lab_part3:app --reload --port 8003
    # open http://127.0.0.1:8003/docs

Try:
  1. POST /notes/   {"title": "fridge", "body": "BANANA-42"}
  2. GET  /notes/
  3. GET  /notes/1
  4. PATCH /notes/1 {"body": "updated"}
  5. DELETE /notes/1

How this maps to your REAL stack:
  - Supabase pgvector = SQL for RAG vectors (already)
  - chat_sessions / rag_sessions = in-memory dicts (lost on restart)
    → SQLModel is how you'd persist those later
  - SQLModel ≈ Prisma / Drizzle: one model talks to DB + validates JSON

[JS] analogies in comments.
"""

from sqlmodel.orm.session import Session


from contextlib import asynccontextmanager
from typing import Annotated, Any, Generator

from fastapi import Depends, FastAPI, HTTPException, Query
from sqlmodel import Field, Session, SQLModel, create_engine, select

# =============================================================================
# MODELS — table=True means "this is a real DB table"
# =============================================================================
# [JS] Prisma: model Note { id Int @id @default(autoincrement()) title String ... }
# SQLModel is Pydantic + SQLAlchemy in one class.


class NoteBase(SQLModel):
    title: str = Field(index=True, min_length=1, max_length=120)
    body: str = Field(default="", max_length=4000)


class Note(NoteBase, table=True):
    """Row in the `note` table. id is assigned by SQLite on insert."""

    id: int | None = Field(default=None, primary_key=True)


class NoteCreate(NoteBase):
    """POST body — no id (client must not send a primary key)."""


class NoteUpdate(SQLModel):
    """PATCH body — every field optional (partial update)."""

    title: str | None = Field(default=None, min_length=1, max_length=120)
    body: str | None = Field(default=None, max_length=4000)


class NotePublic(NoteBase):
    """Response model — always includes id."""

    id: int


# =============================================================================
# ENGINE + SESSION (connection factory)
# =============================================================================
# [JS] like creating a PrismaClient or a pg Pool once at startup.
# SQLite = one file on disk (fine for labs). Production: Postgres URL.

SQLITE_PATH = "labs/lab_part3.db"
sqlite_url = f"sqlite:///{SQLITE_PATH}"
# check_same_thread=False: FastAPI may use the connection across threads
engine = create_engine(sqlite_url, connect_args={"check_same_thread": False})


def create_db_and_tables() -> None:
    SQLModel.metadata.create_all(engine)


def get_session() -> Generator[Session, Any, None]:
    """
    Dependency that opens a DB session per request, then closes it.

    `yield` = FastAPI runs the route, then finishes the `with` block (cleanup).
    [JS] like middleware that does `const db = await pool.connect(); try { next() } finally { db.release() }`
    """
    with Session(engine) as session:
        yield session


# Shorthand type for route signatures (tutorial style)
SessionDep = Annotated[Session, Depends(get_session)]


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create tables on startup (lab Part 1 lifespan idea applied to SQL)
    create_db_and_tables()
    yield


app = FastAPI(title="FastAPI lab Part 3 — SQL", lifespan=lifespan)


# =============================================================================
# CRUD ROUTES
# =============================================================================


@app.post("/notes/", response_model=NotePublic, status_code=201)
def create_note(note: NoteCreate, session: SessionDep) -> Note:
    # note is validated JSON → turn into a table row
    db_note = Note.model_validate(note)
    session.add(db_note)
    session.commit()  # write to SQLite file
    session.refresh(db_note)  # load generated id
    return db_note


@app.get("/notes/", response_model=list[NotePublic])
def list_notes(
    session: SessionDep,
    offset: int = 0,
    limit: Annotated[int, Query(le=50)] = 20,
) -> list[Note]:
    return list(session.exec(select(Note).offset(offset).limit(limit)).all())


@app.get("/notes/{note_id}", response_model=NotePublic)
def read_note(note_id: int, session: SessionDep) -> Note:
    note = session.get(Note, note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    return note


@app.patch("/notes/{note_id}", response_model=NotePublic)
def update_note(note_id: int, data: NoteUpdate, session: SessionDep) -> Note:
    note = session.get(Note, note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    # exclude_unset=True → only fields the client actually sent
    patch = data.model_dump(exclude_unset=True)
    note.sqlmodel_update(patch)
    session.add(note)
    session.commit()
    session.refresh(note)
    return note


@app.delete("/notes/{note_id}", status_code=204)
def delete_note(note_id: int, session: SessionDep) -> None:
    note = session.get(Note, note_id)
    if note is None:
        raise HTTPException(status_code=404, detail="Note not found")
    session.delete(note)
    session.commit()


@app.get("/")
def root():
    return {
        "lab": "part3-sql",
        "db_file": SQLITE_PATH,
        "try": ["POST /notes/", "GET /notes/", "GET /notes/{id}"],
        "real_app_map": {
            "vectors": "Supabase Postgres (pgvector) — already SQL",
            "chat memory": "in-process dict today — SQLModel could persist it",
        },
    }

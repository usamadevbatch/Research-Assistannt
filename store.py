"""SQLite-backed library: papers, chunks (with embeddings), chats, and notes."""
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, fields
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT UNIQUE NOT NULL,
    title TEXT NOT NULL,
    filename TEXT NOT NULL,
    num_pages INTEGER NOT NULL,
    added_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    page INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding BLOB NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(doc_id);
CREATE TABLE IF NOT EXISTS chats (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    doc_id INTEGER REFERENCES documents(id) ON DELETE CASCADE,  -- set for a paper's reader chat
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS messages (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id INTEGER NOT NULL REFERENCES chats(id) ON DELETE CASCADE,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_messages_chat ON messages(chat_id);
CREATE TABLE IF NOT EXISTS projects (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    idea TEXT NOT NULL,
    plan TEXT NOT NULL DEFAULT '{}',
    options TEXT NOT NULL DEFAULT '{}',
    report TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS candidates (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    project_id INTEGER NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
    key TEXT NOT NULL,
    meta TEXT NOT NULL,
    found_in TEXT NOT NULL DEFAULT '',
    relevance REAL,
    decision TEXT NOT NULL DEFAULT 'pending',  -- pending / include / maybe / exclude
    reason TEXT NOT NULL DEFAULT '',
    doc_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    fulltext TEXT NOT NULL DEFAULT '',          -- '' / pdf / abstract / failed
    UNIQUE(project_id, key)
);
CREATE TABLE IF NOT EXISTS xtables (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    doc_ids TEXT NOT NULL DEFAULT '[]',
    columns TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS xcells (
    table_id INTEGER NOT NULL REFERENCES xtables(id) ON DELETE CASCADE,
    doc_id INTEGER NOT NULL,
    col TEXT NOT NULL,
    value TEXT NOT NULL DEFAULT '',
    sources TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (table_id, doc_id, col)
);
CREATE TABLE IF NOT EXISTS jobs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    kind TEXT NOT NULL,
    ref_id INTEGER,
    status TEXT NOT NULL DEFAULT 'running',  -- running / done / failed / cancelled / interrupted
    progress REAL NOT NULL DEFAULT 0,
    message TEXT NOT NULL DEFAULT '',
    error TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS notes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id INTEGER REFERENCES documents(id) ON DELETE SET NULL,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    sources TEXT NOT NULL DEFAULT '[]',
    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
);
"""

# Bibliographic and reading-state columns added after the first release.
DOC_COLUMNS = {
    "authors": "TEXT NOT NULL DEFAULT ''",
    "year": "INTEGER",
    "venue": "TEXT NOT NULL DEFAULT ''",
    "doi": "TEXT NOT NULL DEFAULT ''",
    "arxiv_id": "TEXT NOT NULL DEFAULT ''",
    "abstract": "TEXT NOT NULL DEFAULT ''",
    "url": "TEXT NOT NULL DEFAULT ''",
    "tags": "TEXT NOT NULL DEFAULT ''",
    "status": "TEXT NOT NULL DEFAULT 'To read'",
    "last_page": "INTEGER NOT NULL DEFAULT 1",
}
EDITABLE = set(DOC_COLUMNS) | {"title"}
STATUSES = ["To read", "Reading", "Read"]


@dataclass(frozen=True)
class Document:
    id: int
    sha256: str
    title: str
    filename: str
    num_pages: int
    added_at: str
    num_chunks: int = 0
    authors: str = ""
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    abstract: str = ""
    url: str = ""
    tags: str = ""
    status: str = "To read"
    last_page: int = 1

    @property
    def short_authors(self) -> str:
        names = [a.strip() for a in self.authors.split(";") if a.strip()]
        if not names:
            return ""
        first = names[0].split()[-1]
        return first if len(names) == 1 else f"{first} et al." if len(names) > 2 else \
            f"{first} & {names[1].split()[-1]}"

    @property
    def citation(self) -> str:
        """Short label like 'Vaswani et al., 2017'."""
        parts = [p for p in (self.short_authors, str(self.year or "")) if p]
        return ", ".join(parts)


_DOC_FIELDS = {f.name for f in fields(Document)}


@dataclass
class ChunkSnapshot:
    doc_ids: np.ndarray     # (n,) int64
    pages: np.ndarray       # (n,) int32
    texts: list[str]
    embeddings: np.ndarray  # (n, dim) float32, L2-normalized


class Library:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)
            existing = {r[1] for r in conn.execute("PRAGMA table_info(documents)")}
            for col, ddl in DOC_COLUMNS.items():
                if col not in existing:
                    conn.execute(f"ALTER TABLE documents ADD COLUMN {col} {ddl}")
            cand_cols = {r[1] for r in conn.execute("PRAGMA table_info(candidates)")}
            if "decided_by" not in cand_cols:
                conn.execute("ALTER TABLE candidates ADD COLUMN decided_by TEXT NOT NULL DEFAULT ''")
            conn.execute("INSERT OR IGNORE INTO meta(key, value) VALUES ('revision', '0')")

    @contextmanager
    def _connect(self):
        # A fresh connection per operation keeps this safe across Streamlit threads.
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ---- meta -------------------------------------------------------------
    def get_meta(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set_meta(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )

    def revision(self) -> int:
        """Bumped on every change to chunks; used to invalidate the cached search index."""
        return int(self.get_meta("revision") or 0)

    @staticmethod
    def _bump(conn: sqlite3.Connection) -> None:
        conn.execute("UPDATE meta SET value = CAST(value AS INTEGER) + 1 WHERE key = 'revision'")

    # ---- documents --------------------------------------------------------
    def has_document(self, sha256: str) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM documents WHERE sha256 = ?", (sha256,)).fetchone() is not None

    def find_document(self, *, doi: str = "", arxiv_id: str = "") -> int | None:
        with self._connect() as conn:
            for col, val in (("doi", doi.lower()), ("arxiv_id", arxiv_id)):
                if val:
                    row = conn.execute(f"SELECT id FROM documents WHERE lower({col}) = ?", (val,)).fetchone()
                    if row:
                        return row[0]
        return None

    def add_document(self, sha256, title, filename, num_pages, pages, texts, embeddings, **meta) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO documents(sha256, title, filename, num_pages) VALUES (?, ?, ?, ?)",
                (sha256, title, filename, num_pages),
            )
            doc_id = cur.lastrowid
            conn.executemany(
                "INSERT INTO chunks(doc_id, page, text, embedding) VALUES (?, ?, ?, ?)",
                [
                    (doc_id, int(p), t, np.asarray(e, dtype=np.float32).tobytes())
                    for p, t, e in zip(pages, texts, embeddings)
                ],
            )
            self._bump(conn)
        if meta:
            self.update_document(doc_id, **meta)
        return doc_id

    def update_document(self, doc_id: int, **values) -> None:
        values = {k: v for k, v in values.items() if k in EDITABLE and v is not None}
        if not values:
            return
        cols = ", ".join(f"{k} = ?" for k in values)
        with self._connect() as conn:
            conn.execute(f"UPDATE documents SET {cols} WHERE id = ?", (*values.values(), doc_id))

    def delete_document(self, doc_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))
            self._bump(conn)

    def list_documents(self) -> list[Document]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT d.*, COUNT(c.id) AS num_chunks "
                "FROM documents d LEFT JOIN chunks c ON c.doc_id = d.id "
                "GROUP BY d.id ORDER BY d.added_at DESC, d.id DESC"
            ).fetchall()
        return [Document(**{k: r[k] for k in r.keys() if k in _DOC_FIELDS}) for r in rows]

    def get_document(self, doc_id: int) -> Document | None:
        return next((d for d in self.list_documents() if d.id == doc_id), None)

    def document_chunks(self, doc_id: int) -> list[tuple[int, str]]:
        """(page, text) for one document, in reading order."""
        with self._connect() as conn:
            return [tuple(r) for r in conn.execute(
                "SELECT page, text FROM chunks WHERE doc_id = ? ORDER BY id", (doc_id,)
            )]

    def load_snapshot(self) -> ChunkSnapshot:
        with self._connect() as conn:
            rows = conn.execute("SELECT doc_id, page, text, embedding FROM chunks ORDER BY id").fetchall()
        if not rows:
            return ChunkSnapshot(
                np.empty(0, np.int64), np.empty(0, np.int32), [], np.empty((0, 0), np.float32)
            )
        n = len(rows)
        return ChunkSnapshot(
            doc_ids=np.fromiter((r[0] for r in rows), dtype=np.int64, count=n),
            pages=np.fromiter((r[1] for r in rows), dtype=np.int32, count=n),
            texts=[r[2] for r in rows],
            embeddings=np.vstack([np.frombuffer(r[3], dtype=np.float32) for r in rows]),
        )

    # ---- chats ------------------------------------------------------------
    def create_chat(self, title: str, doc_id: int | None = None) -> int:
        with self._connect() as conn:
            return conn.execute("INSERT INTO chats(title, doc_id) VALUES (?, ?)", (title, doc_id)).lastrowid

    def paper_chat(self, doc_id: int, title: str) -> int:
        """The single persistent chat attached to a paper in the reader."""
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM chats WHERE doc_id = ?", (doc_id,)).fetchone()
        return row[0] if row else self.create_chat(title, doc_id)

    def list_chats(self, limit: int = 50) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT id, title, updated_at FROM chats WHERE doc_id IS NULL "
                "ORDER BY updated_at DESC, id DESC LIMIT ?", (limit,)
            )]

    def rename_chat(self, chat_id: int, title: str) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE chats SET title = ? WHERE id = ?", (title, chat_id))

    def delete_chat(self, chat_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def chat_exists(self, chat_id: int) -> bool:
        with self._connect() as conn:
            return conn.execute("SELECT 1 FROM chats WHERE id = ?", (chat_id,)).fetchone() is not None

    def messages(self, chat_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role, content, sources FROM messages WHERE chat_id = ? ORDER BY id", (chat_id,)
            ).fetchall()
        return [{**dict(r), "sources": json.loads(r["sources"])} for r in rows]

    def add_turn(self, chat_id: int, question: str, answer: str, sources: list[dict]) -> None:
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO messages(chat_id, role, content, sources) VALUES (?, ?, ?, ?)",
                [(chat_id, "user", question, "[]"), (chat_id, "assistant", answer, json.dumps(sources))],
            )
            conn.execute("UPDATE chats SET updated_at = CURRENT_TIMESTAMP WHERE id = ?", (chat_id,))

    def clear_chat(self, chat_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))

    # ---- notes ------------------------------------------------------------
    def add_note(self, title: str, content: str, doc_id: int | None = None, sources=()) -> int:
        with self._connect() as conn:
            return conn.execute(
                "INSERT INTO notes(doc_id, title, content, sources) VALUES (?, ?, ?, ?)",
                (doc_id, title, content, json.dumps(list(sources))),
            ).lastrowid

    def update_note(self, note_id: int, title: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "UPDATE notes SET title = ?, content = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (title, content, note_id),
            )

    def delete_note(self, note_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM notes WHERE id = ?", (note_id,))

    def replace_chunks(self, doc_id: int, sha256: str, filename: str, num_pages: int, pages, texts, embeddings) -> None:
        """Swap an abstract-only document's text for the full PDF once it's available."""
        with self._connect() as conn:
            conn.execute("DELETE FROM chunks WHERE doc_id = ?", (doc_id,))
            conn.execute("UPDATE documents SET sha256 = ?, filename = ?, num_pages = ? WHERE id = ?",
                         (sha256, filename, num_pages, doc_id))
            conn.executemany(
                "INSERT INTO chunks(doc_id, page, text, embedding) VALUES (?, ?, ?, ?)",
                [(doc_id, int(p), t, np.asarray(e, dtype=np.float32).tobytes())
                 for p, t, e in zip(pages, texts, embeddings)],
            )
            self._bump(conn)

    # ---- research projects --------------------------------------------------------
    def create_project(self, title: str, idea: str, options: dict) -> int:
        with self._connect() as conn:
            return conn.execute("INSERT INTO projects(title, idea, options) VALUES (?, ?, ?)",
                                (title, idea, json.dumps(options))).lastrowid

    def get_project(self, project_id: int) -> dict | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM projects WHERE id = ?", (project_id,)).fetchone()
        if not r:
            return None
        return {**dict(r), "plan": json.loads(r["plan"]), "options": json.loads(r["options"])}

    def list_projects(self) -> list[dict]:
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(
                "SELECT id, title, updated_at FROM projects ORDER BY updated_at DESC, id DESC")]

    def update_project(self, project_id: int, **values) -> None:
        for k in ("plan", "options"):
            if k in values:
                values[k] = json.dumps(values[k])
        values = {k: v for k, v in values.items() if k in ("title", "idea", "plan", "options", "report")}
        if values:
            cols = ", ".join(f"{k} = ?" for k in values)
            with self._connect() as conn:
                conn.execute(f"UPDATE projects SET {cols}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                             (*values.values(), project_id))

    def delete_project(self, project_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))

    def upsert_candidates(self, project_id: int, rows: list[tuple[str, dict, str]]) -> None:
        """rows: (key, meta dict, found_in). Existing rows get their meta/found_in refreshed, decisions kept."""
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO candidates(project_id, key, meta, found_in) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(project_id, key) DO UPDATE SET meta = excluded.meta, found_in = excluded.found_in",
                [(project_id, k, json.dumps(m), f) for k, m, f in rows],
            )

    def candidates(self, project_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM candidates WHERE project_id = ? "
                                "ORDER BY relevance IS NULL, relevance DESC, id", (project_id,)).fetchall()
        return [{**dict(r), "meta": json.loads(r["meta"])} for r in rows]

    def update_candidate(self, cand_id: int, **values) -> None:
        values = {k: v for k, v in values.items()
                  if k in ("relevance", "decision", "reason", "doc_id", "fulltext", "meta", "decided_by")}
        if "meta" in values:
            values["meta"] = json.dumps(values["meta"])
        if values:
            cols = ", ".join(f"{k} = ?" for k in values)
            with self._connect() as conn:
                conn.execute(f"UPDATE candidates SET {cols} WHERE id = ?", (*values.values(), cand_id))

    # ---- extraction tables ----------------------------------------------------------
    def create_table(self, name: str, doc_ids: list[int], columns: list[dict], project_id: int | None = None) -> int:
        with self._connect() as conn:
            return conn.execute("INSERT INTO xtables(name, project_id, doc_ids, columns) VALUES (?, ?, ?, ?)",
                                (name, project_id, json.dumps(doc_ids), json.dumps(columns))).lastrowid

    def get_table(self, table_id: int) -> dict | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM xtables WHERE id = ?", (table_id,)).fetchone()
        if not r:
            return None
        return {**dict(r), "doc_ids": json.loads(r["doc_ids"]), "columns": json.loads(r["columns"])}

    def list_tables(self, project_id: int | None = None) -> list[dict]:
        sql, args = "SELECT id, name, project_id, created_at FROM xtables", ()
        if project_id is not None:
            sql, args = sql + " WHERE project_id = ?", (project_id,)
        with self._connect() as conn:
            return [dict(r) for r in conn.execute(sql + " ORDER BY id DESC", args)]

    def update_table(self, table_id: int, *, name=None, doc_ids=None, columns=None) -> None:
        with self._connect() as conn:
            if name is not None:
                conn.execute("UPDATE xtables SET name = ? WHERE id = ?", (name, table_id))
            if doc_ids is not None:
                conn.execute("UPDATE xtables SET doc_ids = ? WHERE id = ?", (json.dumps(doc_ids), table_id))
            if columns is not None:
                conn.execute("UPDATE xtables SET columns = ? WHERE id = ?", (json.dumps(columns), table_id))

    def delete_table(self, table_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM xtables WHERE id = ?", (table_id,))

    def set_cell(self, table_id: int, doc_id: int, col: str, value: str, sources: list) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO xcells(table_id, doc_id, col, value, sources) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(table_id, doc_id, col) DO UPDATE SET value = excluded.value, sources = excluded.sources",
                (table_id, doc_id, col, value, json.dumps(sources)),
            )

    def cells(self, table_id: int) -> dict[tuple[int, str], dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM xcells WHERE table_id = ?", (table_id,)).fetchall()
        return {(r["doc_id"], r["col"]): {"value": r["value"], "sources": json.loads(r["sources"])} for r in rows}

    def clear_cells(self, table_id: int, doc_id: int | None = None) -> None:
        with self._connect() as conn:
            if doc_id is None:
                conn.execute("DELETE FROM xcells WHERE table_id = ?", (table_id,))
            else:
                conn.execute("DELETE FROM xcells WHERE table_id = ? AND doc_id = ?", (table_id, doc_id))

    # ---- background jobs --------------------------------------------------------------
    def create_job(self, kind: str, ref_id: int | None) -> int:
        with self._connect() as conn:
            return conn.execute("INSERT INTO jobs(kind, ref_id, message) VALUES (?, ?, 'Starting…')",
                                (kind, ref_id)).lastrowid

    def update_job(self, job_id: int, **values) -> None:
        values = {k: v for k, v in values.items() if k in ("status", "progress", "message", "error")}
        cols = ", ".join(f"{k} = ?" for k in values)
        with self._connect() as conn:
            conn.execute(f"UPDATE jobs SET {cols}, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                         (*values.values(), job_id))

    def get_job(self, job_id: int) -> dict | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(r) if r else None

    def latest_job(self, kind_prefix: str, ref_id: int) -> dict | None:
        with self._connect() as conn:
            r = conn.execute("SELECT * FROM jobs WHERE kind LIKE ? AND ref_id = ? ORDER BY id DESC LIMIT 1",
                             (kind_prefix + "%", ref_id)).fetchone()
        return dict(r) if r else None

    def interrupt_stale_jobs(self) -> None:
        """Jobs can't survive an app restart; mark leftovers so the UI doesn't wait forever."""
        with self._connect() as conn:
            conn.execute("UPDATE jobs SET status = 'interrupted', message = 'Stopped when the app restarted' "
                         "WHERE status = 'running'")

    def list_notes(self, doc_id: int | None = None) -> list[dict]:
        sql = "SELECT * FROM notes"
        args: tuple = ()
        if doc_id is not None:
            sql, args = sql + " WHERE doc_id = ?", (doc_id,)
        with self._connect() as conn:
            rows = conn.execute(sql + " ORDER BY updated_at DESC, id DESC", args).fetchall()
        return [{**dict(r), "sources": json.loads(r["sources"])} for r in rows]

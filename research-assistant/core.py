"""Shared resources, model settings, ingestion, and the chat UI used by every page."""
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import streamlit as st

import metadata
from config import DATA_DIR, DB_PATH, PDF_DIR, settings
from embeddings import Embedder, Reranker
from ingest import render_page
from llm import LLM, make_llm, ollama_models
from rag import SYSTEM_PROMPT, build_messages, condense_question
from retrieval import Hit, HybridIndex, rerank
from store import Document, Library

GEMINI_MODELS = ["gemini-2.5-flash", "gemini-2.5-flash-lite", "gemini-2.5-pro"]
RERANK_POOL = 30  # hybrid candidates handed to the cross-encoder
READER_PAGE = "views/reader.py"

CSS = """
<style>
  .block-container {padding-top: 2.2rem; max-width: 1200px;}
  [data-testid="stSidebar"] .block-container {padding-top: 1rem;}
  [data-testid="stChatMessage"] {border-radius: 14px; padding: 0.9rem 1rem;}
  [data-testid="stChatMessageContent"] table {font-size: 0.92rem;}
  .ra-hero h1 {font-size: 2.1rem; margin-bottom: 0;}
  .ra-muted {opacity: 0.7; font-size: 0.92rem;}
  .ra-meta {opacity: 0.75; font-size: 0.88rem; margin-top: -0.4rem;}
  .ra-stat {font-size: 1.6rem; font-weight: 700;}
</style>
"""


# ---- cached resources ---------------------------------------------------------

@st.cache_resource
def get_library() -> Library:
    return Library(DB_PATH)


@st.cache_resource(show_spinner="Loading embedding model (downloaded once on first run)...")
def get_embedder() -> Embedder:
    return Embedder(settings.embed_model, DATA_DIR / "models")


@st.cache_resource(show_spinner="Loading re-ranker (downloaded once on first run)...")
def get_reranker() -> Reranker | None:
    try:
        return Reranker(settings.rerank_model, DATA_DIR / "models")
    except Exception:
        return None  # search still works without it


@st.cache_resource(max_entries=1, show_spinner="Building search index...")
def get_index(revision: int) -> HybridIndex:
    return HybridIndex(get_library().load_snapshot())


@st.cache_data(max_entries=128, show_spinner=False)
def page_image(sha: str, page: int, dpi: int = 110) -> bytes | None:
    path = PDF_DIR / f"{sha}.pdf"
    return render_page(path, page, dpi) if path.exists() else None


def pdf_path(doc: Document):
    return PDF_DIR / f"{doc.sha256}.pdf"


def local_time(utc: str) -> str:
    """SQLite timestamps are UTC; show them in the computer's local time."""
    try:
        return datetime.fromisoformat(utc).replace(tzinfo=timezone.utc).astimezone().strftime("%b %d, %Y %H:%M")
    except ValueError:
        return utc


def docs() -> list[Document]:
    return get_library().list_documents()


def label(doc: Document) -> str:
    """Title plus short citation, used in prompts and source lists."""
    return f"{doc.title} ({doc.citation})" if doc.citation else doc.title


# ---- model settings (rendered in the sidebar on every page) --------------------

@dataclass
class ModelConfig:
    backend: str
    model: str
    api_key: str
    context_budget: int  # characters of paper text that fit in one prompt
    top_k: int
    rerank: bool

    def llm(self) -> LLM:
        return make_llm(self.backend, api_key=self.api_key, model=self.model, ollama_url=settings.ollama_url)


def model_sidebar() -> None:
    st.sidebar.markdown("#### 🤖 AI model")
    backend = st.sidebar.segmented_control(
        "Backend", ["Ollama (local)", "Gemini"], key="backend",
        default="Gemini" if settings.gemini_api_key else "Ollama (local)",
        help="Ollama runs on your computer: private, free, unlimited. Gemini is faster but rate limited.",
    ) or "Ollama (local)"
    if backend == "Gemini":
        api_key = st.sidebar.text_input(
            "Gemini API key", value=settings.gemini_api_key, type="password", key="gemini_key",
            help="Free key from https://aistudio.google.com/apikey",
        )
        options = list(dict.fromkeys([settings.gemini_model, *GEMINI_MODELS]))
        model = st.sidebar.selectbox(
            "Model", options, key="gemini_model", accept_new_options=True,
            help="Flash-Lite has the highest free daily quota. Type any other model name to use it.",
        )
        st.sidebar.caption("Google may use free-tier inputs. Use Ollama for unpublished work.")
        cfg = ModelConfig(backend, model, api_key, 60_000, settings.top_k, True)
    else:
        installed = ollama_models(settings.ollama_url)
        if installed:
            default = settings.ollama_model if settings.ollama_model in installed else installed[0]
            model = st.sidebar.selectbox("Model", installed, index=installed.index(default), key="ollama_model",
                                         help="Bigger models give better answers but are slower on CPU.")
        else:
            model = settings.ollama_model
            st.sidebar.warning("Ollama isn't running. Open the Ollama app, then reload this page.")
        # Local models on CPU slow down a lot with long prompts, so send less text.
        cfg = ModelConfig(backend, model, "", 9_000, min(settings.top_k, 5), True)
    cfg.rerank = st.sidebar.toggle(
        "Re-rank search results", value=settings.rerank, key="rerank",
        help="A second model re-scores passages for relevance. Sharper answers, about a second slower.",
    )
    st.session_state["model_cfg"] = cfg


def model_cfg() -> ModelConfig:
    return st.session_state["model_cfg"]


# ---- ingestion ------------------------------------------------------------------

def add_pdf(data: bytes, filename: str, meta: metadata.PaperMeta | None = None) -> tuple[str, str, int | None]:
    """Index a PDF. Returns (status, message, doc_id) with status in added/duplicate/error."""
    import research
    return research.index_pdf(get_library(), get_embedder(), data, filename, meta)


def has_pdf(doc: Document) -> bool:
    return pdf_path(doc).exists()


def upload_widget(key: str) -> None:
    for status, msg in st.session_state.pop(f"up_msgs_{key}", []):
        {"added": st.success, "duplicate": st.info}.get(status, st.error)(msg)
    # Bumping the counter gives the uploader a fresh key, which clears it after indexing.
    n = st.session_state.setdefault(f"up_n_{key}", 0)
    files = st.file_uploader("Drop PDFs here", type="pdf", accept_multiple_files=True, key=f"up_{key}_{n}",
                             label_visibility="collapsed")
    if st.button("Add to library", disabled=not files, type="primary", key=f"add_{key}", width="stretch"):
        bar = st.progress(0.0)
        results = []
        for i, f in enumerate(files):
            bar.progress(i / len(files), text=f"Indexing {f.name}… (also looks up authors, year, DOI)")
            status, msg, _ = add_pdf(f.getvalue(), f.name)
            results.append((status, msg))
        st.session_state[f"up_msgs_{key}"] = results
        st.session_state[f"up_n_{key}"] = n + 1
        st.rerun()


# ---- answering ------------------------------------------------------------------

def retrieve(llm: LLM, history: list[dict], question: str, scope: set[int] | None) -> tuple[str, list[Hit]]:
    cfg = model_cfg()
    query = condense_question(llm, history, question)
    reranker = get_reranker() if cfg.rerank else None
    pool = RERANK_POOL if reranker else cfg.top_k
    index = get_index(get_library().revision())
    hits = index.search(get_embedder().embed_query(query), query, pool, scope or None)
    if reranker:
        hits = rerank(reranker, query, hits, cfg.top_k)
    return query, hits


def cite_badges(text: str) -> str:
    """Show [3] citations as small badges."""
    return re.sub(r"\[(\d{1,2})\]", r":violet-badge[\1]", text)


def unwrap_fence(text: str) -> str:
    """Small models sometimes wrap a whole Markdown answer in ```markdown ... ```, which renders as code."""
    m = re.fullmatch(r"\s*```(?:markdown|md)?\s*\n(.*?)\n?```\s*", text, re.S)
    return m.group(1) if m else text


def cited(text: str) -> set[int]:
    return {int(n) for n in re.findall(r"\[(\d+)\]", text)}


def open_in_reader(doc_id: int, page: int) -> None:
    st.session_state["reader_doc"] = doc_id
    st.session_state["reader_page"] = page
    st.switch_page(READER_PAGE)


def render_sources(sources: list[dict], answer: str, key: str, reader_link: bool = True) -> None:
    used = cited(answer)
    with st.expander(f"📚 Sources ({len(sources)})"):
        for i, s in enumerate(sources, 1):
            with st.container(border=True):
                mark = "✅ " if i in used else ""
                st.markdown(f"{mark}**[{i}] {s['title']}** · page {s['page']}")
                st.caption(s["text"][:500] + ("…" if len(s["text"]) > 500 else ""))
                c1, c2, _ = st.columns([1.3, 1.5, 3])
                show = c1.toggle("Preview page", key=f"pv_{key}_{i}")
                if reader_link and s.get("doc_id") and c2.button("📖 Open in reader", key=f"or_{key}_{i}"):
                    open_in_reader(s["doc_id"], s["page"])
                if show:
                    img = page_image(s.get("sha", ""), s["page"])
                    if img:
                        st.image(img)
                    else:
                        st.caption("Original PDF not available.")


def notes_button(question: str, answer: str, sources: list[dict], key: str) -> None:
    if st.button("📝 Save to notes", key=f"note_{key}", type="tertiary"):
        doc_ids = {s.get("doc_id") for s in sources}
        refs = "\n".join(f"{i}. {s['title']}, p. {s['page']}" for i, s in enumerate(sources, 1))
        body = answer + (f"\n\n**Sources**\n{refs}" if refs else "")
        get_library().add_note(question[:120], body, doc_ids.pop() if len(doc_ids) == 1 else None, sources)
        st.toast("Saved to Notes", icon="📝")


def _sources_from_hits(hits: list[Hit]) -> list[dict]:
    by_id = {d.id: d for d in docs()}
    return [
        {"doc_id": h.doc_id, "title": label(by_id[h.doc_id]) if h.doc_id in by_id else "Unknown",
         "page": h.page, "text": h.text, "sha": by_id[h.doc_id].sha256 if h.doc_id in by_id else ""}
        for h in hits
    ]


def error_hint(cfg: ModelConfig, msg: str) -> str:
    if cfg.backend.startswith("Ollama"):
        return f"Is the Ollama app running, and is the model downloaded? Try: `ollama pull {cfg.model}`"
    if "429" in msg or "RESOURCE_EXHAUSTED" in msg:
        return "Free-tier quota reached. Wait a minute, pick gemini-2.5-flash-lite, or switch to Ollama (no limits)."
    if "API key" in msg or "API_KEY" in msg:
        return "Your Gemini API key looks invalid. Get a free one at https://aistudio.google.com/apikey"
    return "Check the model name and your connection."


@dataclass
class Task:
    """A one-click request with its own prompt and fixed sources (no search)."""
    question: str  # shown in the chat
    prompt: str    # sent to the model
    hits: list[Hit]


def _with_spinner(tokens, cfg: ModelConfig):
    """Show a spinner until the first token arrives (local models read the sources before writing)."""
    hint = " Local models on CPU can take a minute or two." if cfg.backend.startswith("Ollama") else ""
    with st.spinner(f"Reading the sources and writing an answer…{hint}"):
        first = next(tokens, "")
    yield first
    yield from tokens


def respond(chat_id: int, question: str, *, scope: set[int] | None = None, task: Task | None = None,
            key: str = "chat") -> None:
    library, cfg = get_library(), model_cfg()
    if cfg.backend == "Gemini" and not cfg.api_key:
        st.error("Add a Gemini API key in the sidebar, or switch to Ollama.")
        return
    history = [{"role": m["role"], "content": m["content"]} for m in library.messages(chat_id)]
    labels = {d.id: label(d) for d in docs()}
    llm = cfg.llm()

    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            if task:
                hits = task.hits
                messages = build_messages([], task.prompt, hits, labels)
            else:
                with st.status("Searching your papers…", expanded=False) as status:
                    query, hits = retrieve(llm, history, question, scope)
                    status.update(label=f"Found {len(hits)} relevant passages" +
                                  (f" for “{query}”" if query != question else ""), state="complete")
                messages = build_messages(history, question, hits, labels)
            if hits:
                answer = st.write_stream(_with_spinner(llm.stream(SYSTEM_PROMPT, messages), cfg))
            else:
                answer = "I couldn't find any relevant passages in the selected papers."
                st.markdown(answer)
        except Exception as e:
            st.error(f"Request failed: {str(e)[:400]}\n\n{error_hint(cfg, str(e))}")
            return

    sources = _sources_from_hits(hits)
    answer = unwrap_fence(answer)
    library.add_turn(chat_id, question, answer, sources)
    if len(history) == 0 and not task:
        library.rename_chat(chat_id, question[:60])
    st.rerun()


def chat_history(chat_id: int, key: str, reader_link: bool = True) -> None:
    msgs = get_library().messages(chat_id)
    for n, m in enumerate(msgs):
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                st.markdown(cite_badges(m["content"]))
                if m["sources"]:
                    render_sources(m["sources"], m["content"], key=f"{key}_{m['id']}", reader_link=reader_link)
                question = msgs[n - 1]["content"] if n else ""
                import exporters
                import export_ui
                c1, c2, _ = st.columns([1.2, 1.2, 4])
                with c1:
                    notes_button(question, m["content"], m["sources"], key=f"{key}_{m['id']}")
                with c2:
                    export_ui.document_export(
                        f"ans{m['id']}", re.sub(r"\*\*", "", question)[:80] or "Answer",
                        lambda m=m: [exporters.Section("", m["content"], m["sources"])], label="⬇️ Export")
            else:
                st.markdown(m["content"])


def setup_page() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def job_progress(kind_prefix: str, ref_id: int, key: str) -> bool:
    """Live progress for a background job; reruns the page when it finishes. Returns True while running."""
    import jobs
    library = get_library()
    running = jobs.active(library, kind_prefix, ref_id) is not None

    @st.fragment(run_every=2 if running else None)
    def panel():
        job = library.latest_job(kind_prefix, ref_id)
        if job and job["status"] == "running":
            c1, c2 = st.columns([6, 1], vertical_alignment="center")
            c1.progress(job["progress"], text=f"⏳ {job['message']}")
            if c2.button("Stop", key=f"stop_{key}"):
                jobs.cancel(job["id"])
                st.toast("Stopping after the current step…")
        elif running:
            st.rerun(scope="app")  # finished while we were watching: refresh everything
        elif job and job["status"] in ("failed", "interrupted"):
            st.warning(f"Last run {job['status']}: {job['message']}")

    panel()
    return running

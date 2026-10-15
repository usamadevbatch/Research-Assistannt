"""Reusable export popovers: documents (chat, notes, review), tables and reference lists."""
import re

import streamlit as st

import core
import exporters as ex

EXT = {"md": "md", "docx": "docx", "pdf": "pdf", "tex": "tex", "html": "html", "txt": "txt"}


def _slug(s: str) -> str:
    return re.sub(r"[^\w\-]+", "_", s).strip("_")[:50] or "export"


def _ready_button(key: str, label: str, make, filename: str, mime: str) -> None:
    """Build the file only when asked (PDF/Word can be slow), then offer the download."""
    if st.button(label, key=f"prep_{key}", width="stretch"):
        try:
            st.session_state[f"file_{key}"] = (make(), filename, mime)
        except Exception as e:
            st.error(f"Export failed: {e}")
    if f"file_{key}" in st.session_state:
        data, fname, mm = st.session_state[f"file_{key}"]
        st.download_button(f"⬇️ Download {fname}", data, fname, mm, key=f"dl_{key}", width="stretch", type="primary")


def document_export(key: str, title: str, build_sections, *, subtitle: str = "", label: str = "⬇️ Export",
                    width="content", default_mode: str = "Numbered [1]") -> None:
    """build_sections() -> list[exporters.Section]."""
    with st.popover(label, width=width):
        fmt_label = st.selectbox("Format", list(ex.FORMATS), key=f"fmt_{key}")
        mode_label = st.selectbox("Citations", list(ex.CITATION_MODES), key=f"mode_{key}",
                                  index=list(ex.CITATION_MODES).index(default_mode))
        mode = ex.CITATION_MODES[mode_label]
        style = st.selectbox("Reference style", ex.STYLES, key=f"style_{key}", disabled=mode == "none",
                             help="IEEE and Vancouver are numbered; APA, Harvard, MLA and Chicago list references "
                                  "alphabetically when you choose author–year citations.")
        fmt = ex.FORMATS[fmt_label]
        if fmt == "tex":
            st.caption("Single .tex file with the bibliography embedded. Upload it to Overleaf and compile.")
        fname = f"{_slug(title)}.{EXT[fmt]}"
        docs = {d.id: d for d in core.docs()}
        sig = f"{key}_{fmt}_{mode}_{style}"
        if st.session_state.get(f"sig_{key}") != sig:  # options changed: drop the stale file
            st.session_state.pop(f"file_{key}", None)
            st.session_state[f"sig_{key}"] = sig
        _ready_button(key, "Prepare file", lambda: ex.export_document(title, build_sections(), docs, fmt, mode, style,
                                                                      subtitle), fname, ex.MIME[fmt])


TABLE_FORMATS = {
    "Excel (.xlsx)": "xlsx", "CSV (.csv)": "csv", "LaTeX table (.tex)": "tex", "Word (.docx)": "docx",
    "PDF (.pdf)": "pdf", "Markdown (.md)": "md", "HTML (.html)": "html", "JSON (.json)": "json",
}


def table_export(key: str, table_id: int) -> None:
    library = core.get_library()
    with st.popover("⬇️ Export table", width="stretch"):
        fmt_label = st.selectbox("Format", list(TABLE_FORMATS), key=f"tfmt_{key}")
        with_cites = st.toggle("Include citations & evidence", value=True, key=f"tcite_{key}",
                               help="Adds a citation column and a reference list. In Excel/JSON also adds the "
                                    "supporting quote and page for every cell.")
        style = st.selectbox("Reference style", ex.STYLES, key=f"tstyle_{key}", disabled=not with_cites)
        fmt = TABLE_FORMATS[fmt_label]
        table = library.get_table(table_id)
        sig = f"{fmt}_{with_cites}_{style}"
        if st.session_state.get(f"tsig_{key}") != sig:
            st.session_state.pop(f"file_t{key}", None)
            st.session_state[f"tsig_{key}"] = sig

        def make():
            t = library.get_table(table_id)
            return ex.export_table(t, {d.id: d for d in core.docs()}, library.cells(table_id), fmt, with_cites, style)
        _ready_button(f"t{key}", "Prepare file", make, f"{_slug(table['name'])}.{fmt}", ex.MIME[fmt])


REF_FORMATS = {
    "BibTeX (.bib) — Zotero, Mendeley, LaTeX": "bib", "RIS (.ris) — EndNote, Zotero": "ris", "CSV (.csv)": "csv",
    "Formatted list — Word (.docx)": "docx", "Formatted list — Markdown (.md)": "md",
    "Formatted list — PDF (.pdf)": "pdf", "Formatted list — plain text (.txt)": "txt",
}


def references_export(key: str, docs: list, label: str = "⬇️ Export references") -> None:
    with st.popover(label, width="stretch"):
        st.caption(f"{len(docs)} papers")
        fmt_label = st.selectbox("Format", list(REF_FORMATS), key=f"rfmt_{key}")
        fmt = REF_FORMATS[fmt_label]
        style = st.selectbox("Style", ex.STYLES, key=f"rstyle_{key}", disabled=fmt in ("bib", "ris", "csv"))
        sig = f"{fmt}_{style}_{len(docs)}"
        if st.session_state.get(f"rsig_{key}") != sig:
            st.session_state.pop(f"file_r{key}", None)
            st.session_state[f"rsig_{key}"] = sig
        _ready_button(f"r{key}", "Prepare file", lambda: ex.export_references(docs, fmt, style),
                      f"references.{fmt}", ex.MIME[fmt])


def chat_sections(messages: list[dict]) -> list:
    """Sections for a chat: each question as a heading, the answer with its sources."""
    out = []
    pending_q = None
    for m in messages:
        if m["role"] == "user":
            pending_q = re.sub(r"\*\*", "", m["content"])
        else:
            out.append(ex.Section(pending_q or "Answer", m["content"], m.get("sources") or []))
            pending_q = None
    return out


def strip_saved_sources(content: str) -> str:
    """Notes saved from chat end with a plain 'Sources' list; exports build a proper reference list instead."""
    return re.split(r"\n\n\*\*Sources\*\*\n", content)[0]

import streamlit as st

import core
import metadata
from rag import PAGE_TASKS, PAPER_TASKS, page_hits, paper_hits
from store import STATUSES

library = core.get_library()
docs = core.docs()
if not docs:
    st.title("📖 Reader")
    st.info("Your library is empty. Add papers from the Library or Discover page.")
    st.page_link("views/library.py", label="Go to Library", icon="📚")
    st.stop()

by_id = {d.id: d for d in docs}
if st.session_state.get("reader_doc") not in by_id:
    st.session_state["reader_doc"] = docs[0].id


def on_doc_change():
    st.session_state["reader_page"] = by_id[st.session_state["reader_doc"]].last_page


def turn(delta: int):
    st.session_state["reader_page"] = st.session_state.get("reader_page", 1) + delta


# ---- paper picker ---------------------------------------------------------------
st.selectbox("Paper", list(by_id), format_func=lambda i: core.label(by_id[i]), key="reader_doc",
             on_change=on_doc_change, label_visibility="collapsed")
doc = by_id[st.session_state["reader_doc"]]
page = min(max(1, st.session_state.get("reader_page", doc.last_page)), doc.num_pages)
st.session_state["reader_page"] = page
if page != doc.last_page:
    library.update_document(doc.id, last_page=page)

meta_line = " · ".join(x for x in (doc.authors.replace(";", ",")[:120], str(doc.year or ""), doc.venue) if x)
if meta_line:
    st.markdown(f'<p class="ra-meta">{meta_line}</p>', unsafe_allow_html=True)

SIZES = {"Compact": [1, 1.25], "Balanced": [1.15, 1], "Large": [1.9, 1]}
size = st.session_state.get("reader_size", "Balanced")
left, right = st.columns(SIZES.get(size, SIZES["Balanced"]), gap="large")

# ---- page viewer ----------------------------------------------------------------
task: core.Task | None = None
with left:
    c1, c2, c3, c4 = st.columns([1, 1.6, 1, 1.6], vertical_alignment="bottom")
    c1.button("◀", on_click=turn, args=(-1,), disabled=page <= 1, width="stretch", key="prev")
    c2.number_input("Page", min_value=1, max_value=doc.num_pages, key="reader_page",
                    label_visibility="collapsed")
    c3.button("▶", on_click=turn, args=(1,), disabled=page >= doc.num_pages, width="stretch", key="next")
    c4.selectbox("Page size", list(SIZES), key="reader_size", label_visibility="collapsed",
                 index=list(SIZES).index(size) if size in SIZES else 1)
    st.caption(f"Page {page} of {doc.num_pages}")
    img = core.page_image(doc.sha256, page, 160)
    with st.container(height=780, border=True):
        if img:
            st.image(img)
        else:
            st.markdown(f"### {doc.title}")
            st.info("Only the abstract is available for this paper (paywalled or download blocked). "
                    "Download the PDF through your university access and attach it below for full-text answers.")
            if doc.url:
                st.link_button("🔗 Open publisher page", doc.url)
            st.markdown("**Abstract**")
            st.write(doc.abstract or "No abstract available.")
            up = st.file_uploader("Attach the PDF", type="pdf", key=f"attach_{doc.id}")
            if up:
                import research
                try:
                    st.success(research.attach_pdf(library, core.get_embedder(), doc.id, up.getvalue(), up.name))
                    st.rerun()
                except Exception as e:
                    st.error(str(e))
    chunks = library.document_chunks(doc.id)
    cols = st.columns(len(PAGE_TASKS))
    for col, name in zip(cols, PAGE_TASKS):
        if col.button(f"✨ {name}", key=f"pg_{name}", width="stretch"):
            hits = page_hits(doc.id, chunks, page)
            if hits:
                task = core.Task(f"**{name}** (page {page})", PAGE_TASKS[name].format(page=page), hits)
            else:
                st.toast("No text found on this page (it may be a figure or scanned).", icon="⚠️")

# ---- side panel -----------------------------------------------------------------
with right:
    tab_chat, tab_info, tab_notes = st.tabs(["💬 Ask this paper", "ℹ️ Details", "📝 Notes"])

    with tab_chat:
        chat_id = library.paper_chat(doc.id, f"Reader: {doc.title[:50]}")
        with st.popover("🧰 Paper tools", width="stretch"):
            cols = st.columns(2)
            for i, name in enumerate(PAPER_TASKS):
                if cols[i % 2].button(name, key=f"rt_{name}", width="stretch"):
                    hits = paper_hits(doc.id, chunks, core.model_cfg().context_budget)
                    task = core.Task(f"**{name}**", f"{PAPER_TASKS[name]}\n\nPaper: {core.label(doc)}", hits)
        box = st.container(height=640, border=False)
        with box:
            if not library.messages(chat_id):
                st.caption("Ask anything about this paper. Answers only use this paper and cite page numbers.")
            core.chat_history(chat_id, key=f"reader{doc.id}")
        question = st.chat_input(f"Ask about this paper…", key="reader_input")
        if library.messages(chat_id):
            c1, c2 = st.columns(2)
            with c1:
                import export_ui
                export_ui.document_export(f"rchat{chat_id}", f"Notes on {doc.title}",
                                          lambda: export_ui.chat_sections(library.messages(chat_id)),
                                          label="⬇️ Export chat", width="stretch")
            if c2.button("Clear this chat", type="tertiary", width="stretch"):
                library.clear_chat(chat_id)
                st.rerun()
        if task or question:
            with box:
                if task:
                    core.respond(chat_id, task.question, task=task)
                else:
                    core.respond(chat_id, question, scope={doc.id})

    with tab_info:
        status = st.segmented_control("Status", STATUSES, default=doc.status, key=f"status_{doc.id}")
        if status and status != doc.status:
            library.update_document(doc.id, status=status)
            st.rerun()
        if doc.abstract:
            st.markdown("**Abstract**")
            st.write(doc.abstract)
        st.markdown(f"**Pages:** {doc.num_pages} · **File:** {doc.filename}")
        if doc.doi:
            st.markdown(f"**DOI:** [{doc.doi}](https://doi.org/{doc.doi})")
        if doc.arxiv_id:
            st.markdown(f"**arXiv:** [{doc.arxiv_id}](https://arxiv.org/abs/{doc.arxiv_id})")
        st.markdown("**BibTeX**")
        st.code(metadata.bibtex(doc), language="bibtex")
        path = core.pdf_path(doc)
        if path.exists():
            st.download_button("⬇️ Download PDF", path.read_bytes(), doc.filename, "application/pdf")

    with tab_notes:
        with st.form(f"note_form_{doc.id}", clear_on_submit=True, border=False):
            text = st.text_area("New note", placeholder=f"Your thoughts on page {page}…", height=110)
            if st.form_submit_button("Save note", type="primary") and text.strip():
                library.add_note(f"p. {page}: {text.strip()[:60]}", text.strip(), doc.id,
                                 [{"doc_id": doc.id, "title": core.label(doc), "page": page, "text": ""}])
                st.toast("Note saved", icon="📝")
        for n in library.list_notes(doc.id):
            with st.container(border=True):
                st.markdown(f"**{n['title']}**")
                st.markdown(core.cite_badges(n["content"]))
                st.caption(core.local_time(n["updated_at"]))

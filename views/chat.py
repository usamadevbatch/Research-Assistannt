import streamlit as st

import core
import export_ui
from rag import MULTI_TASKS, PAPER_TASKS, SUGGESTED_QUESTIONS, multi_paper_hits, paper_hits

library = core.get_library()
docs = core.docs()
labels = {d.id: core.label(d) for d in docs}


def ensure_chat() -> int:
    chat_id = st.session_state.get("chat_id")
    if not chat_id or not library.chat_exists(chat_id):
        chat_id = library.create_chat("New chat")
        st.session_state["chat_id"] = chat_id
    return chat_id


# ---- sidebar: chat history ------------------------------------------------------
with st.session_state["sidebar_slot"]:
    st.markdown("#### 💬 Chats")
    if st.button("➕ New chat", width="stretch", type="primary"):
        st.session_state.pop("chat_id", None)
        st.rerun()
    current = st.session_state.get("chat_id")
    for c in library.list_chats(30):
        title = c["title"] if len(c["title"]) <= 34 else c["title"][:33] + "…"
        if st.button(title, key=f"chat_{c['id']}", width="stretch",
                     type="secondary" if c["id"] == current else "tertiary"):
            st.session_state["chat_id"] = c["id"]
            st.rerun()
    st.divider()

# ---- header ---------------------------------------------------------------------
st.markdown('<div class="ra-hero"><h1>💬 Chat with your papers</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="ra-muted">Every answer cites the paper and page it came from.</p>',
            unsafe_allow_html=True)

if not docs:
    with st.container(border=True):
        st.markdown("### 👋 Welcome! Add your first papers")
        st.write("Upload PDFs below, or find open-access papers on the **Discover** page.")
        core.upload_widget("chat_empty")
        st.page_link("views/discover.py", label="Discover papers", icon="🔎")
    st.stop()

chat_id = st.session_state.get("chat_id")
if chat_id and not library.chat_exists(chat_id):
    chat_id = None

# ---- toolbar --------------------------------------------------------------------
c1, c2, c3 = st.columns([4, 1.2, 1.2], vertical_alignment="bottom")
scope = c1.multiselect("Search in", options=list(labels), format_func=labels.get,
                       placeholder="All papers in your library", key="chat_scope")
with c2.popover("➕ Add papers", width="stretch"):
    core.upload_widget("chat")
if chat_id and library.messages(chat_id):
    with c3:
        title = next((c["title"] for c in library.list_chats() if c["id"] == chat_id), "Research chat")
        export_ui.document_export(f"chat{chat_id}", title,
                                  lambda: export_ui.chat_sections(library.messages(chat_id)),
                                  label="⬇️ Export chat", width="stretch")
        if st.button("🗑 Delete chat", width="stretch", type="tertiary"):
            library.delete_chat(chat_id)
            st.session_state.pop("chat_id", None)
            st.rerun()

task: core.Task | None = None
with st.expander("🧰 Paper tools: summarize, compare, literature review, research gaps"):
    one, many = st.tabs(["One paper", "Several papers"])
    with one:
        doc_id = st.selectbox("Paper", list(labels), format_func=labels.get, key="tool_doc")
        cols = st.columns(3)
        for i, name in enumerate(PAPER_TASKS):
            if cols[i % 3].button(name, key=f"pt_{name}", width="stretch"):
                hits = paper_hits(doc_id, library.document_chunks(doc_id), core.model_cfg().context_budget)
                task = core.Task(f"**{name}:** {labels[doc_id]}", f"{PAPER_TASKS[name]}\n\nPaper: {labels[doc_id]}", hits)
    with many:
        chosen = st.multiselect("Papers (pick 2 or more)", list(labels), format_func=labels.get, key="tool_docs")
        cols = st.columns(len(MULTI_TASKS))
        for col, name in zip(cols, MULTI_TASKS):
            if col.button(name, key=f"mt_{name}", width="stretch", disabled=len(chosen) < 2):
                budget = core.model_cfg().context_budget
                hits = multi_paper_hits({d: library.document_chunks(d) for d in chosen}, budget)
                names = "; ".join(labels[d] for d in chosen)
                task = core.Task(f"**{name}:** {len(chosen)} papers", f"{MULTI_TASKS[name]}\n\nPapers: {names}", hits)
        if len(chosen) < 2:
            st.caption("Select at least two papers to compare them.")

# ---- conversation ---------------------------------------------------------------
suggestion = None
if not chat_id or not library.messages(chat_id):
    st.markdown("##### Try asking")
    cols = st.columns(2)
    for i, q in enumerate(SUGGESTED_QUESTIONS):
        if cols[i % 2].button(q, key=f"sq_{i}", width="stretch"):
            suggestion = q
else:
    core.chat_history(chat_id, key="chat")

question = st.chat_input("Ask anything about your papers…")
if task:
    core.respond(ensure_chat(), task.question, task=task)
elif question or suggestion:
    core.respond(ensure_chat(), question or suggestion, scope=set(scope))

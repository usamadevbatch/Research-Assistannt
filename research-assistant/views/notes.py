from datetime import datetime

import streamlit as st

import core
import export_ui
import exporters

library = core.get_library()
docs = core.docs()
labels = {d.id: d.title for d in docs}

st.markdown('<div class="ra-hero"><h1>📝 Notes</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="ra-muted">Your own notes plus any answers you saved from chats.</p>',
            unsafe_allow_html=True)

with st.expander("➕ New note"):
    with st.form("new_note", clear_on_submit=True, border=False):
        title = st.text_input("Title")
        paper = st.selectbox("Paper (optional)", [None, *labels], format_func=lambda i: labels.get(i, "None"))
        body = st.text_area("Note", height=160, placeholder="Markdown supported")
        if st.form_submit_button("Save", type="primary") and (title.strip() or body.strip()):
            library.add_note(title.strip() or body.strip()[:60], body.strip(), paper)
            st.rerun()

c1, c2, c3 = st.columns([2.2, 2, 1.2], vertical_alignment="bottom")
text = c1.text_input("Search notes", placeholder="Search notes…", label_visibility="collapsed")
paper_filter = c2.selectbox("Paper", ["All papers", *labels], format_func=lambda i: labels.get(i, i),
                            label_visibility="collapsed")
notes = library.list_notes(None if paper_filter == "All papers" else paper_filter)
notes = [n for n in notes if not text or text.lower() in f"{n['title']} {n['content']}".lower()]

with c3:
    export_ui.document_export(
        "notes", f"Research notes {datetime.now():%Y-%m-%d}",
        lambda: [exporters.Section(n["title"], (f"*Paper: {labels[n['doc_id']]}*\n\n" if n["doc_id"] in labels else "")
                                   + export_ui.strip_saved_sources(n["content"]), n["sources"]) for n in notes],
        label="⬇️ Export notes", width="stretch")

if not notes:
    st.info("No notes yet. Use **📝 Save to notes** under any answer, or write one above.")

for n in notes:
    with st.container(border=True):
        paper = labels.get(n["doc_id"])
        st.markdown(f"**{n['title']}**")
        st.caption(" · ".join(x for x in (paper, core.local_time(n["updated_at"])) if x))
        editing = st.session_state.get("edit_note") == n["id"]
        if editing:
            new_title = st.text_input("Title", n["title"], key=f"t_{n['id']}")
            new_body = st.text_area("Note", n["content"], height=220, key=f"b_{n['id']}")
            c1, c2, _ = st.columns([1, 1, 4])
            if c1.button("Save", key=f"s_{n['id']}", type="primary"):
                library.update_note(n["id"], new_title, new_body)
                st.session_state.pop("edit_note")
                st.rerun()
            if c2.button("Cancel", key=f"c_{n['id']}"):
                st.session_state.pop("edit_note")
                st.rerun()
        else:
            st.markdown(core.cite_badges(n["content"]))
            c1, c2, c3, _ = st.columns([1, 1, 1.4, 3])
            if c1.button("✏️ Edit", key=f"e_{n['id']}", type="tertiary"):
                st.session_state["edit_note"] = n["id"]
                st.rerun()
            if c2.button("🗑 Delete", key=f"d_{n['id']}", type="tertiary"):
                library.delete_note(n["id"])
                st.rerun()
            src = next((s for s in n["sources"] if s.get("doc_id") in labels), None)
            if src and c3.button("📖 Open paper", key=f"o_{n['id']}", type="tertiary"):
                core.open_in_reader(src["doc_id"], src.get("page", 1))

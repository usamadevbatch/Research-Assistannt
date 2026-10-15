import streamlit as st

import core
import research
import table_ui

library = core.get_library()
docs = core.docs()
labels = {d.id: d.title for d in docs}

st.markdown('<div class="ra-hero"><h1>📊 Tables</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="ra-muted">Turn a set of papers into a table: one row per paper, one column per question, '
            'every cell backed by a quote and page number.</p>', unsafe_allow_html=True)

tables = library.list_tables()
names = {t["id"]: t["name"] + (" (research project)" if t["project_id"] else "") for t in tables}

with st.expander("➕ New table", expanded=not tables):
    if not docs:
        st.info("Add papers to your library first.")
    else:
        with st.form("new_table", border=False):
            name = st.text_input("Table name", placeholder="e.g. Methods comparison")
            chosen = st.multiselect("Papers", list(labels), format_func=labels.get, placeholder="Choose papers…")
            template = st.selectbox("Columns template", list(research.TEMPLATES))
            if st.form_submit_button("Create table", type="primary") and chosen:
                tid = library.create_table(name.strip() or "Untitled table", chosen, research.TEMPLATES[template])
                st.session_state["table_id"] = tid
                st.rerun()

if tables:
    current = st.session_state.get("table_id")
    tid = st.selectbox("Table", list(names), format_func=names.get,
                       index=list(names).index(current) if current in names else 0)
    st.session_state["table_id"] = tid
    t = library.get_table(tid)
    if not t["project_id"]:
        with st.expander("Papers in this table"):
            chosen = st.multiselect("Papers", list(labels), default=[d for d in t["doc_ids"] if d in labels],
                                    format_func=labels.get, key=f"tdocs_{tid}")
            c1, c2 = st.columns(2)
            if c1.button("Update papers") and chosen != t["doc_ids"]:
                library.update_table(tid, doc_ids=chosen)
                st.rerun()
            if c2.button("🗑 Delete table", type="tertiary"):
                library.delete_table(tid)
                st.session_state.pop("table_id", None)
                st.rerun()
    table_ui.render(tid, key=f"t{tid}")

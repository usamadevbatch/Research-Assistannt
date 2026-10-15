"""Extraction table component: one row per paper, one column per question, every cell cited."""
import io

import pandas as pd
import streamlit as st

import core
import export_ui
import jobs
import research

TYPES = ["text", "number", "yes/no"]


def start_extraction(table_id: int, only_missing: bool = True) -> None:
    library, cfg = core.get_library(), core.model_cfg()
    if cfg.backend == "Gemini" and not cfg.api_key:
        st.error("Add a Gemini API key in the sidebar, or switch to Ollama.")
        return
    llm, embedder = cfg.llm(), core.get_embedder()

    def work(report, cancelled):
        research.extract_table(library, embedder, llm, table_id, cfg.context_budget, report, cancelled, only_missing)
        return "Table complete"

    jobs.start(library, "extract", table_id, work)
    st.rerun()


def columns_editor(table: dict, key: str) -> None:
    library = core.get_library()
    c1, c2 = st.columns([3, 1], vertical_alignment="bottom")
    template = c1.selectbox("Start from a template", ["(keep current columns)", *research.TEMPLATES], key=f"tpl_{key}")
    if c2.button("Apply template", key=f"apply_{key}", width="stretch", disabled=template.startswith("(")):
        library.update_table(table["id"], columns=research.TEMPLATES[template])
        library.clear_cells(table["id"])
        st.rerun()
    df = pd.DataFrame(table["columns"] or research.DEFAULT_COLUMNS, columns=["name", "instruction", "type"])
    edited = st.data_editor(
        df, num_rows="dynamic", hide_index=True, width="stretch", key=f"cols_{key}",
        column_config={
            "name": st.column_config.TextColumn("Column", required=True, width="small"),
            "instruction": st.column_config.TextColumn("What to extract", width="large"),
            "type": st.column_config.SelectboxColumn("Type", options=TYPES, default="text", width="small"),
        },
    )
    if st.button("Save columns", key=f"savecols_{key}"):
        cols = [
            {"name": str(r["name"]).strip()[:40], "instruction": str(r["instruction"] or "").strip(),
             "type": r["type"] if r["type"] in TYPES else "text"}
            for _, r in edited.iterrows() if str(r["name"] or "").strip()
        ]
        old = {c["name"]: c for c in table["columns"]}
        new = {c["name"]: c for c in cols}
        library.update_table(table["id"], columns=cols)
        # Cells of changed columns are stale: blank them so "Fill table" re-extracts them.
        stale = [n for n in new if old.get(n) != new[n]]
        for d in table["doc_ids"]:
            for name in stale:
                library.set_cell(table["id"], d, name, "", [])
        st.toast("Columns saved. Click “Fill table” to extract the new values.")
        st.rerun()


def downloads(df: pd.DataFrame, name: str, key: str) -> None:
    c1, c2, c3 = st.columns(3)
    c1.download_button("⬇️ CSV", df.to_csv(index=False), f"{name}.csv", "text/csv", key=f"csv_{key}", width="stretch")
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="Extraction")
    c2.download_button("⬇️ Excel", buf.getvalue(), f"{name}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", key=f"xlsx_{key}",
                       width="stretch")
    c3.download_button("⬇️ Markdown", df.to_markdown(index=False), f"{name}.md", "text/markdown",
                       key=f"md_{key}", width="stretch")


def render(table_id: int, key: str) -> None:
    library = core.get_library()
    table = library.get_table(table_id)
    if not table:
        st.info("Table not found.")
        return
    cells = library.cells(table_id)
    filled = sum(1 for v in cells.values() if v["value"])
    total = len(table["doc_ids"]) * len(table["columns"])

    with st.expander(f"⚙️ Columns ({len(table['columns'])})", expanded=filled == 0):
        columns_editor(table, key)

    running = core.job_progress("extract", table_id, key)
    c1, c2, c3 = st.columns([1.3, 1.3, 3], vertical_alignment="center")
    if c1.button("▶️ Fill table", type="primary", key=f"fill_{key}", disabled=running or not table["doc_ids"],
                 width="stretch", help="Extract every empty cell"):
        start_extraction(table_id, only_missing=True)
    if c2.button("🔄 Re-run all", key=f"rerun_{key}", disabled=running or not filled, width="stretch"):
        library.clear_cells(table_id)
        start_extraction(table_id, only_missing=False)
    c3.caption(f"{len(table['doc_ids'])} papers × {len(table['columns'])} columns · {filled}/{total} cells filled. "
               "Local models take ~2 min per paper; Gemini is much faster.")

    df = research.table_frame(library, table_id)
    if df.empty:
        st.info("No papers in this table yet.")
        return
    event = st.dataframe(
        df, hide_index=True, width="stretch", on_select="rerun", selection_mode="single-row", key=f"df_{key}",
        column_config={"Paper": st.column_config.TextColumn(width="large"),
                       "Year": st.column_config.NumberColumn(format="%d")},
        height=min(600, 38 + 35 * len(df)),
    )
    st.caption("Select a row to see the evidence behind every cell.")
    c1, c2 = st.columns([1, 2])
    with c1:
        export_ui.table_export(key, table_id)

    rows = event.selection.rows if event and event.selection else []
    if rows:
        doc_id = table["doc_ids"][rows[0]] if rows[0] < len(table["doc_ids"]) else None
        docs = {d.id: d for d in core.docs()}
        doc = docs.get(doc_id)
        if doc:
            with st.container(border=True):
                st.markdown(f"#### 🔍 Evidence: {doc.title}")
                for col in table["columns"]:
                    cell = cells.get((doc_id, col["name"]), {})
                    st.markdown(f"**{col['name']}:** {cell.get('value') or '—'}")
                    for i, s in enumerate(cell.get("sources", []), 1):
                        c1, c2 = st.columns([6, 1.2], vertical_alignment="center")
                        c1.caption(f"p. {s['page']}: “{s['text'][:350]}…”")
                        if core.has_pdf(doc) and c2.button("📖 Open", key=f"ev_{key}_{col['name']}_{i}"):
                            core.open_in_reader(doc.id, s["page"])

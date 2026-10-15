import pandas as pd
import streamlit as st

import core
import importers
import jobs
import research
import sources
from metadata import PaperMeta
import table_ui
import export_ui
import exporters as ex

library = core.get_library()

DECISION_ICON = {"include": "✅ Include", "maybe": "🤔 Maybe", "exclude": "❌ Exclude", "pending": "⏳ Not screened"}
ICON_DECISION = {v: k for k, v in DECISION_ICON.items()}


def llm_or_stop():
    cfg = core.model_cfg()
    if cfg.backend == "Gemini" and not cfg.api_key:
        st.error("Add a Gemini API key in the sidebar, or switch to Ollama.")
        st.stop()
    return cfg, cfg.llm()


def project_table_id(project_id: int, plan: dict, included_docs: list[int]) -> int:
    tables = library.list_tables(project_id)
    if tables:
        tid = tables[0]["id"]
        library.update_table(tid, doc_ids=included_docs)
        return tid
    return library.create_table("Extraction table", included_docs, plan.get("extraction_columns") or research.DEFAULT_COLUMNS,
                                project_id)


# ---- background steps ----------------------------------------------------------------------

def start_pipeline(project_id: int, steps: list[str]) -> None:
    """Run one or more steps (plan, search, screen, fulltext, table, report) as a background job."""
    cfg, llm = llm_or_stop()
    embedder = core.get_embedder()
    reranker = core.get_reranker() if cfg.rerank else None

    def work(report, cancelled):
        n = len(steps)
        msg = "Done"
        for i, step in enumerate(steps):
            if cancelled():
                break

            def sub(p, m, i=i, step=step):
                report(None if p is None else (i + p) / n, f"[{i + 1}/{n}] {m}")

            project = library.get_project(project_id)
            plan, opts = project["plan"], project["options"]
            if step == "plan":
                sub(0.1, "Designing search strategy, keywords and criteria…")
                plan = research.make_plan(llm, project["idea"], opts.get("notes", ""))
                library.update_project(project_id, plan=plan, title=plan["title"])
                msg = "Plan ready"
            elif step == "search":
                stats = research.run_search(library, project_id, plan, opts, sub)
                research.rank_candidates(library, embedder, reranker, project_id, project["idea"], plan, sub)
                library.update_project(project_id, options={**opts, "search_stats": stats})
                msg = f"Found {stats['_unique']} unique papers"
            elif step == "screen":
                done = research.screen(library, llm, project_id, project["idea"], plan,
                                       int(opts.get("screen_n", 20)), sub, cancelled)
                msg = f"Screened {done} papers"
            elif step == "fulltext":
                c = research.get_fulltexts(library, embedder, project_id, sub, cancelled)
                msg = f"{c['pdf']} PDFs, {c['abstract']} abstract only"
            elif step == "table":
                included = [c["doc_id"] for c in library.candidates(project_id) if c["decision"] == "include" and c["doc_id"]]
                tid = project_table_id(project_id, plan, included)
                research.extract_table(library, embedder, llm, tid, cfg.context_budget, sub, cancelled)
                msg = "Table complete"
            elif step == "report":
                sub(0.1, "Writing the literature review…")
                tables = library.list_tables(project_id)
                text, refs = research.write_report(library, llm, project_id, tables[0]["id"] if tables else None,
                                                   cfg.context_budget)
                library.update_project(project_id, report=text,
                                       options={**library.get_project(project_id)["options"], "report_refs": refs,
                                                "report_model": f"{cfg.backend}: {cfg.model}"})
                msg = "Report ready"
            elif step == "snowball":
                r = research.snowball_included(library, embedder, reranker, project_id, project["idea"], plan,
                                               sub, cancelled)
                msg = f"Snowballing added {r['backward']} references and {r['forward']} citing papers"
        return msg

    jobs.start(library, "project", project_id, work)
    st.rerun()


# ---- sidebar: projects ------------------------------------------------------------------------
with st.session_state["sidebar_slot"]:
    st.markdown("#### 🔬 Research projects")
    if st.button("➕ New project", width="stretch", type="primary"):
        st.session_state.pop("project_id", None)
        st.query_params.clear()
        st.rerun()
    for p in library.list_projects()[:30]:
        title = p["title"] if len(p["title"]) <= 34 else p["title"][:33] + "…"
        if st.button(title, key=f"proj_{p['id']}", width="stretch",
                     type="secondary" if p["id"] == st.session_state.get("project_id") else "tertiary"):
            st.session_state["project_id"] = p["id"]
            st.rerun()
    st.divider()

if "project" in st.query_params:  # bookmarkable project links
    try:
        st.session_state["project_id"] = int(st.query_params["project"])
    except ValueError:
        pass
project_id = st.session_state.get("project_id")
if project_id:
    st.query_params["project"] = str(project_id)
elif "project" in st.query_params:
    del st.query_params["project"]
project = library.get_project(project_id) if project_id else None

# ---- new project ----------------------------------------------------------------------------------
if not project:
    st.markdown('<div class="ra-hero"><h1>🔬 Research agent</h1></div>', unsafe_allow_html=True)
    st.markdown('<p class="ra-muted">Describe your idea. The agent designs search strings, searches IEEE, ACM, '
                'ScienceDirect, Springer, arXiv, PubMed and more, screens abstracts, fetches open-access PDFs, '
                'builds a cited extraction table, and drafts a literature review.</p>', unsafe_allow_html=True)
    available = [s for s in sources.SOURCES if sources.available(s)]
    default = ["openalex", "pub:IEEE", "pub:ACM Digital Library", "pub:Elsevier / ScienceDirect", "arxiv"]
    with st.form("new_project"):
        idea = st.text_area("Your research idea or question", height=120,
                            placeholder="e.g. Using federated learning for intrusion detection in IoT networks: "
                                        "which models, datasets and privacy techniques work best?")
        notes = st.text_input("Extra requirements (optional)",
                              placeholder="e.g. only empirical studies, English, focus on edge devices")
        c1, c2, c3 = st.columns(3)
        year_from = c1.number_input("From year", 1950, 2100, value=2019)
        year_to = c2.number_input("To year", 1950, 2100, value=pd.Timestamp.now().year)
        per_source = c3.slider("Results per source", 10, 100, 30, step=10)
        chosen = st.multiselect(
            "Search in", [s.key for s in available], default=[k for k in default if k in {s.key for s in available}],
            format_func=lambda k: sources.BY_KEY[k].label.split(" (")[0],
            help="Publishers are searched through OpenAlex (no key needed). Direct APIs need a free key in Settings.",
        )
        st.caption("Scopus, IEEE Xplore API, ScienceDirect API, Springer and CORE appear here after you add a free "
                   "key on the Settings page.")
        c1, c2 = st.columns(2)
        screen_n = c1.slider("Papers for the AI to screen", 5, 60, 20, step=5,
                             help="The most relevant papers (by a local ranking model) are screened by the AI.")
        auto = c2.toggle("Run everything automatically", value=True,
                         help="Plan → search → screen → fetch PDFs → table → report, without stopping. "
                              "Turn off to review the plan first.")
        go = st.form_submit_button("🚀 Start research", type="primary")
    if go:
        if len(idea.strip()) < 10:
            st.error("Describe your idea in a sentence or two.")
            st.stop()
        if not chosen:
            st.error("Pick at least one source.")
            st.stop()
        pid = library.create_project(idea.strip()[:60], idea.strip(), {
            "notes": notes, "year_from": int(year_from), "year_to": int(year_to), "per_source": per_source,
            "sources": chosen, "screen_n": screen_n,
        })
        st.session_state["project_id"] = pid
        start_pipeline(pid, ["plan", "search", "screen", "fulltext", "table", "report"] if auto else ["plan"])
    st.stop()

# ---- project view -----------------------------------------------------------------------------------
plan, opts = project["plan"], project["options"]
cands = library.candidates(project_id)

st.markdown(f'<div class="ra-hero"><h1>🔬 {project["title"]}</h1></div>', unsafe_allow_html=True)
st.markdown(f'<p class="ra-muted">{project["idea"]}</p>', unsafe_allow_html=True)
running = core.job_progress("project", project_id, f"proj{project_id}")

included = [c for c in cands if c["decision"] == "include"]
stats = opts.get("search_stats", {})
m = st.columns(6)
for col, (val, name) in zip(m, [
    (stats.get("_total_found", 0), "records found"), (len(cands), "after duplicates"),
    (sum(c["decision"] != "pending" for c in cands), "screened"), (len(included), "included"),
    (sum(c["fulltext"] == "pdf" for c in included), "full PDFs"), (sum(c["fulltext"] == "abstract" for c in included), "abstract only"),
]):
    col.markdown(f'<div class="ra-stat">{val}</div><div class="ra-muted">{name}</div>', unsafe_allow_html=True)

tab_plan, tab_papers, tab_table, tab_report = st.tabs(["① Plan & search strings", "② Papers & screening",
                                                       "③ Extraction table", "④ Review & export"])

# ---- ① plan --------------------------------------------------------------------------------------
with tab_plan:
    if not plan:
        st.info("The AI is designing your search strategy…" if running else "No plan yet.")
        if not running and st.button("Create plan", type="primary"):
            start_pipeline(project_id, ["plan"])
    else:
        if plan.get("research_questions"):
            st.markdown("**Research questions**")
            st.markdown("\n".join(f"{i}. {q}" for i, q in enumerate(plan["research_questions"], 1)))
        st.markdown("**Search concepts** (synonyms are combined with OR, concepts with AND). Edit to refine:")
        cdf = pd.DataFrame([{"Concept": c["name"], "Synonyms (comma separated)": ", ".join(c["synonyms"])}
                            for c in plan["concepts"]])
        edited = st.data_editor(cdf, num_rows="dynamic", hide_index=True, width="stretch", key=f"concepts_{project_id}")
        c1, c2 = st.columns(2)
        inc = c1.text_area("Inclusion criteria (one per line)", "\n".join(plan.get("inclusion_criteria", [])), height=130)
        exc = c2.text_area("Exclusion criteria (one per line)", "\n".join(plan.get("exclusion_criteria", [])), height=130)
        new_concepts = [
            {"name": str(r["Concept"]).strip(),
             "synonyms": [s.strip() for s in str(r["Synonyms (comma separated)"] or "").split(",") if s.strip()]}
            for _, r in edited.iterrows() if str(r["Concept"] or "").strip()
        ]

        def save_plan():
            library.update_project(project_id, plan={
                **plan, "concepts": new_concepts,
                "inclusion_criteria": [x.strip() for x in inc.splitlines() if x.strip()],
                "exclusion_criteria": [x.strip() for x in exc.splitlines() if x.strip()],
            })

        b1, b2, b3 = st.columns([1, 1.6, 2.4])
        if b1.button("💾 Save plan", disabled=running):
            save_plan()
            st.toast("Plan saved")
        if b2.button("🔎 Search with this plan", type="primary", disabled=running):
            save_plan()
            start_pipeline(project_id, ["search"])
        if b3.button("🚀 Run all remaining steps", disabled=running,
                     help="Search → screen → PDFs → table → literature review"):
            save_plan()
            start_pipeline(project_id, ["search", "screen", "fulltext", "table", "report"])

        st.download_button("⬇️ Search protocol (Markdown)", research.protocol_markdown(library, project_id),
                           f"search_protocol_{project_id}.md", "text/markdown",
                           help="Research questions, concepts, criteria, exact search strings per database, dates "
                                "and counts: everything needed to report and reproduce the search.")
        st.markdown("#### Search strings for each database")
        st.caption("Copy these into IEEE Xplore, Scopus, ScienceDirect, Web of Science or PubMed through your "
                   "university access, then import the exported results below.")
        for name, q in sources.database_strings(new_concepts, opts.get("year_from"), opts.get("year_to")).items():
            st.markdown(f"**{name}**")
            st.code(q, language=None, wrap_lines=True)
        if stats:
            st.markdown("#### Last search")
            for name, v in stats.items():
                if not name.startswith("_"):
                    st.markdown(f"- **{name}**: {v.get('found', 0)} results" + (f" · ⚠️ {v['error']}" if v.get("error") else ""))

        st.markdown("#### Import results exported from a database")
        st.caption("Scopus, IEEE Xplore, ACM DL, ScienceDirect, Web of Science, Zotero... (RIS, BibTeX or CSV)")
        files = st.file_uploader("Export files", type=["ris", "bib", "csv", "txt"], accept_multiple_files=True,
                                 key=f"imp_{project_id}", label_visibility="collapsed")
        if files and st.button("Import into this project", disabled=running):
            rows, n = [], 0
            for f in files:
                try:
                    papers = importers.parse_file(f.name, f.getvalue())
                except Exception as e:
                    st.error(f"{f.name}: {e}")
                    continue
                n += len(papers)
                rows += [(sources.dedupe_key(p), p.__dict__, f"Import: {f.name}") for p in papers]
            library.upsert_candidates(project_id, rows)
            cfg = core.model_cfg()
            research.rank_candidates(library, core.get_embedder(), core.get_reranker() if cfg.rerank else None,
                                     project_id, project["idea"], plan, lambda p, m: None)
            st.success(f"Imported {n} records.")
            st.rerun()

# ---- ② papers ----------------------------------------------------------------------------------
with tab_papers:
    if not cands:
        st.info("No papers yet. Run the search from the Plan tab.")
    else:
        c1, c2, c3, c4 = st.columns([1.2, 1.6, 1.8, 2.2])
        n_screen = c1.number_input("Screen next", 5, 100, int(opts.get("screen_n", 20)), step=5,
                                   label_visibility="collapsed")
        if c2.button(f"🤖 AI-screen next {n_screen}", disabled=running, width="stretch"):
            library.update_project(project_id, options={**opts, "screen_n": int(n_screen)})
            start_pipeline(project_id, ["screen"])
        if c3.button(f"📥 Get PDFs for included ({len(included)})", disabled=running or not included, width="stretch"):
            start_pipeline(project_id, ["fulltext"])
        show = c4.segmented_control("Show", ["All", "Included", "Maybe", "Excluded", "Not screened"],
                                    default="All", key=f"show_{project_id}", label_visibility="collapsed") or "All"
        c1, c2, c3 = st.columns([1.6, 1.6, 3])
        if c1.button("🔁 Snowball citations", disabled=running or not included, width="stretch",
                     help="Backward (references) and forward (cited-by) citation searching from included papers. "
                          "New papers are added as 'not screened' and ranked."):
            start_pipeline(project_id, ["snowball"])
        c2.download_button("⬇️ Screening log (CSV)", research.screening_log_csv(library, project_id).encode("utf-8-sig"),
                           f"screening_log_{project_id}.csv", "text/csv", width="stretch",
                           help="Every record with its source, decision, who decided, and why: your audit trail.")
        retracted = [c for c in cands if c["meta"].get("retracted")]
        if retracted:
            c3.warning(f"⚠️ {len(retracted)} retracted paper(s) found. They are marked in the table; do not cite them.")
        want = {"Included": "include", "Maybe": "maybe", "Excluded": "exclude", "Not screened": "pending"}.get(show)
        rows = [c for c in cands if not want or c["decision"] == want]
        df = pd.DataFrame([{
            "id": c["id"],
            "Decision": DECISION_ICON.get(c["decision"], c["decision"]),
            "Relevance": c["relevance"],
            "Title": ("⚠️ RETRACTED · " if c["meta"].get("retracted") else "") + c["meta"]["title"],
            "Year": c["meta"].get("year"),
            "Venue": c["meta"].get("venue", ""),
            "Cited": c["meta"].get("cited_by"),
            "Found in": c["found_in"],
            "PDF": "✅ in library" if c["fulltext"] == "pdf" else (
                ("🔓 free on site" if c["meta"].get("pdf_url") else "🔒 abstract only") if c["fulltext"] == "abstract"
                else ("🔓 free" if c["meta"].get("pdf_url") else "")),
            "AI reason": c["reason"],
            "Decided by": c.get("decided_by", ""),
            "Link": c["meta"].get("url", ""),
        } for c in rows])
        edited = st.data_editor(
            df, hide_index=True, width="stretch", key=f"cands_{project_id}_{show}", disabled=[
                "id", "Relevance", "Title", "Year", "Venue", "Cited", "Found in", "PDF", "AI reason", "Decided by", "Link"],
            column_config={
                "id": None,
                "Decision": st.column_config.SelectboxColumn(options=list(DECISION_ICON.values()), width="small"),
                "Relevance": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.0f", width="small"),
                "Title": st.column_config.TextColumn(width="large"),
                "Year": st.column_config.NumberColumn(format="%d", width="small"),
                "Cited": st.column_config.NumberColumn(width="small"),
                "Link": st.column_config.LinkColumn(display_text="Open", width="small"),
            },
            height=min(700, 38 + 35 * max(1, len(df))),
        )
        changes = {int(r["id"]): ICON_DECISION.get(r["Decision"], "pending") for _, r in edited.iterrows()}
        original = {c["id"]: c["decision"] for c in rows}
        changed = {k: v for k, v in changes.items() if original.get(k) != v}
        if changed:
            if st.button(f"💾 Save {len(changed)} decision change(s)", type="primary"):
                for cid, d in changed.items():
                    library.update_candidate(cid, decision=d, reason="Set manually", decided_by="You")
                st.rerun()

        # Papers we couldn't fetch: offer the publisher link + attach PDF
        missing = [c for c in included if c["fulltext"] == "abstract" and c["doc_id"]]
        if missing:
            with st.expander(f"📄 {len(missing)} included papers have only an abstract — attach their PDFs"):
                st.caption("These are paywalled or block automatic downloads. Open each link (your university login "
                           "works there), download the PDF, and attach it here.")
                for c in missing:
                    c1, c2, c3 = st.columns([4, 1, 2.2], vertical_alignment="center")
                    free = c["meta"].get("pdf_url")
                    c1.markdown(f"**{c['meta']['title'][:110]}**  \n"
                                + ("🔓 Free on the publisher's site (it blocks automatic downloads)" if free
                                   else "🔒 Paywalled: use your university access"))
                    link = free or c["meta"].get("url")
                    if link:
                        c2.link_button("PDF" if free else "Open", link, width="stretch")
                    up = c3.file_uploader("PDF", type="pdf", key=f"att_{c['id']}", label_visibility="collapsed")
                    if up:
                        try:
                            st.success(research.attach_pdf(library, core.get_embedder(), c["doc_id"], up.getvalue(), up.name))
                            library.update_candidate(c["id"], fulltext="pdf")
                        except Exception as e:
                            st.error(str(e))

        sel = st.selectbox("Read an abstract", [c["id"] for c in rows],
                           format_func=lambda i: next(c["meta"]["title"] for c in rows if c["id"] == i),
                           index=None, placeholder="Pick a paper to read its abstract…")
        if sel:
            c = next(c for c in rows if c["id"] == sel)
            meta = c["meta"]
            with st.container(border=True):
                st.markdown(f"**{meta['title']}**")
                st.caption(" · ".join(x for x in (", ".join(meta.get("authors", [])[:6]), str(meta.get("year") or ""),
                                                  meta.get("venue", "")) if x))
                st.write(meta.get("abstract") or "_No abstract available from the sources searched._")
                if c["reason"]:
                    st.caption(f"AI screening: {DECISION_ICON.get(c['decision'])} — {c['reason']}")

# ---- ③ table ------------------------------------------------------------------------------------
with tab_table:
    docs_included = [c["doc_id"] for c in included if c["doc_id"]]
    if not docs_included:
        st.info("Include papers and fetch their full texts first (Papers tab). The table fills from the included papers.")
    else:
        tid = project_table_id(project_id, plan, docs_included)
        table_ui.render(tid, key=f"pt{project_id}")

# ---- ④ report & export ------------------------------------------------------------------------
with tab_report:
    refs = opts.get("report_refs", [])
    docs_by_id = {d.id: d for d in core.docs()}

    def review_sections():
        return [ex.Section("", project["report"], [{"doc_id": d, "page": ""} for d in refs])]

    c1, c2, c3 = st.columns([1.6, 1.4, 1.4])
    if c1.button("✍️ Write literature review", type="primary", disabled=running or not included, width="stretch"):
        start_pipeline(project_id, ["report"])
    if project["report"]:
        with c2:
            export_ui.document_export(f"rev{project_id}", f"Literature review: {project['title']}", review_sections,
                                      label="⬇️ Export review", width="stretch")
        if c3.button("📝 Save to notes", width="stretch"):
            library.add_note(f"Literature review: {project['title']}", project["report"], None,
                             [{"doc_id": d, "page": "", "title": core.label(docs_by_id[d])} for d in refs if d in docs_by_id])
            st.toast("Saved to Notes", icon="📝")
        if opts.get("report_model"):
            st.caption(f"Written by {opts['report_model']}. Verify claims against the extraction table evidence.")
        with st.container(border=True):
            st.markdown(core.cite_badges(project["report"]))
            if refs:
                st.markdown("##### References")
                st.markdown("\n\n".join(f"[{i}] " + ex.format_reference(docs_by_id[d], "APA 7")
                                          for i, d in enumerate(refs, 1) if d in docs_by_id))
    else:
        st.info("The review is written from the included papers and the extraction table, with numbered citations "
                "and a reference list.")

    st.markdown("#### 📋 Reporting (PRISMA 2020)")
    counts = research.prisma_counts(library, project_id)
    svg = research.prisma_svg(counts)
    with st.expander("PRISMA flow diagram", expanded=bool(cands)):
        st.markdown(f'<div style="overflow-x:auto">{svg}</div>', unsafe_allow_html=True)
        st.download_button("⬇️ Diagram (SVG)", svg, f"prisma_{project_id}.svg", "image/svg+xml")
    with st.expander("Methods section (generated from your search log)"):
        methods = research.methods_section(library, project_id,
                                           screener=f"a language model ({core.model_cfg().model})")
        st.markdown(methods)
        export_ui.document_export(f"meth{project_id}", f"Methods: {project['title']}",
                                  lambda: [ex.Section("", methods)], label="⬇️ Export methods",
                                  default_mode="No citations (clean text)")

    st.markdown("#### 📦 Export everything")
    st.caption("One ZIP with the search protocol, PRISMA diagram, screening log, included references (BibTeX, RIS), "
               "extraction table (Excel, CSV, LaTeX), methods section and literature review (Word, LaTeX, PDF, Markdown).")
    bstyle = st.selectbox("Reference style for the bundle", ex.STYLES, key=f"bstyle_{project_id}")

    def bundle() -> bytes:
        inc_docs = [docs_by_id[c["doc_id"]] for c in included if c["doc_id"] in docs_by_id]
        files = {
            "01_search_protocol.md": research.protocol_markdown(library, project_id),
            "02_prisma_flow.svg": svg,
            "03_screening_log.csv": research.screening_log_csv(library, project_id).encode("utf-8-sig"),
            "04_included_references.bib": ex.bibtex(inc_docs),
            "04_included_references.ris": ex.ris(inc_docs),
            "04_included_references.docx": ex.export_references(inc_docs, "docx", bstyle),
        }
        meth = [ex.Section("", research.methods_section(library, project_id))]
        files["05_methods.docx"] = ex.export_document("Methods", meth, docs_by_id, "docx", "none", bstyle)
        files["05_methods.tex"] = ex.export_document("Methods", meth, docs_by_id, "tex", "none", bstyle)
        tables = library.list_tables(project_id)
        if tables:
            t = library.get_table(tables[0]["id"])
            cells = library.cells(t["id"])
            for fmt in ("xlsx", "csv", "tex", "docx"):
                files[f"06_extraction_table.{fmt}"] = ex.export_table(t, docs_by_id, cells, fmt, True, bstyle)
        if project["report"]:
            mode = "numbered" if bstyle in ex.NUMERIC_STYLES else "author-year"
            for fmt in ("md", "docx", "tex", "pdf"):
                files[f"07_literature_review.{fmt}"] = ex.export_document(
                    f"Literature review: {project['title']}", review_sections(), docs_by_id, fmt, mode, bstyle)
        return ex.zip_bytes(files)

    export_ui._ready_button(f"zip{project_id}_{bstyle}", "📦 Prepare project bundle (.zip)", bundle,
                            f"{export_ui._slug(project['title'])}_review_package.zip", "application/zip")

st.divider()
if st.button("🗑 Delete project", type="tertiary"):
    library.delete_project(project_id)
    st.session_state.pop("project_id", None)
    st.query_params.clear()
    st.rerun()

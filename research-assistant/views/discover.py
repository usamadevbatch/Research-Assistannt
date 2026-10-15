import re
from datetime import date

import streamlit as st

import core
import metadata

library = core.get_library()

st.markdown('<div class="ra-hero"><h1>🔎 Discover papers</h1></div>', unsafe_allow_html=True)
st.markdown('<p class="ra-muted">Search 250M+ papers (OpenAlex) or arXiv for free, and add open-access PDFs '
            'to your library in one click.</p>', unsafe_allow_html=True)


def safe_filename(meta: metadata.PaperMeta) -> str:
    base = re.sub(r"[^\w\- ]", "", meta.title)[:80].strip() or "paper"
    return f"{base}.pdf"


def import_paper(meta: metadata.PaperMeta, key: str) -> None:
    existing = library.find_document(doi=meta.doi, arxiv_id=meta.arxiv_id)
    if existing:
        st.toast("Already in your library", icon="ℹ️")
        return
    with st.spinner(f"Downloading and indexing “{meta.title[:60]}”…"):
        try:
            data = metadata.download_pdf(meta.pdf_url)
        except Exception as e:
            st.error(f"Couldn't download the PDF: {e}. Try the publisher link and upload it manually.")
            return
        status, msg, doc_id = core.add_pdf(data, safe_filename(meta), meta)
    {"added": st.success, "duplicate": st.info}.get(status, st.error)(msg)
    if doc_id:
        st.session_state.setdefault("discover_added", set()).add(key)


def result_card(meta: metadata.PaperMeta, i: int) -> None:
    key = meta.doi or meta.arxiv_id or meta.title
    with st.container(border=True):
        st.markdown(f"**{meta.title}**")
        authors = ", ".join(meta.authors[:4]) + (" et al." if len(meta.authors) > 4 else "")
        info = " · ".join(x for x in (authors, str(meta.year or ""), meta.venue,
                                      f"cited by {meta.cited_by:,}" if meta.cited_by is not None else "") if x)
        st.caption(info)
        if meta.abstract:
            with st.expander("Abstract"):
                st.write(meta.abstract)
        c1, c2, _ = st.columns([1.4, 1, 3])
        in_lib = key in st.session_state.get("discover_added", set()) or \
            library.find_document(doi=meta.doi, arxiv_id=meta.arxiv_id)
        if in_lib:
            c1.button("✅ In library", key=f"in_{i}", disabled=True, width="stretch")
        elif meta.pdf_url:
            if c1.button("➕ Add to library", key=f"add_{i}", type="primary", width="stretch"):
                import_paper(meta, key)
                st.rerun()
        else:
            c1.button("No free PDF", key=f"nopdf_{i}", disabled=True, width="stretch",
                      help="No open-access copy found. Download it via your institution and upload it.")
        if meta.url:
            c2.link_button("🔗 Open", meta.url, width="stretch")


# ---- search ---------------------------------------------------------------------
with st.form("search"):
    query = st.text_input("Search", placeholder="e.g. protein structure prediction with transformers",
                          label_visibility="collapsed")
    c1, c2, c3, c4 = st.columns([1.3, 1.2, 1, 1.3], vertical_alignment="bottom")
    source = c1.selectbox("Source", ["OpenAlex (all fields)", "arXiv (preprints)"])
    sort = c2.selectbox("Sort by", ["relevance", "citations", "newest"])
    year_from = c3.number_input("From year", min_value=1900, max_value=date.today().year, value=None,
                                placeholder="Any")
    oa_only = c4.checkbox("Free PDF available only", value=True)
    submitted = st.form_submit_button("Search", type="primary")

if submitted and query.strip():
    with st.spinner("Searching…"):
        try:
            if source.startswith("arXiv"):
                results = metadata.search_arxiv(query, 20, "newest" if sort == "newest" else "relevance")
                if year_from:
                    results = [r for r in results if (r.year or 0) >= year_from]
            else:
                results = metadata.search_openalex(query, 20, oa_only, year_from, sort)
            st.session_state["discover_results"] = results
        except Exception as e:
            st.error(f"Search failed: {e}. Check your internet connection and try again.")

results = st.session_state.get("discover_results")
if results is not None:
    st.markdown(f"##### {len(results)} results")
    if not results:
        st.info("No results. Try broader keywords or untick “Free PDF available only”.")
    for i, meta in enumerate(results):
        result_card(meta, i)

# ---- add by identifier ----------------------------------------------------------
st.divider()
st.markdown("#### Add a specific paper")
with st.form("by_id"):
    ident = st.text_input("DOI, arXiv ID, or a direct PDF link",
                          placeholder="10.1038/s41586-021-03819-2  ·  1706.03762  ·  https://…/paper.pdf")
    go = st.form_submit_button("Add paper", type="primary")
if go and ident.strip():
    doi, arxiv_id = metadata.parse_identifier(ident)
    if doi or arxiv_id:
        with st.spinner("Looking up the paper…"):
            meta = metadata.lookup(doi, arxiv_id)
        if meta is None:
            st.error("Couldn't find that paper. Check the identifier.")
        else:
            if not meta.pdf_url and meta.doi:
                meta.pdf_url = metadata.find_pdf_url(meta.doi)
            if meta.pdf_url:
                import_paper(meta, meta.doi or meta.arxiv_id)
            else:
                st.warning(f"Found “{meta.title}”, but there's no free PDF. Download it through your "
                           "library or the publisher, then upload it on the Library page.")
    elif ident.strip().lower().startswith("http"):
        with st.spinner("Downloading…"):
            try:
                data = metadata.download_pdf(ident.strip())
                status, msg, _ = core.add_pdf(data, ident.rstrip("/").rsplit("/", 1)[-1][:80] or "paper.pdf")
                {"added": st.success, "duplicate": st.info}.get(status, st.error)(msg)
            except Exception as e:
                st.error(f"Couldn't download: {e}")
    else:
        st.error("That doesn't look like a DOI, arXiv ID, or link.")

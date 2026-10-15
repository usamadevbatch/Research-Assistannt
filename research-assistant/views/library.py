import streamlit as st

import core
import export_ui
import metadata
from store import STATUSES

library = core.get_library()
docs = core.docs()

st.markdown('<div class="ra-hero"><h1>📚 Library</h1></div>', unsafe_allow_html=True)

notes = library.list_notes()
c = st.columns(4)
for col, (num, name) in zip(c, [(len(docs), "papers"), (sum(d.num_pages for d in docs), "pages"),
                                (sum(d.status == "Read" for d in docs), "read"), (len(notes), "notes")]):
    col.markdown(f'<div class="ra-stat">{num:,}</div><div class="ra-muted">{name}</div>', unsafe_allow_html=True)

with st.expander("➕ Add papers (PDF)", expanded=not docs):
    core.upload_widget("library")
    st.page_link("views/discover.py", label="Or search for papers online", icon="🔎")

if not docs:
    st.stop()

# ---- filters --------------------------------------------------------------------
c1, c2, c3, c4 = st.columns([2.2, 1.2, 1.2, 1.2], vertical_alignment="bottom")
text = c1.text_input("Filter", placeholder="Search titles, authors, tags…", label_visibility="collapsed")
status = c2.selectbox("Status", ["All statuses", *STATUSES], label_visibility="collapsed")
all_tags = sorted({t.strip() for d in docs for t in d.tags.split(",") if t.strip()})
tag = c3.selectbox("Tag", ["All tags", *all_tags], label_visibility="collapsed")
sort = c4.selectbox("Sort", ["Recently added", "Title", "Year (newest)", "Year (oldest)"],
                    label_visibility="collapsed")

shown = [
    d for d in docs
    if (not text or text.lower() in f"{d.title} {d.authors} {d.tags} {d.venue}".lower())
    and (status == "All statuses" or d.status == status)
    and (tag == "All tags" or tag in [t.strip() for t in d.tags.split(",")])
]
if sort == "Title":
    shown.sort(key=lambda d: d.title.lower())
elif sort.startswith("Year"):
    shown.sort(key=lambda d: d.year or 0, reverse=sort.endswith("(newest)"))

c1, c2 = st.columns([3, 1.2], vertical_alignment="center")
c1.caption(f"Showing {len(shown)} of {len(docs)} papers")
with c2:
    export_ui.references_export("library", shown, label=f"⬇️ Export {len(shown)} references")

STATUS_ICON = {"To read": "⚪", "Reading": "🟡", "Read": "🟢"}

for d in shown:
    with st.container(border=True):
        top, actions = st.columns([5, 1.3], vertical_alignment="top")
        with top:
            st.markdown(f"{STATUS_ICON.get(d.status, '⚪')} **{d.title}**")
            info = " · ".join(x for x in (d.authors.replace(";", ",")[:140], str(d.year or ""), d.venue,
                                          f"{d.num_pages} pages") if x)
            st.caption(info)
            if d.tags:
                st.markdown(" ".join(f":gray-badge[{t.strip()}]" for t in d.tags.split(",") if t.strip()))
        with actions:
            if st.button("📖 Read", key=f"read_{d.id}", width="stretch", type="primary"):
                core.open_in_reader(d.id, d.last_page)
        with st.expander("Edit details · BibTeX · remove"):
            with st.form(f"edit_{d.id}", border=False):
                title = st.text_input("Title", d.title)
                authors = st.text_input("Authors (separate with ;)", d.authors)
                c1, c2, c3 = st.columns([1, 2, 2])
                year = c1.number_input("Year", min_value=0, max_value=2100, value=d.year or 0)
                venue = c2.text_input("Venue", d.venue)
                doi = c3.text_input("DOI", d.doi)
                c1, c2 = st.columns([3, 1.2])
                tags = c1.text_input("Tags (comma separated)", d.tags, placeholder="e.g. thesis, methods, to-cite")
                new_status = c2.selectbox("Status", STATUSES, index=STATUSES.index(d.status) if d.status in STATUSES else 0)
                if st.form_submit_button("Save", type="primary"):
                    library.update_document(d.id, title=title.strip() or d.title, authors=authors.strip(),
                                            year=year or None, venue=venue.strip(), doi=doi.strip(),
                                            tags=tags.strip(), status=new_status)
                    st.rerun()
            st.code(metadata.bibtex(d), language="bibtex")
            c1, c2, c3 = st.columns(3)
            if c1.button("🔄 Look up metadata online", key=f"refresh_{d.id}", width="stretch",
                         help="Finds authors, year, venue and abstract from the DOI or arXiv ID"):
                ident_doi, ident_arxiv = d.doi, d.arxiv_id
                if not (ident_doi or ident_arxiv):
                    first = " ".join(t for p, t in library.document_chunks(d.id) if p <= 2)
                    ident_doi, ident_arxiv = metadata.find_identifiers(first)
                meta = metadata.lookup(ident_doi, ident_arxiv) if (ident_doi or ident_arxiv) else None
                if meta:
                    library.update_document(d.id, **meta.as_document_fields())
                    st.rerun()
                else:
                    st.warning("No DOI or arXiv ID found in this paper. Add a DOI above and try again.")
            path = core.pdf_path(d)
            if path.exists():
                c2.download_button("⬇️ PDF", path.read_bytes(), d.filename, "application/pdf",
                                   key=f"dl_{d.id}", width="stretch")
            if c3.button("🗑 Remove paper", key=f"del_{d.id}", width="stretch"):
                st.session_state[f"confirm_{d.id}"] = True
            if st.session_state.get(f"confirm_{d.id}"):
                st.warning("Remove this paper, its reader chat and its search index? Notes are kept.")
                y, n, _ = st.columns([1, 1, 3])
                if y.button("Yes, remove", key=f"yes_{d.id}", type="primary"):
                    library.delete_document(d.id)
                    path.unlink(missing_ok=True)
                    st.session_state.pop(f"confirm_{d.id}", None)
                    st.rerun()
                if n.button("Cancel", key=f"no_{d.id}"):
                    st.session_state.pop(f"confirm_{d.id}", None)
                    st.rerun()

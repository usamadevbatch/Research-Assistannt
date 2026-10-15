"""Bibliographic metadata and paper discovery using free, keyless scholarly APIs.

- arXiv API for arXiv papers, Crossref for DOIs (metadata lookup)
- OpenAlex (250M+ works) and arXiv for searching and finding open-access PDFs
"""
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import requests

TIMEOUT = 20
HEADERS = {"User-Agent": "ResearchAssistant/1.0 (local research tool)"}
_ATOM = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}

_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"<>]+)", re.I)
_ARXIV = re.compile(r"arXiv:\s*(\d{4}\.\d{4,5})(?:v\d+)?", re.I)
_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", re.I)


@dataclass
class PaperMeta:
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    venue: str = ""
    doi: str = ""
    arxiv_id: str = ""
    abstract: str = ""
    url: str = ""
    pdf_url: str = ""
    cited_by: int | None = None
    retracted: bool = False
    openalex_id: str = ""

    def as_document_fields(self) -> dict:
        out = {
            "authors": "; ".join(self.authors), "year": self.year, "venue": self.venue,
            "doi": self.doi, "arxiv_id": self.arxiv_id, "abstract": self.abstract, "url": self.url,
        }
        if self.title:
            out["title"] = self.title
        return {k: v for k, v in out.items() if v not in ("", None)}


def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", s or "")).strip()


def _get(url: str, **params):
    r = requests.get(url, params=params or None, headers=HEADERS, timeout=TIMEOUT)
    r.raise_for_status()
    return r


# ---- identifiers --------------------------------------------------------------

def find_identifiers(text: str) -> tuple[str, str]:
    """(doi, arxiv_id) found in the first pages of a paper, if any."""
    arxiv = _ARXIV.search(text) or _ARXIV_URL.search(text)
    doi = _DOI.search(text)
    doi_str = doi.group(1).rstrip(".,;)]}") if doi else ""
    return doi_str, arxiv.group(1) if arxiv else ""


def parse_identifier(s: str) -> tuple[str, str]:
    """Accept a DOI, arXiv id, or doi.org / arxiv.org URL typed by the user."""
    s = s.strip()
    if m := _ARXIV_URL.search(s):
        return "", m.group(1)
    if m := re.fullmatch(r"(?:arxiv:)?\s*(\d{4}\.\d{4,5})(?:v\d+)?", s, re.I):
        return "", m.group(1)
    if m := _DOI.search(s):
        return m.group(1).rstrip(".,;)]}"), ""
    return "", ""


# ---- lookups --------------------------------------------------------------------

def _arxiv_entry(e) -> PaperMeta:
    aid = e.findtext("a:id", "", _ATOM).rsplit("/", 1)[-1]
    aid = re.sub(r"v\d+$", "", aid)
    published = e.findtext("a:published", "", _ATOM)
    return PaperMeta(
        title=_clean(e.findtext("a:title", "", _ATOM)),
        authors=[_clean(a.findtext("a:name", "", _ATOM)) for a in e.findall("a:author", _ATOM)],
        year=int(published[:4]) if published[:4].isdigit() else None,
        venue=_clean(e.findtext("arxiv:journal_ref", "", _ATOM)) or "arXiv",
        doi=e.findtext("arxiv:doi", "", _ATOM) or "",
        arxiv_id=aid,
        abstract=_clean(e.findtext("a:summary", "", _ATOM)),
        url=f"https://arxiv.org/abs/{aid}",
        pdf_url=f"https://arxiv.org/pdf/{aid}",
    )


def lookup_arxiv(arxiv_id: str) -> PaperMeta | None:
    try:
        root = ET.fromstring(_get("https://export.arxiv.org/api/query", id_list=arxiv_id).content)
        entry = root.find("a:entry", _ATOM)
        return _arxiv_entry(entry) if entry is not None and entry.findtext("a:title", "", _ATOM) else None
    except Exception:
        return None


def lookup_crossref(doi: str) -> PaperMeta | None:
    try:
        m = _get(f"https://api.crossref.org/works/{doi}").json()["message"]
    except Exception:
        return None
    year = None
    for key in ("published", "published-print", "published-online", "issued"):
        parts = (m.get(key) or {}).get("date-parts") or [[None]]
        if parts[0] and parts[0][0]:
            year = parts[0][0]
            break
    return PaperMeta(
        title=_clean((m.get("title") or [""])[0]),
        authors=[f"{a.get('given', '')} {a.get('family', '')}".strip() for a in m.get("author", [])],
        year=year,
        venue=_clean((m.get("container-title") or [""])[0]),
        doi=doi,
        abstract=_clean(m.get("abstract", "")),
        url=f"https://doi.org/{doi}",
    )


def lookup(doi: str = "", arxiv_id: str = "") -> PaperMeta | None:
    meta = lookup_arxiv(arxiv_id) if arxiv_id else None
    if meta is None and doi:
        meta = lookup_crossref(doi)
    if meta and doi and not meta.doi:
        meta.doi = doi
    return meta


def find_pdf_url(doi: str) -> str:
    """Best open-access PDF for a DOI, via OpenAlex."""
    try:
        w = _get(f"https://api.openalex.org/works/doi:{doi}").json()
    except Exception:
        return ""
    for loc in [w.get("best_oa_location") or {}, *(w.get("locations") or [])]:
        if loc and loc.get("pdf_url"):
            return loc["pdf_url"]
    return ""


# ---- search ---------------------------------------------------------------------

def _openalex_abstract(inv: dict | None) -> str:
    if not inv:
        return ""
    words = sorted((pos, w) for w, positions in inv.items() for pos in positions)
    return " ".join(w for _, w in words)


def search_openalex(query: str, limit: int = 15, open_access_only: bool = False,
                    year_from: int | None = None, sort: str = "relevance") -> list[PaperMeta]:
    # Plain `search` matches anywhere in the full text, which drowns results in loosely related papers.
    q = re.sub(r"[,:|]", " ", query).strip()
    filters = [f"title_and_abstract.search:{q}"]
    if open_access_only:
        filters.append("open_access.is_oa:true")
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    params = {"filter": ",".join(filters), "per_page": limit}
    if sort == "citations":
        params["sort"] = "cited_by_count:desc"
    elif sort == "newest":
        params["sort"] = "publication_date:desc"
    results = _get("https://api.openalex.org/works", **params).json().get("results", [])
    out = []
    for w in results:
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        pdf = ""
        for loc in [w.get("best_oa_location") or {}, *(w.get("locations") or [])]:
            if loc and loc.get("pdf_url"):
                pdf = loc["pdf_url"]
                break
        ids = w.get("ids") or {}
        arxiv = ""
        for loc in w.get("locations") or []:
            if m := _ARXIV_URL.search((loc or {}).get("landing_page_url") or ""):
                arxiv = m.group(1)
                break
        if arxiv and not pdf:
            pdf = f"https://arxiv.org/pdf/{arxiv}"
        out.append(PaperMeta(
            title=_clean(w.get("title") or w.get("display_name") or ""),
            authors=[a["author"]["display_name"] for a in w.get("authorships", []) if a.get("author")],
            year=w.get("publication_year"),
            venue=_clean((((w.get("primary_location") or {}).get("source")) or {}).get("display_name") or ""),
            doi=doi,
            arxiv_id=arxiv,
            abstract=_openalex_abstract(w.get("abstract_inverted_index")),
            url=f"https://doi.org/{doi}" if doi else ids.get("openalex", ""),
            pdf_url=pdf,
            cited_by=w.get("cited_by_count"),
        ))
    return out


def search_arxiv(query: str, limit: int = 15, sort: str = "relevance") -> list[PaperMeta]:
    sort_by = {"newest": "submittedDate"}.get(sort, "relevance")
    terms = " AND ".join(f"all:{t}" for t in re.findall(r"\w+", query)) or query
    root = ET.fromstring(_get(
        "https://export.arxiv.org/api/query",
        search_query=terms, max_results=limit, sortBy=sort_by, sortOrder="descending",
    ).content)
    return [_arxiv_entry(e) for e in root.findall("a:entry", _ATOM)]


def download_pdf(url: str) -> bytes:
    """Download a PDF, raising ValueError if the server returns something else (e.g. a paywall page)."""
    url = re.sub(r"arxiv\.org/abs/", "arxiv.org/pdf/", url)
    r = requests.get(url, headers={**HEADERS, "Accept": "application/pdf"}, timeout=60, allow_redirects=True)
    r.raise_for_status()
    if not r.content.startswith(b"%PDF"):
        raise ValueError("The link didn't return a PDF (it may be behind a paywall or login).")
    return r.content


# ---- citations ----------------------------------------------------------------

def bibtex(doc) -> str:
    """BibTeX entry for a store.Document."""
    authors = [a.strip() for a in (doc.authors or "").split(";") if a.strip()]
    last = re.sub(r"\W", "", authors[0].split()[-1]).lower() if authors else "anon"
    first_word = next((w.lower() for w in re.findall(r"[A-Za-z]+", doc.title) if len(w) > 3), "paper")
    key = f"{last}{doc.year or ''}{first_word}"
    is_preprint = (doc.venue or "").lower() in ("", "arxiv")
    kind = "misc" if is_preprint else "article"
    fields = {
        "title": "{" + doc.title + "}",
        "author": " and ".join(authors),
        "year": str(doc.year or ""),
        "journal" if kind == "article" else "howpublished": doc.venue if kind == "article" else
        (f"arXiv:{doc.arxiv_id}" if doc.arxiv_id else ""),
        "doi": doc.doi,
        "eprint": doc.arxiv_id,
        "archivePrefix": "arXiv" if doc.arxiv_id else "",
        "url": doc.url,
    }
    body = ",\n".join(f"  {k} = {{{v}}}" for k, v in fields.items() if v)
    return f"@{kind}{{{key},\n{body}\n}}"

"""Search connectors for scholarly databases and publishers.

Keyless: OpenAlex (incl. per-publisher search for IEEE, ACM, Elsevier/ScienceDirect, Springer, Wiley, ...),
arXiv, Europe PMC (PubMed), Semantic Scholar.
With a free API key (Settings page): IEEE Xplore, Scopus, ScienceDirect, Springer Nature, CORE.
"""
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import requests

import settings_store
from metadata import HEADERS, PaperMeta, _arxiv_entry, _clean, _openalex_abstract, _ATOM

TIMEOUT = 30

# OpenAlex publisher ids (publisher_lineage covers imprints and subsidiaries).
PUBLISHERS = {
    "IEEE": "P4310319808",
    "ACM Digital Library": "P4310319798",
    "Elsevier / ScienceDirect": "P4310320990",
    "Springer Nature": "P4310319900|P4310320108",
    "Wiley": "P4310320595",
    "Taylor & Francis": "P4310320547",
    "MDPI": "P4310310987",
}


@dataclass(frozen=True)
class Source:
    key: str
    label: str
    needs_key: str = ""  # settings_store key that must be set
    group: str = "Databases"


SOURCES = [
    Source("openalex", "OpenAlex (all publishers, 250M+ works)"),
    *[Source(f"pub:{name}", f"{name}", group="Publishers (via OpenAlex, no key)") for name in PUBLISHERS],
    Source("arxiv", "arXiv (preprints)"),
    Source("europepmc", "PubMed / Europe PMC (biomedical)"),
    Source("semantic_scholar", "Semantic Scholar"),
    Source("ieee_api", "IEEE Xplore API", "ieee_key", "Direct APIs (free key)"),
    Source("scopus", "Scopus", "elsevier_key", "Direct APIs (free key)"),
    Source("sciencedirect", "ScienceDirect API", "elsevier_key", "Direct APIs (free key)"),
    Source("springer_api", "Springer Nature API", "springer_key", "Direct APIs (free key)"),
    Source("core", "CORE (open-access full text)", "core_key", "Direct APIs (free key)"),
]
BY_KEY = {s.key: s for s in SOURCES}


def available(source: Source) -> bool:
    return not source.needs_key or bool(settings_store.get(source.needs_key))


# ---- query building -----------------------------------------------------------

def boolean_query(concepts: list[dict]) -> str:
    """(“a” OR “b”) AND (“c” OR “d”) from [{'name':..,'synonyms':[..]}]."""
    groups = []
    for c in concepts:
        terms = list(dict.fromkeys([c.get("name", ""), *c.get("synonyms", [])]))
        terms = [t.strip().replace('"', "") for t in terms if t and t.strip()]
        if terms:
            groups.append("(" + " OR ".join(f'"{t}"' if " " in t or "-" in t else t for t in terms) + ")")
    return " AND ".join(groups)


def database_strings(concepts: list[dict], year_from: int | None = None, year_to: int | None = None) -> dict[str, str]:
    """Search strings formatted for each database's advanced search (also handy to paste into their websites)."""
    b = boolean_query(concepts)
    years = ""
    if year_from or year_to:
        years = f" AND PUBYEAR > {(year_from or 1900) - 1}" + (f" AND PUBYEAR < {year_to + 1}" if year_to else "")
    arxiv_groups = []
    for c in concepts:
        terms = [t.replace('"', "") for t in [c.get("name", ""), *c.get("synonyms", [])] if t]
        arxiv_groups.append("(" + " OR ".join(f'all:"{t}"' for t in dict.fromkeys(terms)) + ")")
    return {
        "Generic Boolean (Google Scholar, OpenAlex, ACM DL, Web of Science TS=)": b,
        "Scopus (Advanced search)": f"TITLE-ABS-KEY({b}){years}",
        "IEEE Xplore (Command search)": " AND ".join(
            "(" + " OR ".join(f'"All Metadata":{t}' for t in re.findall(r'"[^"]+"|[^\s()"]+', g) if t not in ("OR",)) + ")"
            for g in re.findall(r"\(([^()]*)\)", b)
        ),
        "ScienceDirect (Title, abstract, keywords)": b,
        "Web of Science": f"TS=({b})",
        "PubMed": " AND ".join(
            "(" + " OR ".join(f"{t}[tiab]" for t in re.findall(r'"[^"]+"|[^\s()"]+', g) if t != "OR") + ")"
            for g in re.findall(r"\(([^()]*)\)", b)
        ),
        "arXiv": " AND ".join(arxiv_groups),
    }


# ---- helpers ------------------------------------------------------------------

def _get(url: str, *, params=None, headers=None, retries: int = 3):
    for attempt in range(retries + 1):
        r = requests.get(url, params=params, headers={**HEADERS, **(headers or {})}, timeout=TIMEOUT)
        if r.status_code == 429 and attempt < retries:
            try:
                wait = float(r.json().get("retryAfter", 0)) or 0
            except ValueError:
                wait = 0
            time.sleep(min(30, wait or float(r.headers.get("Retry-After", 0) or 0) or 3 * (attempt + 1)))
            continue
        r.raise_for_status()
        return r
    r.raise_for_status()
    return r


def _first_pdf(locations) -> str:
    for loc in locations:
        if loc and loc.get("pdf_url"):
            return loc["pdf_url"]
    return ""


# ---- connectors -------------------------------------------------------------------

def search_openalex(query: str, limit: int, year_from=None, year_to=None, publisher_ids: str = "") -> list[PaperMeta]:
    q = query.replace(",", " ")
    filters = [f"title_and_abstract.search:{q}"]
    if publisher_ids:
        filters.append(f"primary_location.source.publisher_lineage:{publisher_ids}")
    if year_from:
        filters.append(f"from_publication_date:{year_from}-01-01")
    if year_to:
        filters.append(f"to_publication_date:{year_to}-12-31")
    params = {"filter": ",".join(filters), "per_page": min(limit, 200)}
    if key := settings_store.get("openalex_key"):
        params["api_key"] = key
    return _parse_openalex(_get("https://api.openalex.org/works", params=params).json().get("results", []))


def _parse_openalex(results: list) -> list[PaperMeta]:
    out = []
    for w in results:
        doi = (w.get("doi") or "").replace("https://doi.org/", "")
        locations = [w.get("best_oa_location") or {}, *(w.get("locations") or [])]
        arxiv = ""
        for loc in w.get("locations") or []:
            if m := re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", (loc or {}).get("landing_page_url") or ""):
                arxiv = m.group(1)
        src = ((w.get("primary_location") or {}).get("source")) or {}
        out.append(PaperMeta(
            title=_clean(w.get("title") or ""),
            authors=[a["author"]["display_name"] for a in w.get("authorships", []) if a.get("author")],
            year=w.get("publication_year"),
            venue=_clean(src.get("display_name") or ""),
            doi=doi, arxiv_id=arxiv,
            abstract=_openalex_abstract(w.get("abstract_inverted_index")),
            url=f"https://doi.org/{doi}" if doi else (w.get("id") or ""),
            pdf_url=_first_pdf(locations) or (f"https://arxiv.org/pdf/{arxiv}" if arxiv else ""),
            cited_by=w.get("cited_by_count"),
            retracted=bool(w.get("is_retracted")),
            openalex_id=(w.get("id") or "").rsplit("/", 1)[-1],
        ))
    return out


def _openalex_works(filter_: str, limit: int, sort: str = "") -> list[PaperMeta]:
    """Raw OpenAlex works query (used for citation snowballing)."""
    params = {"filter": filter_, "per_page": min(limit, 200)}
    if sort:
        params["sort"] = sort
    if key := settings_store.get("openalex_key"):
        params["api_key"] = key
    data = _get("https://api.openalex.org/works", params=params).json()
    # Reuse the parser of search_openalex by faking its response shape.
    return _parse_openalex(data.get("results", []))


def snowball(doi: str, openalex_id: str = "", backward: int = 50, forward: int = 25, arxiv_id: str = "",
             title: str = "") -> tuple[list[PaperMeta], list[PaperMeta]]:
    """(references of the paper, papers citing it) from OpenAlex."""
    params = {"api_key": k} if (k := settings_store.get("openalex_key")) else None
    idents = [openalex_id, f"doi:{doi}" if doi else "", f"doi:10.48550/arXiv.{arxiv_id}" if arxiv_id else ""]
    w = None
    for ident in filter(None, idents):
        try:
            w = _get(f"https://api.openalex.org/works/{ident}", params=params, retries=2).json()
            break
        except Exception:
            continue
    if w is None and title:  # last resort: exact-ish title match
        try:
            res = _get("https://api.openalex.org/works", params={
                **(params or {}), "filter": f"title.search:{re.sub(r'[,:|]', ' ', title)[:200]}", "per_page": 3}).json()
            w = next((x for x in res.get("results", []) if _norm(x.get("title") or "") == _norm(title)), None)
        except Exception:
            w = None
    if not w:
        return [], []
    wid = (w.get("id") or "").rsplit("/", 1)[-1]
    refs = [r.rsplit("/", 1)[-1] for r in (w.get("referenced_works") or [])][:backward]
    back = []
    for i in range(0, len(refs), 50):
        back += _openalex_works("openalex:" + "|".join(refs[i:i + 50]), 50)
    fwd = _openalex_works(f"cites:{wid}", forward, "cited_by_count:desc") if wid else []
    return back, fwd


def search_arxiv(query: str, limit: int, year_from=None, year_to=None) -> list[PaperMeta]:
    # Convert a Boolean string to arXiv syntax: quoted phrases and words become all:..., AND/OR kept.
    def term(m):
        t = m.group(0)
        return t if t in ("AND", "OR", "NOT", "(", ")") else f"all:{t}"
    q = re.sub(r'"[^"]+"|\(|\)|[^\s()"]+', term, query).replace("NOT ", "ANDNOT ")
    root = ET.fromstring(_get("https://export.arxiv.org/api/query", params={
        "search_query": q, "max_results": limit, "sortBy": "relevance"}).content)
    out = [_arxiv_entry(e) for e in root.findall("a:entry", _ATOM)]
    return [p for p in out if (not year_from or (p.year or 0) >= year_from) and (not year_to or (p.year or 0) <= year_to)]


def search_europepmc(query: str, limit: int, year_from=None, year_to=None) -> list[PaperMeta]:
    q = query
    if year_from or year_to:
        q += f" AND PUB_YEAR:[{year_from or 1900} TO {year_to or 2100}]"
    data = _get("https://www.ebi.ac.uk/europepmc/webservices/rest/search", params={
        "query": q, "format": "json", "resultType": "core", "pageSize": min(limit, 100)}).json()
    out = []
    for r in data.get("resultList", {}).get("result", []):
        pdf = next((u["url"] for u in r.get("fullTextUrlList", {}).get("fullTextUrl", [])
                    if u.get("documentStyle") == "pdf" and u.get("availabilityCode") in ("OA", "F")), "")
        if not pdf and r.get("pmcid") and r.get("isOpenAccess") == "Y":
            pdf = f"https://europepmc.org/articles/{r['pmcid']}?pdf=render"
        out.append(PaperMeta(
            title=_clean(r.get("title", "")).rstrip("."),
            authors=[a.strip() for a in (r.get("authorString") or "").rstrip(".").split(",") if a.strip()],
            year=int(r["pubYear"]) if str(r.get("pubYear", "")).isdigit() else None,
            venue=((r.get("journalInfo") or {}).get("journal") or {}).get("title", "") or r.get("bookOrReportDetails", {}).get("publisher", ""),
            doi=r.get("doi", ""), abstract=_clean(r.get("abstractText", "")),
            url=f"https://doi.org/{r['doi']}" if r.get("doi") else f"https://europepmc.org/article/{r.get('source')}/{r.get('id')}",
            pdf_url=pdf, cited_by=r.get("citedByCount"),
        ))
    return out


def search_semantic_scholar(query: str, limit: int, year_from=None, year_to=None) -> list[PaperMeta]:
    plain = re.sub(r"\b(AND|OR|NOT)\b|[()\"]", " ", query)
    params = {"query": " ".join(plain.split())[:300], "limit": min(limit, 100),
              "fields": "title,authors,year,venue,abstract,externalIds,openAccessPdf,citationCount,url"}
    if year_from or year_to:
        params["year"] = f"{year_from or ''}-{year_to or ''}"
    headers = {"x-api-key": k} if (k := settings_store.get("semantic_scholar_key")) else {}
    out = []
    for p in _get("https://api.semanticscholar.org/graph/v1/paper/search", params=params, headers=headers).json().get("data", []):
        ids = p.get("externalIds") or {}
        out.append(PaperMeta(
            title=_clean(p.get("title", "")), authors=[a.get("name", "") for a in p.get("authors", [])],
            year=p.get("year"), venue=p.get("venue", ""), doi=ids.get("DOI", ""), arxiv_id=ids.get("ArXiv", ""),
            abstract=_clean(p.get("abstract") or ""), url=p.get("url", ""),
            pdf_url=(p.get("openAccessPdf") or {}).get("url") or "", cited_by=p.get("citationCount"),
        ))
    return out


def search_ieee(query: str, limit: int, year_from=None, year_to=None) -> list[PaperMeta]:
    params = {"querytext": query, "max_records": min(limit, 200), "apikey": settings_store.get("ieee_key")}
    if year_from:
        params["start_year"] = year_from
    if year_to:
        params["end_year"] = year_to
    out = []
    for a in _get("https://ieeexploreapi.ieee.org/api/v1/search/articles", params=params).json().get("articles", []):
        out.append(PaperMeta(
            title=_clean(a.get("title", "")),
            authors=[x.get("full_name", "") for x in (a.get("authors") or {}).get("authors", [])],
            year=int(a["publication_year"]) if str(a.get("publication_year", "")).isdigit() else None,
            venue=a.get("publication_title", ""), doi=a.get("doi", ""), abstract=_clean(a.get("abstract", "")),
            url=a.get("html_url") or (f"https://doi.org/{a['doi']}" if a.get("doi") else ""),
            pdf_url=a.get("pdf_url", "") if a.get("access_type", "").upper() in ("OPEN_ACCESS", "OPEN ACCESS") else "",
            cited_by=a.get("citing_paper_count"),
        ))
    return out


def _elsevier_headers() -> dict:
    h = {"X-ELS-APIKey": settings_store.get("elsevier_key"), "Accept": "application/json"}
    if tok := settings_store.get("elsevier_insttoken"):
        h["X-ELS-Insttoken"] = tok
    return h


def search_scopus(query: str, limit: int, year_from=None, year_to=None) -> list[PaperMeta]:
    q = query if query.upper().startswith(("TITLE-ABS-KEY", "TITLE(", "ABS(")) else f"TITLE-ABS-KEY({query})"
    if year_from:
        q += f" AND PUBYEAR > {year_from - 1}"
    if year_to:
        q += f" AND PUBYEAR < {year_to + 1}"
    data = _get("https://api.elsevier.com/content/search/scopus", params={"query": q, "count": min(limit, 25)},
                headers=_elsevier_headers()).json()
    out = []
    for e in data.get("search-results", {}).get("entry", []):
        if "error" in e:
            continue
        doi = e.get("prism:doi", "")
        out.append(PaperMeta(
            title=_clean(e.get("dc:title", "")), authors=[e["dc:creator"]] if e.get("dc:creator") else [],
            year=int(e["prism:coverDate"][:4]) if e.get("prism:coverDate", "")[:4].isdigit() else None,
            venue=e.get("prism:publicationName", ""), doi=doi, abstract=_clean(e.get("dc:description", "")),
            url=f"https://doi.org/{doi}" if doi else "",
            cited_by=int(e["citedby-count"]) if str(e.get("citedby-count", "")).isdigit() else None,
        ))
    return out


def search_sciencedirect(query: str, limit: int, year_from=None, year_to=None) -> list[PaperMeta]:
    body = {"qs": query, "display": {"show": min(limit, 100)}}
    if year_from or year_to:
        body["date"] = f"{year_from or 1900}-{year_to or 2100}"
    r = requests.put("https://api.elsevier.com/content/search/sciencedirect", json=body,
                     headers={**HEADERS, **_elsevier_headers(), "Content-Type": "application/json"}, timeout=TIMEOUT)
    r.raise_for_status()
    out = []
    for x in r.json().get("results", []):
        doi = x.get("doi", "")
        out.append(PaperMeta(
            title=_clean(x.get("title", "")), authors=[a.get("name", "") for a in x.get("authors", []) or []],
            year=int(x["publicationDate"][:4]) if str(x.get("publicationDate", ""))[:4].isdigit() else None,
            venue=x.get("sourceTitle", ""), doi=doi, url=x.get("uri") or (f"https://doi.org/{doi}" if doi else ""),
        ))
    return out


def search_springer(query: str, limit: int, year_from=None, year_to=None) -> list[PaperMeta]:
    q = query
    if year_from or year_to:
        q += f" AND onlinedatefrom:{year_from or 1900}-01-01 AND onlinedateto:{year_to or 2100}-12-31"
    data = _get("https://api.springernature.com/meta/v2/json",
                params={"q": q, "p": min(limit, 50), "api_key": settings_store.get("springer_key")}).json()
    out = []
    for r in data.get("records", []):
        pdf = next((u["value"] for u in r.get("url", []) if u.get("format") == "pdf"), "")
        out.append(PaperMeta(
            title=_clean(r.get("title", "")), authors=[c.get("creator", "") for c in r.get("creators", [])],
            year=int(r["publicationDate"][:4]) if str(r.get("publicationDate", ""))[:4].isdigit() else None,
            venue=r.get("publicationName", ""), doi=r.get("doi", ""), abstract=_clean(str(r.get("abstract", ""))),
            url=f"https://doi.org/{r['doi']}" if r.get("doi") else "",
            pdf_url=pdf if str(r.get("openaccess")).lower() == "true" else "",
        ))
    return out


def search_core(query: str, limit: int, year_from=None, year_to=None) -> list[PaperMeta]:
    q = query
    if year_from:
        q += f" AND yearPublished>={year_from}"
    if year_to:
        q += f" AND yearPublished<={year_to}"
    data = _get("https://api.core.ac.uk/v3/search/works", params={"q": q, "limit": min(limit, 100)},
                headers={"Authorization": f"Bearer {settings_store.get('core_key')}"}).json()
    out = []
    for r in data.get("results", []):
        out.append(PaperMeta(
            title=_clean(r.get("title") or ""), authors=[a.get("name", "") for a in r.get("authors", [])],
            year=r.get("yearPublished"), venue=r.get("publisher") or "", doi=r.get("doi") or "",
            abstract=_clean(r.get("abstract") or ""),
            url=f"https://doi.org/{r['doi']}" if r.get("doi") else (r.get("links") or [{}])[0].get("url", ""),
            pdf_url=r.get("downloadUrl") or "", cited_by=r.get("citationCount"),
        ))
    return out


def run(source_key: str, query: str, limit: int, year_from=None, year_to=None) -> list[PaperMeta]:
    if source_key == "openalex":
        return search_openalex(query, limit, year_from, year_to)
    if source_key.startswith("pub:"):
        return search_openalex(query, limit, year_from, year_to, PUBLISHERS[source_key[4:]])
    fn = {
        "arxiv": search_arxiv, "europepmc": search_europepmc, "semantic_scholar": search_semantic_scholar,
        "ieee_api": search_ieee, "scopus": search_scopus, "sciencedirect": search_sciencedirect,
        "springer_api": search_springer, "core": search_core,
    }[source_key]
    return fn(query, limit, year_from, year_to)


# ---- full-text resolution ------------------------------------------------------------

def _norm(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (title or "").lower())[:120]


def dedupe_key(p: PaperMeta) -> str:
    return (p.doi or "").lower() or (f"arxiv:{p.arxiv_id}" if p.arxiv_id else "") or _norm(p.title)


def merge(a: PaperMeta, b: PaperMeta) -> PaperMeta:
    """Fill gaps in a with data from b (same paper from another source)."""
    for f in ("title", "venue", "doi", "arxiv_id", "abstract", "url", "pdf_url"):
        if not getattr(a, f) and getattr(b, f):
            setattr(a, f, getattr(b, f))
    if len(b.abstract or "") > len(a.abstract or ""):
        a.abstract = b.abstract
    if not a.authors or (len(b.authors) > len(a.authors)):
        a.authors = b.authors or a.authors
    a.year = a.year or b.year
    a.cited_by = max(a.cited_by or 0, b.cited_by or 0) or a.cited_by
    return a


def pdf_candidates(p: PaperMeta) -> list[str]:
    """Every legal open-access PDF location we can find, best first."""
    urls = [p.pdf_url] if p.pdf_url else []
    if p.arxiv_id:
        urls.append(f"https://arxiv.org/pdf/{p.arxiv_id}")
    if p.doi:
        try:
            params = {"api_key": k} if (k := settings_store.get("openalex_key")) else None
            w = _get(f"https://api.openalex.org/works/doi:{p.doi}", params=params, retries=1).json()
            locs = [w.get("best_oa_location") or {}, *(w.get("locations") or [])]
            urls += [l["pdf_url"] for l in locs if l and l.get("pdf_url")]
            for l in locs:
                if m := re.search(r"arxiv\.org/(?:abs|pdf)/(\d{4}\.\d{4,5})", (l or {}).get("landing_page_url") or ""):
                    urls.append(f"https://arxiv.org/pdf/{m.group(1)}")
                pmcid = re.search(r"(PMC\d+)", (l or {}).get("landing_page_url") or "")
                if pmcid:
                    urls.append(f"https://europepmc.org/articles/{pmcid.group(1)}?pdf=render")
        except Exception:
            pass
        # PubMed Central copy (many open-access journal articles are mirrored there).
        try:
            r = _get("https://www.ebi.ac.uk/europepmc/webservices/rest/search",
                     params={"query": f'DOI:"{p.doi}"', "format": "json", "resultType": "lite"}, retries=1).json()
            for x in r.get("resultList", {}).get("result", []):
                if x.get("pmcid") and x.get("isOpenAccess") == "Y":
                    urls.append(f"https://europepmc.org/articles/{x['pmcid']}?pdf=render")
        except Exception:
            pass
    if not p.arxiv_id and p.title:
        urls += arxiv_by_title(p.title)
    return list(dict.fromkeys(u for u in urls if u))


def arxiv_by_title(title: str) -> list[str]:
    """Preprint versions of a (possibly paywalled) paper, matched by near-identical title."""
    stop = {"the", "and", "for", "with", "from", "into", "via", "using", "based", "towards", "toward", "its", "are"}
    words = [w for w in re.findall(r"[A-Za-z0-9]+", title) if len(w) > 2 and w.lower() not in stop][:8]
    if len(words) < 3:
        return []
    try:
        root = ET.fromstring(_get("https://export.arxiv.org/api/query", params={
            "search_query": " AND ".join(f"ti:{w}" for w in words), "max_results": 5}, retries=1).content)
    except Exception:
        return []
    out = []
    for e in root.findall("a:entry", _ATOM):
        p = _arxiv_entry(e)
        if _norm(p.title) == _norm(title) or (_norm(p.title) and _similar(p.title, title) > 0.9):
            out.append(f"https://arxiv.org/pdf/{p.arxiv_id}")
    return out


def _similar(a: str, b: str) -> float:
    wa, wb = set(re.findall(r"\w+", a.lower())), set(re.findall(r"\w+", b.lower()))
    return len(wa & wb) / max(1, len(wa | wb))


BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/129.0 Safari/537.36",
    "Accept": "application/pdf,*/*;q=0.8",
}


def fetch_pdf(p: PaperMeta) -> tuple[bytes | None, str]:
    """(pdf bytes, url) for the first candidate that returns a real PDF, else (None, '')."""
    for url in pdf_candidates(p):
        try:
            r = requests.get(url, headers=BROWSER_HEADERS, timeout=60, allow_redirects=True)
            if r.ok and r.content[:5] == b"%PDF-":
                return r.content, url
        except Exception:
            continue
    return None, ""

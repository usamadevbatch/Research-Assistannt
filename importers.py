"""Import search results exported from Scopus, IEEE Xplore, ACM DL, ScienceDirect, Web of Science, Zotero...

Supports RIS, BibTeX and CSV exports.
"""
import csv
import io
import re

from metadata import PaperMeta, _clean


def _year(s) -> int | None:
    m = re.search(r"(19|20)\d{2}", str(s or ""))
    return int(m.group(0)) if m else None


def parse_ris(text: str) -> list[PaperMeta]:
    out, cur = [], None
    for line in text.splitlines():
        m = re.match(r"^([A-Z][A-Z0-9])  - ?(.*)$", line)
        if not m:
            continue
        tag, val = m.group(1), m.group(2).strip()
        if tag == "TY":
            cur = PaperMeta()
        elif cur is None:
            continue
        elif tag == "ER":
            if cur.title:
                out.append(cur)
            cur = None
        elif tag in ("TI", "T1") and not cur.title:
            cur.title = _clean(val)
        elif tag in ("AU", "A1"):
            cur.authors.append(val)
        elif tag in ("PY", "Y1", "DA") and not cur.year:
            cur.year = _year(val)
        elif tag in ("T2", "JO", "JF", "BT") and not cur.venue:
            cur.venue = val
        elif tag == "DO":
            cur.doi = re.sub(r"^https?://(dx\.)?doi\.org/", "", val)
        elif tag == "AB":
            cur.abstract = _clean(val)
        elif tag == "UR" and not cur.url:
            cur.url = val
        elif tag == "L1" and val.lower().endswith(".pdf"):
            cur.pdf_url = val
    return out


def _bib_fields(body: str) -> dict:
    fields, i = {}, 0
    for m in re.finditer(r"(\w+)\s*=\s*", body):
        if m.start() < i:
            continue
        key, j = m.group(1).lower(), m.end()
        if j < len(body) and body[j] == "{":
            depth, k = 0, j
            while k < len(body):
                depth += body[k] == "{"
                depth -= body[k] == "}"
                if depth == 0:
                    break
                k += 1
            fields[key], i = body[j + 1:k], k + 1
        elif j < len(body) and body[j] == '"':
            k = body.find('"', j + 1)
            fields[key], i = body[j + 1:k], k + 1
        else:
            v = re.match(r"[^,}\n]+", body[j:])
            fields[key], i = (v.group(0).strip() if v else ""), j + (v.end() if v else 0)
    return fields


def _tex(s: str) -> str:
    return re.sub(r"[{}]", "", s).replace("\\&", "&").replace("\\_", "_").strip()


def parse_bibtex(text: str) -> list[PaperMeta]:
    out = []
    for m in re.finditer(r"@\w+\s*\{[^,]*,(.*?)\n\}", text, re.S):
        f = _bib_fields(m.group(1))
        title = _clean(_tex(f.get("title", "")))
        if not title:
            continue
        out.append(PaperMeta(
            title=title,
            authors=[re.sub(r"[{}]", "", a).strip() for a in re.split(r"\s+and\s+", f.get("author", "")) if a.strip()],
            year=_year(f.get("year")), venue=_tex(f.get("journal") or f.get("booktitle") or ""),
            doi=f.get("doi", "").strip(), abstract=_clean(re.sub(r"[{}]", "", f.get("abstract", ""))),
            url=f.get("url", ""), arxiv_id=f.get("eprint", "") if "arxiv" in f.get("archiveprefix", "").lower() else "",
        ))
    return out


# Column names used by Scopus, IEEE Xplore, ACM, ScienceDirect and Web of Science CSV exports.
_CSV = {
    "title": ("title", "document title", "article title", "ti"),
    "authors": ("authors", "author", "author full names", "au"),
    "year": ("year", "publication year", "py", "publication date"),
    "venue": ("source title", "publication title", "journal", "source", "so", "publication"),
    "doi": ("doi", "di"),
    "abstract": ("abstract", "ab"),
    "url": ("link", "url", "document link"),
    "pdf_url": ("pdf link",),
    "cited_by": ("cited by", "article citation count", "times cited, all databases", "tc"),
}


def parse_csv(text: str) -> list[PaperMeta]:
    reader = csv.DictReader(io.StringIO(text.lstrip("﻿")))
    header = {h.lower().strip(): h for h in reader.fieldnames or []}
    col = {k: next((header[n] for n in names if n in header), None) for k, names in _CSV.items()}
    out = []
    for row in reader:
        def g(k):
            return (row.get(col[k]) or "").strip() if col[k] else ""
        if not g("title"):
            continue
        authors = g("authors")
        sep = ";" if ";" in authors else ","
        cited = re.sub(r"\D", "", g("cited_by"))
        out.append(PaperMeta(
            title=_clean(g("title")), authors=[a.strip() for a in authors.split(sep) if a.strip()],
            year=_year(g("year")), venue=g("venue"), doi=g("doi"), abstract=_clean(g("abstract")),
            url=g("url"), pdf_url=g("pdf_url"), cited_by=int(cited) if cited else None,
        ))
    return out


def parse_file(name: str, data: bytes) -> list[PaperMeta]:
    text = data.decode("utf-8", errors="replace")
    low = name.lower()
    if low.endswith((".ris", ".txt")) and re.search(r"^TY  -", text, re.M):
        return parse_ris(text)
    if low.endswith((".bib", ".bibtex")) or text.lstrip().startswith("@"):
        return parse_bibtex(text)
    if low.endswith(".csv"):
        return parse_csv(text)
    if re.search(r"^TY  -", text, re.M):
        return parse_ris(text)
    raise ValueError("Unsupported file. Export results as RIS, BibTeX or CSV.")

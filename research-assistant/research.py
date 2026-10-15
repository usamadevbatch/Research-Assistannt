"""The research agent: idea -> plan -> multi-source search -> ranking -> AI screening -> full text -> tables -> report.

Every function takes its dependencies explicitly so it can run in a background thread (no Streamlit calls here).
"""
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime

import numpy as np

import sources
from config import PDF_DIR, settings
from ingest import chunk_pages, parse_pdf, sha256_of
from metadata import PaperMeta
from retrieval import HybridIndex, rerank

# ---- prompts ------------------------------------------------------------------------

PLAN_PROMPT = """You are an expert research librarian who designs systematic literature searches.
Given a research idea, return JSON with exactly these keys:
{
  "title": "short project title (max 8 words)",
  "research_questions": ["3-5 precise research questions"],
  "concepts": [{"name": "core concept as written in paper titles", "synonyms": ["3-6 synonyms, abbreviations, spelling variants or narrower terms used in the literature"]}],
  "inclusion_criteria": ["3-5 criteria a paper must meet"],
  "exclusion_criteria": ["2-4 reasons to exclude a paper"],
  "extraction_columns": [{"name": "short column name", "instruction": "what to extract from each paper", "type": "text|number|yes/no"}]
}
Rules for concepts:
- Each concept is ONE idea (2-4 concepts in total). Concepts are combined with AND, a concept's synonyms with OR.
- Never put "AND" or several ideas inside one concept name.
- Synonyms must mean the same thing as their concept (true synonyms, abbreviations, variants), not broader topics.
- Use terminology that actually appears in paper titles and abstracts.
Example for "deep learning to detect diabetic retinopathy from fundus images":
"concepts": [
  {"name": "deep learning", "synonyms": ["convolutional neural network", "CNN", "neural network", "transformer"]},
  {"name": "diabetic retinopathy", "synonyms": ["DR", "diabetic eye disease"]},
  {"name": "fundus image", "synonyms": ["retinal image", "fundus photograph", "fundus photography"]}
]
Extraction columns: 5-8 human-readable names (e.g. "Privacy technique", not "privacy_technique") for comparing
papers; do not include title, authors or year (those are added automatically)."""

SCREEN_PROMPT = """You screen papers for a literature review. For each paper decide:
- "include": clearly relevant and meets the inclusion criteria
- "maybe": possibly relevant, or not enough information (e.g. no abstract)
- "exclude": off-topic or meets an exclusion criterion
Return JSON: {"decisions": [{"id": <paper number>, "decision": "include|maybe|exclude", "reason": "one short sentence"}]}"""

EXTRACT_PROMPT = """You extract data from one research paper into a table. Use ONLY the numbered excerpts.
For each field return the value and the excerpt numbers that support it.
If the paper does not report it, use "Not reported" with no sources. Be concise (max ~25 words per value),
keep exact numbers and units, and for yes/no fields answer "Yes" or "No" with a short qualifier if needed.
Return JSON: {"<field name>": {"value": "...", "sources": [1, 2]}, ...} with every field name exactly as given."""

REPORT_PROMPT = """You are writing a literature review section from a set of papers. Use ONLY the information given.
Cite papers with their number in square brackets, e.g. [3] or [2][5], right after each claim.
Structure: ## Overview, then thematic sections (## headings) grouping related work, ## Comparison
(a Markdown table of the most important papers if helpful), ## Research gaps and future directions, ## Conclusion.
Be specific: methods, datasets, results with numbers. Do not wrap the answer in a code block."""


# ---- plan --------------------------------------------------------------------------------

def make_plan(llm, idea: str, notes: str = "") -> dict:
    msg = f"Research idea:\n{idea}" + (f"\n\nExtra requirements:\n{notes}" if notes.strip() else "")
    try:
        plan = llm.complete_json(PLAN_PROMPT, [{"role": "user", "content": msg}])
    except ValueError:
        plan = {}  # fall back to a keyword plan the user can edit
    return normalize_plan(plan if isinstance(plan, dict) else {}, idea)


def normalize_plan(plan: dict, idea: str) -> dict:
    def strings(v):
        return [str(x).strip() for x in (v or []) if str(x).strip()] if isinstance(v, list) else []
    concepts = []
    for c in plan.get("concepts") or []:
        if not (isinstance(c, dict) and str(c.get("name", "")).strip()):
            continue
        names = [n.strip() for n in re.split(r"\s+AND\s+|\s*&\s*|,", str(c["name"])) if n.strip()]
        syns = strings(c.get("synonyms"))[:8]
        if len(names) == 1:
            concepts.append({"name": names[0], "synonyms": list(dict.fromkeys(syns))})
            continue
        # The model merged several ideas into one concept: split it and give each synonym to the
        # concept it shares the most words with (synonyms matching none are dropped as too broad).
        split = [{"name": n, "synonyms": []} for n in names]
        for s in syns:
            words = set(re.findall(r"\w+", s.lower()))
            overlap = [len(words & set(re.findall(r"\w+", n.lower()))) for n in names]
            if max(overlap) > 0:
                split[overlap.index(max(overlap))]["synonyms"].append(s)
        concepts += split
    if not concepts:
        concepts = [{"name": w, "synonyms": []} for w in re.findall(r"\w{4,}", idea)[:3]]
    cols = []
    for c in plan.get("extraction_columns") or []:
        if isinstance(c, dict) and str(c.get("name", "")).strip():
            name = str(c["name"]).replace("_", " ").strip()[:40]
            name = name[:1].upper() + name[1:]
            if name.lower() in ("year", "title", "authors", "author", "publication year", "venue"):
                continue
            t = str(c.get("type", "text")).lower()
            cols.append({"name": name, "instruction": str(c.get("instruction", "")).strip(),
                         "type": t if t in ("text", "number", "yes/no") else "text"})
    return {
        "title": str(plan.get("title") or idea[:60]).strip(),
        "research_questions": strings(plan.get("research_questions")),
        "concepts": concepts[:5],
        "inclusion_criteria": strings(plan.get("inclusion_criteria")),
        "exclusion_criteria": strings(plan.get("exclusion_criteria")),
        "extraction_columns": cols or DEFAULT_COLUMNS,
    }


DEFAULT_COLUMNS = [
    {"name": "Aim", "instruction": "Main research question or objective", "type": "text"},
    {"name": "Method", "instruction": "Approach, model, or study design", "type": "text"},
    {"name": "Data", "instruction": "Datasets, participants, or sample size", "type": "text"},
    {"name": "Key results", "instruction": "Main quantitative results with numbers", "type": "text"},
    {"name": "Limitations", "instruction": "Limitations stated or evident", "type": "text"},
]

TEMPLATES = {
    "General (aim, method, data, results, limitations)": DEFAULT_COLUMNS,
    "Computer science / engineering": [
        {"name": "Problem", "instruction": "Problem addressed", "type": "text"},
        {"name": "Proposed approach", "instruction": "Proposed method, model or architecture", "type": "text"},
        {"name": "Datasets", "instruction": "Datasets or benchmarks used", "type": "text"},
        {"name": "Metrics", "instruction": "Evaluation metrics", "type": "text"},
        {"name": "Best result", "instruction": "Best reported result with metric and value", "type": "text"},
        {"name": "Baselines", "instruction": "Methods compared against", "type": "text"},
        {"name": "Code available", "instruction": "Is source code publicly released?", "type": "yes/no"},
        {"name": "Limitations", "instruction": "Limitations or future work", "type": "text"},
    ],
    "Clinical / health (PICO)": [
        {"name": "Population", "instruction": "Population or participants and sample size", "type": "text"},
        {"name": "Intervention", "instruction": "Intervention or exposure", "type": "text"},
        {"name": "Comparator", "instruction": "Comparison or control group", "type": "text"},
        {"name": "Outcomes", "instruction": "Primary outcomes and results with effect sizes", "type": "text"},
        {"name": "Study design", "instruction": "Study design (RCT, cohort, ...)", "type": "text"},
        {"name": "Sample size", "instruction": "Total number of participants", "type": "number"},
        {"name": "Risk of bias", "instruction": "Limitations or risk-of-bias concerns", "type": "text"},
    ],
    "Quality appraisal (CASP-style)": [
        {"name": "Clear aims", "instruction": "Are the research aims clearly stated?", "type": "yes/no"},
        {"name": "Appropriate design", "instruction": "Is the study design appropriate for the aims? Why?", "type": "text"},
        {"name": "Sample / data adequate", "instruction": "Is the sample or dataset adequate and described?", "type": "text"},
        {"name": "Bias addressed", "instruction": "Are confounders, bias or threats to validity discussed?", "type": "yes/no"},
        {"name": "Results reproducible", "instruction": "Are code, data or enough detail provided to reproduce?", "type": "yes/no"},
        {"name": "Results clearly reported", "instruction": "Are results reported with statistics or metrics?", "type": "yes/no"},
        {"name": "Overall quality", "instruction": "Overall quality: High, Medium or Low, with a short reason", "type": "text"},
    ],
    "Social science / survey": [
        {"name": "Research question", "instruction": "Research question or hypothesis", "type": "text"},
        {"name": "Theory", "instruction": "Theoretical framework", "type": "text"},
        {"name": "Method", "instruction": "Qualitative/quantitative method", "type": "text"},
        {"name": "Sample", "instruction": "Sample, country and size", "type": "text"},
        {"name": "Findings", "instruction": "Main findings", "type": "text"},
        {"name": "Limitations", "instruction": "Limitations", "type": "text"},
    ],
}


# ---- search ----------------------------------------------------------------------------

def run_search(library, project_id: int, plan: dict, options: dict, report) -> dict:
    """Search every selected source, dedupe, store candidates. Returns per-source counts/errors."""
    query = sources.boolean_query(plan["concepts"])
    simple = " AND ".join(f'"{c["name"]}"' if " " in c["name"] else c["name"] for c in plan["concepts"])
    keys = options.get("sources", ["openalex"])
    limit = int(options.get("per_source", 25))
    yf, yt = options.get("year_from"), options.get("year_to")
    found: dict[str, PaperMeta] = {}
    found_in: dict[str, set] = {}
    by_title: dict[str, str] = {}  # normalized title -> key, to merge journal/preprint/conference versions
    stats = {}

    def one(key):
        res = sources.run(key, query, limit, yf, yt)
        if not res and simple != query:
            res = sources.run(key, simple, limit, yf, yt)
        return res

    # OpenAlex rate-limits bursts, so keep concurrency modest.
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {pool.submit(one, k): k for k in keys}
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            label = sources.BY_KEY[key].label if key in sources.BY_KEY else key
            try:
                res = fut.result()
                stats[label] = {"found": len(res)}
                for p in res:
                    if not p.title:
                        continue
                    k = sources.dedupe_key(p)
                    k = by_title.get(sources._norm(p.title), k)
                    if k in found:
                        sources.merge(found[k], p)
                    else:
                        found[k] = p
                        by_title[sources._norm(p.title)] = k
                    found_in.setdefault(k, set()).add(label.split(" (")[0])
            except Exception as e:
                stats[label] = {"found": 0, "error": str(e)[:200]}
            report(i / len(keys) * 0.7, f"Searched {i}/{len(keys)} sources · {len(found)} unique papers so far")

    # Also merge with papers already in the project (from an earlier search or an import).
    existing = {sources._norm(c["meta"]["title"]): c["key"] for c in library.candidates(project_id)}
    rows = []
    for k, p in found.items():
        rows.append((existing.get(sources._norm(p.title), k), asdict(p), ", ".join(sorted(found_in[k]))))
    library.upsert_candidates(project_id, rows)
    stats["_total_found"] = sum(v.get("found", 0) for v in stats.values() if isinstance(v, dict))
    stats["_unique"] = len(found)
    stats["_searched_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    stats["_query"] = query
    stats["_years"] = [yf, yt]
    return stats


def rank_candidates(library, embedder, reranker, project_id: int, idea: str, plan: dict, report) -> None:
    """Relevance 0-100 from a cross-encoder over title + abstract (embedding pre-ranking for large sets)."""
    cands = library.candidates(project_id)
    if not cands:
        return
    texts = [f"{c['meta']['title']}. {c['meta'].get('abstract', '')[:1500]}" for c in cands]
    query = idea + " " + " ".join(plan.get("research_questions", [])[:2])
    report(0.75, f"Ranking {len(cands)} papers by relevance…")
    vecs = embedder.embed_passages(texts)
    dense = vecs @ embedder.embed_query(query)
    order = np.argsort(-dense)
    lo, hi = float(dense.min()), float(dense.max())
    scores = {int(i): 100 * (float(dense[i]) - lo) / max(1e-6, hi - lo) for i in order}
    top = [int(i) for i in order[:150]]
    if reranker and top:
        ce = np.asarray(reranker.scores(idea, [texts[i] for i in top]), dtype=float)
        ce_lo, ce_hi = float(ce.min()), float(ce.max())
        # Blend: cross-encoder decides the order, scaled to 0-100 within this project; the long tail stays below.
        floor = max((scores[i] for i in order[150:]), default=0) if len(order) > 150 else 0
        for i, s in zip(top, ce):
            scores[i] = floor + (100 - floor) * (s - ce_lo) / max(1e-6, ce_hi - ce_lo)
    for i, c in enumerate(cands):
        library.update_candidate(c["id"], relevance=round(scores.get(i, 0.0), 1))


def screen(library, llm, project_id: int, idea: str, plan: dict, top_n: int, report, cancelled) -> int:
    screener = f"AI ({getattr(llm, 'model', 'model')})"
    """AI screening of the top-N pending papers by title + abstract. Returns number screened."""
    cands = [c for c in library.candidates(project_id) if c["decision"] == "pending"][:top_n]
    criteria = (
        f"Research idea: {idea}\n\nInclusion criteria:\n- " + "\n- ".join(plan.get("inclusion_criteria") or ["Relevant to the idea"])
        + "\n\nExclusion criteria:\n- " + "\n- ".join(plan.get("exclusion_criteria") or ["Off-topic"])
    )
    batch_size = 5
    done = 0
    for start in range(0, len(cands), batch_size):
        if cancelled():
            break
        batch = cands[start:start + batch_size]
        listing = "\n\n".join(
            f"Paper {i}: {c['meta']['title']} ({c['meta'].get('year') or 'n.d.'}, {c['meta'].get('venue') or 'unknown venue'})\n"
            f"Abstract: {(c['meta'].get('abstract') or 'No abstract available.')[:1400]}"
            for i, c in enumerate(batch, 1)
        )
        try:
            out = llm.complete_json(SCREEN_PROMPT, [{"role": "user", "content": f"{criteria}\n\n{listing}"}])
            decisions = {int(d.get("id", 0)): d for d in (out.get("decisions") or []) if isinstance(d, dict)}
        except Exception as e:
            decisions = {}
            report(None, f"Screening batch failed ({str(e)[:80]}), continuing…")
        for i, c in enumerate(batch, 1):
            d = decisions.get(i, {})
            decision = str(d.get("decision", "maybe")).lower()
            decision = decision if decision in ("include", "maybe", "exclude") else "maybe"
            library.update_candidate(c["id"], decision=decision, reason=str(d.get("reason", "Not screened"))[:300],
                                     decided_by=screener)
        done += len(batch)
        report(done / max(1, len(cands)), f"Screened {done}/{len(cands)} papers")
    return done


def snowball_included(library, embedder, reranker, project_id: int, idea: str, plan: dict, report, cancelled) -> dict:
    """Backward (references) and forward (citing papers) citation chasing from included papers."""
    included = [c for c in library.candidates(project_id) if c["decision"] == "include"]
    existing = {sources._norm(c["meta"]["title"]): c["key"] for c in library.candidates(project_id)}
    yf, yt = (library.get_project(project_id)["options"].get(k) for k in ("year_from", "year_to"))
    new, n_back, n_fwd = {}, 0, 0
    for i, c in enumerate(included, 1):
        if cancelled():
            break
        report((i - 1) / max(1, len(included)) * 0.8, f"Snowballing {i}/{len(included)}: {c['meta']['title'][:50]}")
        m = c["meta"]
        back, fwd = sources.snowball(m.get("doi", ""), m.get("openalex_id", ""), arxiv_id=m.get("arxiv_id", ""),
                                     title=m.get("title", ""))
        for papers, label in ((back, "Snowballing: references"), (fwd, "Snowballing: cited by")):
            for p in papers:
                if not p.title or sources._norm(p.title) in existing:
                    continue
                if label.endswith("cited by") and ((yf and (p.year or 0) < yf) or (yt and (p.year or 9999) > yt)):
                    continue
                k = sources.dedupe_key(p)
                if k not in new:
                    new[k] = (p, label)
                    n_back += label.endswith("references")
                    n_fwd += label.endswith("cited by")
    library.upsert_candidates(project_id, [(k, asdict(p), lbl) for k, (p, lbl) in new.items()])
    report(0.85, f"Added {len(new)} papers from citations; re-ranking…")
    rank_candidates(library, embedder, reranker, project_id, idea, plan, report)
    opts = library.get_project(project_id)["options"]
    log = opts.get("snowball_log", []) + [{"date": datetime.now().strftime("%Y-%m-%d %H:%M"),
                                           "from": len(included), "backward": n_back, "forward": n_fwd}]
    library.update_project(project_id, options={**opts, "snowball_log": log})
    return {"backward": n_back, "forward": n_fwd}


# ---- full text ------------------------------------------------------------------------------

def index_pdf(library, embedder, data: bytes, filename: str, meta: PaperMeta | None) -> tuple[str, str, int | None]:
    """Index a PDF into the library (no Streamlit). Returns (status, message, doc_id)."""
    import metadata as md
    sha = sha256_of(data)
    if library.has_document(sha):
        return "duplicate", f"{filename} is already in your library.", None
    try:
        parsed = parse_pdf(data, filename)
    except Exception as e:
        return "error", f"{filename}: could not read PDF ({e})", None
    chunks = chunk_pages(parsed.pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        return "error", f"{filename}: no extractable text. Scanned PDFs need OCR first (ocrmypdf in.pdf out.pdf).", None
    if meta is None:
        doi, arxiv_id = md.find_identifiers(" ".join(parsed.pages[:2]))
        meta = md.lookup(doi, arxiv_id) if (doi or arxiv_id) else None
        meta = meta or md.PaperMeta(doi=doi, arxiv_id=arxiv_id)
    fields = meta.as_document_fields()
    title = fields.pop("title", None) or parsed.title
    pages, texts = zip(*chunks)
    vectors = embedder.embed_passages(list(texts))
    (PDF_DIR / f"{sha}.pdf").write_bytes(data)
    doc_id = library.add_document(sha, title, filename, parsed.num_pages, pages, texts, vectors, **fields)
    return "added", f"Added “{title}” ({parsed.num_pages} pages).", doc_id


def attach_pdf(library, embedder, doc_id: int, data: bytes, filename: str) -> str:
    """Replace an abstract-only document's text with its full PDF."""
    parsed = parse_pdf(data, filename)
    chunks = chunk_pages(parsed.pages, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("No extractable text in this PDF (scanned PDFs need OCR first).")
    sha = sha256_of(data)
    pages, texts = zip(*chunks)
    vectors = embedder.embed_passages(list(texts))
    (PDF_DIR / f"{sha}.pdf").write_bytes(data)
    library.replace_chunks(doc_id, sha, filename, parsed.num_pages, pages, texts, vectors)
    return f"Attached the full PDF ({parsed.num_pages} pages)."


def index_abstract(library, embedder, meta: PaperMeta) -> int:
    """Add a paper we couldn't get a PDF for, using its abstract as the text."""
    text = f"{meta.title}\n\n{meta.abstract or 'No abstract available.'}"
    sha = "abstract-" + sha256_of((sources.dedupe_key(meta) or meta.title).encode())
    vectors = embedder.embed_passages([text])
    fields = meta.as_document_fields()
    title = fields.pop("title", None) or "Untitled"
    return library.add_document(sha, title, "(abstract only)", 1, [1], [text], vectors, **fields)


def get_fulltexts(library, embedder, project_id: int, report, cancelled) -> dict:
    cands = [c for c in library.candidates(project_id) if c["decision"] == "include" and not c["doc_id"]]
    counts = {"pdf": 0, "abstract": 0}
    for i, c in enumerate(cands, 1):
        if cancelled():
            break
        meta = PaperMeta(**{k: v for k, v in c["meta"].items() if k in PaperMeta.__dataclass_fields__})
        report((i - 1) / max(1, len(cands)), f"Fetching full text {i}/{len(cands)}: {meta.title[:60]}")
        existing = library.find_document(doi=meta.doi, arxiv_id=meta.arxiv_id)
        if existing:
            library.update_candidate(c["id"], doc_id=existing, fulltext="pdf")
            counts["pdf"] += 1
            continue
        data, url = sources.fetch_pdf(meta)
        doc_id = None
        if data:
            name = re.sub(r"[^\w\- ]", "", meta.title)[:80].strip() or "paper"
            status, _, doc_id = index_pdf(library, embedder, data, f"{name}.pdf", meta)
            if status == "duplicate":
                doc_id = library.find_document(doi=meta.doi, arxiv_id=meta.arxiv_id)
        if doc_id:
            library.update_candidate(c["id"], doc_id=doc_id, fulltext="pdf")
            counts["pdf"] += 1
        else:
            doc_id = index_abstract(library, embedder, meta)
            library.update_candidate(c["id"], doc_id=doc_id, fulltext="abstract")
            counts["abstract"] += 1
    report(1.0, f"Done: {counts['pdf']} full PDFs, {counts['abstract']} abstract only")
    return counts


# ---- extraction tables ----------------------------------------------------------------------

def extract_table(library, embedder, llm, table_id: int, budget_chars: int, report, cancelled, only_missing=True) -> None:
    table = library.get_table(table_id)
    cols = table["columns"]
    if not cols:
        return
    index = HybridIndex(library.load_snapshot())
    docs = {d.id: d for d in library.list_documents()}
    existing = library.cells(table_id)
    todo = [d for d in table["doc_ids"] if d in docs and
            (not only_missing or any(not existing.get((d, c["name"]), {}).get("value") for c in cols))]
    per_col = max(1, min(3, (budget_chars // 1200) // max(1, len(cols))))
    for n, doc_id in enumerate(todo, 1):
        if cancelled():
            break
        doc = docs[doc_id]
        report((n - 1) / max(1, len(todo)), f"Extracting {n}/{len(todo)}: {doc.title[:60]}")
        chunks = library.document_chunks(doc_id)
        if len(chunks) <= 3:
            picked = [(p, t) for p, t in chunks]
        else:
            seen, picked = set(), []
            for c in cols:
                q = f"{c['name']}: {c['instruction']}"
                hits = index.search(embedder.embed_query(q), q, per_col, {doc_id})
                for h in hits:
                    if (h.page, h.text) not in seen:
                        seen.add((h.page, h.text))
                        picked.append((h.page, h.text))
            # Always include the opening (title/abstract) for context.
            if chunks[0] not in seen:
                picked.insert(0, chunks[0])
            total, trimmed = 0, []
            for p, t in picked:
                if total + len(t) > budget_chars:
                    break
                trimmed.append((p, t))
                total += len(t)
            picked = trimmed
        excerpts = "\n\n".join(f"[{i}] (page {p})\n{t}" for i, (p, t) in enumerate(picked, 1))
        fields = "\n".join(f"- {c['name']} ({c['type']}): {c['instruction']}" for c in cols)
        prompt = f"Paper: {doc.title}\n\nFields to extract:\n{fields}\n\nExcerpts:\n\n{excerpts}"
        try:
            out = llm.complete_json(EXTRACT_PROMPT, [{"role": "user", "content": prompt}])
        except Exception as e:
            out = {}
            report(None, f"Extraction failed for one paper ({str(e)[:80]})")
        lower = {str(k).lower().strip(): v for k, v in (out or {}).items()} if isinstance(out, dict) else {}
        for c in cols:
            v = lower.get(c["name"].lower().strip())
            if isinstance(v, dict):
                value = str(v.get("value", "")).strip() or "Not reported"
                nums = [int(x) for x in (v.get("sources") or []) if str(x).isdigit()]
            else:
                value, nums = (str(v).strip() if v not in (None, "") else "Failed"), []
            srcs = [{"page": picked[k - 1][0], "text": picked[k - 1][1]} for k in nums if 0 < k <= len(picked)]
            if not srcs and len(picked) == 1 and value not in ("Not reported", "Failed"):
                srcs = [{"page": picked[0][0], "text": picked[0][1]}]  # abstract-only paper: the abstract is the source
            library.set_cell(table_id, doc_id, c["name"], value, srcs)
    report(1.0, "Table complete")


def table_frame(library, table_id: int):
    """pandas DataFrame of a table (paper rows x columns)."""
    import pandas as pd
    table = library.get_table(table_id)
    docs = {d.id: d for d in library.list_documents()}
    cells = library.cells(table_id)
    rows = []
    for doc_id in table["doc_ids"]:
        d = docs.get(doc_id)
        if not d:
            continue
        row = {"Paper": d.title, "Authors": d.short_authors, "Year": d.year, "Venue": d.venue,
               "Full text": "PDF" if (PDF_DIR / f"{d.sha256}.pdf").exists() else "Abstract only"}
        for c in table["columns"]:
            row[c["name"]] = cells.get((doc_id, c["name"]), {}).get("value", "")
        rows.append(row)
    return pd.DataFrame(rows)


# ---- report -------------------------------------------------------------------------------------

def write_report(library, llm, project_id: int, table_id: int | None, budget_chars: int) -> tuple[str, list[int]]:
    """(review Markdown without a reference list, doc ids in citation-number order)."""
    project = library.get_project(project_id)
    docs = {d.id: d for d in library.list_documents()}
    included = [c for c in library.candidates(project_id) if c["decision"] == "include" and c["doc_id"] in docs]
    cells = library.cells(table_id) if table_id else {}
    cols = library.get_table(table_id)["columns"] if table_id else []
    per = max(400, budget_chars // max(1, len(included)))
    blocks, refs = [], []
    for i, c in enumerate(included, 1):
        d = docs[c["doc_id"]]
        facts = "; ".join(f"{col['name']}: {cells[(d.id, col['name'])]['value']}"
                          for col in cols if (d.id, col["name"]) in cells
                          and cells[(d.id, col["name"])]["value"] not in ("Not reported", "Failed"))
        body = facts or (d.abstract or "")
        blocks.append(f"[{i}] {d.title} ({d.citation or d.year or ''})\n{body[:per]}")
        refs.append(f"{i}. {d.authors.replace(';', ',')} ({d.year or 'n.d.'}). {d.title}. {d.venue}."
                    + (f" https://doi.org/{d.doi}" if d.doi else ""))
    if not blocks:
        return "No included papers yet. Screen papers and fetch full texts first.", []
    rq = "\n".join(f"- {q}" for q in project["plan"].get("research_questions", []))
    prompt = f"Topic: {project['idea']}\n\nResearch questions:\n{rq}\n\nPapers:\n\n" + "\n\n".join(blocks)
    text = llm.complete(REPORT_PROMPT, [{"role": "user", "content": prompt}])
    m = re.fullmatch(r"\s*```(?:markdown|md)?\s*\n(.*?)\n?```\s*", text, re.S)
    text = m.group(1) if m else text
    text = re.sub(r"\n#+\s*References\s*\n.*$", "", text.strip(), flags=re.S | re.I)  # we add our own list
    return text.strip(), [c["doc_id"] for c in included]


# ---- reporting good practice: PRISMA flow, methods section, protocol ---------------------------------

def prisma_counts(library, project_id: int) -> dict:
    project = library.get_project(project_id)
    cands = library.candidates(project_id)
    stats = project["options"].get("search_stats", {})
    per_db = {k: v.get("found", 0) for k, v in stats.items() if not k.startswith("_") and isinstance(v, dict)}
    imported = sum(1 for c in cands if c["found_in"].startswith("Import"))
    snowballed = sum(1 for c in cands if c["found_in"].startswith("Snowballing"))
    from_search = len(cands) - snowballed
    identified = sum(per_db.values()) + imported + snowballed
    screened = [c for c in cands if c["decision"] != "pending"]
    included = [c for c in cands if c["decision"] == "include"]
    return {
        "per_db": per_db, "imported": imported, "identified": identified,
        "duplicates": max(0, sum(per_db.values()) + imported - from_search), "records_screened": len(screened),
        "not_screened": sum(1 for c in cands if c["decision"] == "pending"),
        "excluded": sum(1 for c in screened if c["decision"] == "exclude"),
        "maybe": sum(1 for c in screened if c["decision"] == "maybe"),
        "snowballed": snowballed,
        "sought": len(included), "not_retrieved": sum(1 for c in included if c["fulltext"] == "abstract"),
        "full_text": sum(1 for c in included if c["fulltext"] == "pdf"), "included": len(included),
        "retracted": sum(1 for c in included if c["meta"].get("retracted")),
    }


def prisma_svg(counts: dict) -> str:
    """PRISMA 2020-style flow diagram as standalone SVG."""
    import html
    db_lines = [f"{k.split(' (')[0]} (n = {v})" for k, v in counts["per_db"].items()]
    if counts["imported"]:
        db_lines.append(f"Imported exports (n = {counts['imported']})")
    if counts["snowballed"]:
        db_lines.append(f"Citation searching (n = {counts['snowballed']})")
    boxes = [
        ("Identification", f"Records identified (n = {counts['identified']})", db_lines,
         f"Duplicates removed (n = {counts['duplicates']})", []),
        ("Screening", f"Records screened by title/abstract (n = {counts['records_screened']})",
         [f"Not yet screened (n = {counts['not_screened']})"] if counts["not_screened"] else [],
         f"Records excluded (n = {counts['excluded']})",
         [f"Uncertain / maybe (n = {counts['maybe']})"] if counts["maybe"] else []),
        ("Retrieval", f"Reports sought for retrieval (n = {counts['sought']})", [],
         f"Full text not retrieved (n = {counts['not_retrieved']})", ["(assessed on abstract)"]),
        ("Included", f"Studies included in review (n = {counts['included']})",
         [f"with full text (n = {counts['full_text']})"] +
         ([f"⚠ retracted (n = {counts['retracted']})"] if counts["retracted"] else []), "", []),
    ]
    w, row_h, y0 = 900, 130, 20
    parts = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{y0 + row_h * len(boxes) + 20}" '
             f'font-family="Helvetica, Arial, sans-serif" font-size="13">',
             '<rect width="100%" height="100%" fill="white"/>',
             '<defs><marker id="a" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">'
             '<polygon points="0 0, 10 3.5, 0 7" fill="#333"/></marker></defs>']

    def box(x, y, bw, bh, title, lines, fill):
        parts.append(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="6" fill="{fill}" stroke="#333"/>')
        parts.append(f'<text x="{x + 10}" y="{y + 22}" font-weight="bold">{html.escape(title)}</text>')
        for i, ln in enumerate(lines[:5]):
            parts.append(f'<text x="{x + 16}" y="{y + 42 + i * 16}" font-size="12">{html.escape(ln)}</text>')

    for i, (stage, main, sub, side, side_sub) in enumerate(boxes):
        y = y0 + i * row_h
        parts.append(f'<rect x="10" y="{y}" width="30" height="{row_h - 20}" fill="#c7d2fe" stroke="#333"/>')
        parts.append(f'<text transform="translate(30,{y + (row_h - 20) / 2}) rotate(-90)" text-anchor="middle" '
                     f'font-weight="bold" font-size="12">{stage}</text>')
        box(55, y, 430, row_h - 20, main, sub, "#eef2ff")
        if side:
            box(545, y, 340, row_h - 20, side, side_sub, "#f9fafb")
            parts.append(f'<line x1="485" y1="{y + 45}" x2="545" y2="{y + 45}" stroke="#333" marker-end="url(#a)"/>')
        if i < len(boxes) - 1:
            parts.append(f'<line x1="270" y1="{y + row_h - 20}" x2="270" y2="{y + row_h}" stroke="#333" marker-end="url(#a)"/>')
    parts.append("</svg>")
    return "\n".join(parts)


def methods_section(library, project_id: int, screener: str = "a local language model") -> str:
    """Methods text generated from what was actually done (search log, criteria, counts)."""
    project = library.get_project(project_id)
    plan, opts = project["plan"], project["options"]
    stats = opts.get("search_stats", {})
    c = prisma_counts(library, project_id)
    dbs = ", ".join(k.split(" (")[0] for k in c["per_db"]) or "the selected databases"
    years = opts.get("year_from"), opts.get("year_to")
    inc = "; ".join(plan.get("inclusion_criteria", [])) or "relevance to the research question"
    exc = "; ".join(plan.get("exclusion_criteria", [])) or "off-topic studies"
    snow = opts.get("snowball_log", [])
    rq = "\n".join(f"{i}. {q}" for i, q in enumerate(plan.get("research_questions", []), 1))
    cols = []
    tables = library.list_tables(project_id)
    if tables:
        cols = [x["name"] for x in library.get_table(tables[0]["id"])["columns"]]
    text = f"""## Methods

### Research questions

{rq or project["idea"]}

### Search strategy

We searched {dbs} on {stats.get("_searched_at", "the search date")}, limited to publications from {years[0] or "any year"} to {years[1] or "the present"}. The search combined the key concepts with Boolean operators (synonyms joined by OR, concepts joined by AND):

`{stats.get("_query", "")}`

Database-specific versions of this string are provided in the supplementary material. {"Additional records were imported from database exports (n = " + str(c["imported"]) + ")." if c["imported"] else ""}

### Eligibility criteria

Studies were included if they met the following criteria: {inc}. Studies were excluded for the following reasons: {exc}.

### Study selection

Records were de-duplicated by DOI, arXiv identifier and normalised title. Of {c["identified"]} records identified, {c["duplicates"]} duplicates were removed. Records were ranked by semantic relevance to the research question (bi-encoder embeddings followed by a cross-encoder re-ranker), and the top {c["records_screened"]} were screened on title and abstract by {screener} against the eligibility criteria, with every decision and its rationale logged; uncertain records were marked for manual review. {c["excluded"]} records were excluded at this stage and {c["included"]} studies were included.{" Backward and forward citation searching (snowballing) of included studies added " + str(c["snowballed"]) + " records." if snow else ""}

### Data collection

Full texts were retrieved from open-access sources (publisher sites, arXiv, PubMed Central and institutional repositories); {c["not_retrieved"]} included studies were assessed on their abstract only because no open-access full text was available. Data were extracted into a structured table{(" with the fields: " + ", ".join(cols)) if cols else ""}. Each extracted value is linked to the supporting passage and page in the source document.{" " + str(c["retracted"]) + " included studies were flagged as retracted." if c["retracted"] else ""}

### Limitations of the review process

Screening and extraction were assisted by a language model and should be verified by a human reviewer; only open-access full texts were analysed.
"""
    return text


def protocol_markdown(library, project_id: int) -> str:
    """Search protocol / log for reproducibility: question, criteria, strings per database, counts."""
    project = library.get_project(project_id)
    plan, opts = project["plan"], project["options"]
    stats = opts.get("search_stats", {})
    c = prisma_counts(library, project_id)
    lines = [f"# Search protocol: {project['title']}", "", f"**Research idea:** {project['idea']}", ""]
    if plan.get("research_questions"):
        lines += ["## Research questions", ""] + [f"{i}. {q}" for i, q in enumerate(plan["research_questions"], 1)] + [""]
    lines += ["## Concepts and synonyms", ""] + [f"- **{x['name']}**: {', '.join(x['synonyms']) or '—'}"
                                                  for x in plan.get("concepts", [])] + [""]
    lines += ["## Eligibility criteria", "", "**Include:**"] + [f"- {x}" for x in plan.get("inclusion_criteria", [])]
    lines += ["", "**Exclude:**"] + [f"- {x}" for x in plan.get("exclusion_criteria", [])] + [""]
    lines += ["## Search log", "", f"- Date: {stats.get('_searched_at', 'n/a')}",
              f"- Years: {opts.get('year_from') or 'any'}–{opts.get('year_to') or 'present'}",
              f"- Results per source: {opts.get('per_source')}", ""]
    lines += ["| Source | Records |", "|---|---|"] + [f"| {k} | {v} |" for k, v in c["per_db"].items()] + [""]
    errors = [f"- {k}: {v['error']}" for k, v in stats.items() if isinstance(v, dict) and v.get("error")]
    if errors:
        lines += ["**Errors:**"] + errors + [""]
    lines += ["## Search strings", ""]
    for name, q in sources.database_strings(plan.get("concepts", []), opts.get("year_from"), opts.get("year_to")).items():
        lines += [f"**{name}**", "", "```", q, "```", ""]
    for s in opts.get("snowball_log", []):
        lines.append(f"- Snowballing on {s['date']}: from {s['from']} included papers, +{s['backward']} references, "
                     f"+{s['forward']} citing papers")
    lines += ["", "## Counts (PRISMA)", ""] + [f"- {k.replace('_', ' ')}: {v}" for k, v in c.items() if k != "per_db"]
    return "\n".join(lines) + "\n"


def screening_log_csv(library, project_id: int) -> str:
    import csv, io
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["Title", "Authors", "Year", "Venue", "DOI", "Found in", "Relevance", "Decision", "Decided by",
                "Reason", "Full text", "Retracted", "URL", "Abstract"])
    for c in library.candidates(project_id):
        m = c["meta"]
        w.writerow([m.get("title"), "; ".join(m.get("authors", [])), m.get("year") or "", m.get("venue", ""),
                    m.get("doi", ""), c["found_in"], c["relevance"], c["decision"], c.get("decided_by", ""), c["reason"],
                    c["fulltext"], "yes" if m.get("retracted") else "", m.get("url", ""), m.get("abstract", "")])
    return buf.getvalue()

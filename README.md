# Research Assistant

A free, local alternative to Anara: chat with your papers and get answers that cite the paper and page.

## Start it

```bash
./run.sh          # macOS / Linux
run.bat           # Windows
```

Then open http://localhost:8501. Make sure the **Ollama** app is running (menu-bar llama icon) if you use the local model.

## Pages

| Page | What it does |
|---|---|
| 🔬 **Research agent** | Describe an idea. The agent writes a search plan (research questions, concepts + synonyms, inclusion/exclusion criteria), builds search strings for **Scopus, IEEE Xplore, ScienceDirect, Web of Science, PubMed, arXiv**, searches **IEEE, ACM, Elsevier/ScienceDirect, Springer, Wiley, T&F, MDPI** (via OpenAlex), **arXiv, PubMed** and optional key-based APIs, removes duplicates, ranks by relevance, **AI-screens abstracts**, downloads **open-access PDFs** (publisher, arXiv, PubMed Central, repositories), builds a **cited extraction table** and writes a **literature review**. Shows PRISMA-style counts. You can also import RIS/BibTeX/CSV exports from Scopus, IEEE Xplore, ACM, ScienceDirect, or Web of Science. Runs in the background. |
| 📊 **Tables** | Any set of papers → table (one row per paper, one column per question), every cell linked to the quote and page. Templates (CS/engineering, PICO, social science), CSV/Excel/Markdown export. |
| 💬 **Chat** | Ask questions across all papers (or only selected ones). Answers cite sources `[1]`; open **Sources** to preview the exact page or jump to it in the Reader. Chats are saved in the sidebar. **Paper tools**: Summarize, Key findings, Methods, Limitations, Explain simply, Key terms; with 2+ papers: Compare (table), Literature review, Agree / disagree, Research gaps. |
| 📖 **Reader** | Read a paper page by page with a chat beside it that only uses that paper. *Explain this page* / *Summarize this page*. Details tab has abstract, links and BibTeX. Remembers your last page. |
| 📝 **Notes** | Your own notes plus answers saved with **📝 Save to notes**. Search, edit, export to Markdown. |
| 📚 **Library** | Upload PDFs, filter, tag, mark To read / Reading / Read, edit details, export BibTeX for your reference manager. |
| 🔎 **Discover** | Search OpenAlex (250M+ papers) or arXiv and add open-access PDFs in one click, or add by DOI / arXiv ID / PDF link. |

Authors, year, venue, DOI and abstract are filled in automatically when a paper has a DOI or arXiv ID.

## AI models

- **Ollama (local)**: private, free, unlimited. On this Mac (CPU only) expect roughly 1–3 minutes per answer. A full automatic research run takes about 20–40 minutes, but it runs in the background.
  - `qwen2.5:3b`: faster
  - `qwen2.5:7b` (default): better answers and citations, slower
  - Install others with `ollama pull <model>`; they appear in the sidebar automatically.
- **Gemini**: much faster, needs a free key from https://aistudio.google.com/apikey (put it in `.env` or the sidebar). The free tier has daily limits and Google may use your inputs, so use Ollama for unpublished work. If you hit the limit, choose `gemini-2.5-flash-lite`.

## Exports

| From | Formats | Options |
|---|---|---|
| Whole chat, single answer, reader chat, notes, literature review, methods section | Markdown, **Word**, **PDF**, **LaTeX** (single Overleaf-ready .tex with embedded bibliography), HTML, plain text | Citations: numbered `[1]`, author–year `(Smith et al., 2020)`, or **none**; reference style: APA 7, IEEE, Harvard, MLA 9, Chicago, Vancouver |
| Extraction tables | Excel (with *Evidence* and *References* sheets), CSV, LaTeX longtable, Word, PDF, Markdown, HTML, JSON | With or without citations and evidence |
| Library / included papers | BibTeX, RIS (EndNote, Zotero), CSV, formatted reference list (Word, Markdown, PDF, text) | Any reference style |
| Research project | **One ZIP** with search protocol, PRISMA diagram, screening log, references, table, methods and review | Reference style for the bundle |

## Research good practice built in

- **Reproducible search log:** date, exact Boolean string, per-database strings and per-source counts (search protocol export).
- **PRISMA 2020 flow diagram:** generated from your project (SVG).
- **Screening audit trail:** every record's decision, who decided (AI model or you) and why (screening log CSV).
- **Citation snowballing:** backward (references) and forward (cited-by) searching from included papers.
- **Retraction check:** retracted papers are flagged so you don't cite them.
- **Quality appraisal:** CASP-style table template (aims, design, bias, reproducibility, overall quality).
- **Auto-written Methods section:** generated from what was actually done, not invented.
- **Evidence for every table cell:** quote and page, exportable.

## About paywalls

The agent only downloads **legal open-access** copies. Paywalled papers, and sites that block automated downloads (ACM, Springer, MDPI), are added as **abstract only**. The Papers tab gives you a link for each one. Download the PDF in your browser (with your university login if needed) and attach it. Anara works the same way.

## Optional API keys (Settings page)

IEEE Xplore, Elsevier (Scopus + ScienceDirect), Springer Nature and CORE offer free keys that add direct search of those databases. OpenAlex and Semantic Scholar keys avoid rate limits. Keys are stored only in `data/settings.json`.

## Tips

- Scanned PDFs have no text. Run `ocrmypdf in.pdf out.pdf` first.
- All your data lives in `data/` (library database, PDFs, models). Back up that folder to keep your library.
- Settings (chunk size, number of sources, models) are in `.env`. Changing `EMBED_MODEL` requires deleting `data/library.db` and re-adding papers.

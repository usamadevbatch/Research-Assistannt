"""PDF parsing and page-aware chunking."""
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

MIN_CHUNK_CHARS = 50
_HYPHEN_BREAK = re.compile(r"(\w)-\n(\w)")
_WHITESPACE = re.compile(r"\s+")


@dataclass
class ParsedPDF:
    title: str
    num_pages: int
    pages: list[str]  # cleaned text; index 0 is page 1


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def clean_text(text: str) -> str:
    text = _HYPHEN_BREAK.sub(r"\1\2", text)  # re-join words split across lines
    return _WHITESPACE.sub(" ", text).strip()


def _looks_like_title(s: str) -> bool:
    low = s.lower()
    return (
        len(s) > 5
        and not low.startswith(("microsoft word", "untitled"))
        and not low.endswith((".pdf", ".doc", ".docx", ".tex"))
    )


def _first_page_title(doc) -> str:
    """Many papers have no title metadata; the largest text on page 1 is usually the title."""
    try:
        spans = [
            (round(span["size"], 1), span["text"].strip())
            for block in doc[0].get_text("dict")["blocks"]
            for line in block.get("lines", [])
            # Skip rotated text such as the arXiv identifier stamped in the margin.
            if abs(line["dir"][1]) < 0.1
            for span in line["spans"]
            if span["text"].strip() and not span["text"].strip().lower().startswith("arxiv:")
        ]
    except Exception:
        return ""
    if not spans:
        return ""
    biggest = max(size for size, _ in spans)
    title = " ".join(text for size, text in spans if size == biggest)
    return title if 8 <= len(title) <= 250 else ""


def parse_pdf(data: bytes, filename: str) -> ParsedPDF:
    with pymupdf.open(stream=data, filetype="pdf") as doc:
        meta_title = ((doc.metadata or {}).get("title") or "").strip()
        pages = [clean_text(page.get_text("text")) for page in doc]
        if _looks_like_title(meta_title):
            title = meta_title
        else:
            title = _first_page_title(doc) or Path(filename).stem
    return ParsedPDF(title=title, num_pages=len(pages), pages=pages)


def render_page(pdf_path: Path, page: int, dpi: int = 110) -> bytes:
    with pymupdf.open(pdf_path) as doc:
        return doc[page - 1].get_pixmap(dpi=dpi).tobytes("png")


def _split(text: str, size: int, overlap: int) -> list[str]:
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # Prefer ending on a sentence boundary in the second half of the window.
            cut = text.rfind(". ", start + size // 2, end)
            if cut != -1:
                end = cut + 1
        chunks.append(text[start:end].strip())
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def chunk_pages(pages: list[str], size: int, overlap: int) -> list[tuple[int, str]]:
    """Chunk within page boundaries so every chunk has an exact page citation."""
    out = []
    for page_no, text in enumerate(pages, start=1):
        for chunk in _split(text, size, overlap):
            if len(chunk) >= MIN_CHUNK_CHARS:
                out.append((page_no, chunk))
    return out

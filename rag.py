"""Prompt construction, follow-up question condensing, and paper-level tasks."""
from llm import LLM
from retrieval import Hit

MAX_TURNS = 3

SYSTEM_PROMPT = """You are a careful research assistant helping a researcher understand their own library of papers.

Rules:
1. Answer ONLY from the numbered sources in the latest message. Do not use outside knowledge for factual claims.
2. Cite every factual claim inline, right after the claim, with its source number in square brackets, e.g. [2] or [1][4]. Never put the citations in a list at the end. Source numbers refer only to the sources in the latest message.
3. If the sources do not contain the answer, say so clearly and suggest what to look for instead. Never invent findings, numbers, or citations.
4. When sources disagree, point out the disagreement and cite both sides.
5. Be precise about numbers, methods, sample sizes, and limitations. Use Markdown (headings, bullet points, tables) when it helps, but never wrap the whole answer in a code block. Reply in the same language as the question."""

CONDENSE_PROMPT = """Rewrite the user's latest question as one standalone search query that makes sense without the conversation. Resolve pronouns and references like "it" or "the second paper". Return only the query text."""

# One-click tasks that run over a whole paper rather than a search.
PAPER_TASKS = {
    "Summarize": (
        "Write a structured summary of this paper with these headings: "
        "**Research question**, **Methods** (data, sample size, design), **Key findings** (with numbers), "
        "**Limitations**, **Why it matters**. Cite source numbers for every claim."
    ),
    "Key findings": (
        "List the paper's main findings as bullet points, each with the exact numbers, effect sizes, "
        "or statistics reported, and a source citation."
    ),
    "Methods": (
        "Explain the paper's methodology step by step: data and participants, study design, "
        "models/algorithms, evaluation metrics, and baselines. Cite sources."
    ),
    "Limitations": (
        "What limitations do the authors acknowledge, and what weaknesses can you identify from the text? "
        "What future work do they propose? Cite sources."
    ),
    "Explain simply": (
        "Explain this paper to a smart undergraduate with no background in the field: what problem it tackles, "
        "the core idea, and the result. Define any jargon. Cite sources."
    ),
    "Key terms": (
        "Make a glossary of the 8-12 most important technical terms, acronyms, and symbols in this paper. "
        "For each: the term in bold, a one-sentence plain definition as used in the paper, and a citation."
    ),
}

# Tasks that need two or more papers.
MULTI_TASKS = {
    "Compare": (
        "Compare these papers in a Markdown table with one row per paper and columns: "
        "Paper, Research question, Method/approach, Data, Key results, Limitations. "
        "Then write 3-5 bullet points on the most important similarities and differences. Cite sources."
    ),
    "Literature review": (
        "Write a short literature review synthesizing these papers. Organize it by theme, not paper by paper. "
        "Explain how the works relate, build on, or contradict each other, and end with open questions. "
        "Cite sources for every claim."
    ),
    "Agree / disagree": (
        "Where do these papers agree, and where do they disagree or report conflicting results? "
        "Use two sections, **Agreements** and **Disagreements**, and cite both sides of every point."
    ),
    "Research gaps": (
        "Based only on these papers, identify research gaps: limitations they share, questions none of them "
        "answers, and promising directions for new work. Propose 3-5 concrete research ideas, each tied to the "
        "gap it addresses. Cite sources."
    ),
}

PAGE_TASKS = {
    "Explain this page": (
        "Explain the content of page {page} in clear, plain language: the main point, how it fits into the "
        "paper, and any equations, figures, or tables mentioned on it. Define jargon. Cite sources."
    ),
    "Summarize this page": "Summarize page {page} in 3-6 bullet points. Cite sources.",
}

SUGGESTED_QUESTIONS = [
    "What are the main findings across my papers?",
    "Which methods and datasets are used?",
    "What limitations do the authors mention?",
    "What future work is suggested?",
]


def format_context(hits: list[Hit], titles: dict[int, str]) -> str:
    return "\n\n".join(
        f"[{i}] {titles.get(h.doc_id, 'Unknown')} (p. {h.page})\n{h.text}"
        for i, h in enumerate(hits, 1)
    )


def build_messages(history, question, hits, titles) -> list[dict]:
    msgs = [{"role": m["role"], "content": m["content"]} for m in history[-2 * MAX_TURNS:]]
    msgs.append({
        "role": "user",
        "content": f"Sources:\n\n{format_context(hits, titles)}\n\nQuestion: {question}",
    })
    return msgs


def condense_question(llm: LLM, history: list[dict], question: str) -> str:
    if not history:
        return question
    transcript = "\n".join(
        f"{m['role'].upper()}: {m['content'][:1500]}" for m in history[-2 * MAX_TURNS:]
    )
    try:
        rewritten = llm.complete(
            CONDENSE_PROMPT,
            [{"role": "user", "content": f"Conversation:\n{transcript}\n\nLatest question: {question}"}],
        ).strip()
        return rewritten if 0 < len(rewritten) < 500 else question
    except Exception:
        return question  # retrieval still works with the raw question


def paper_hits(doc_id: int, chunks: list[tuple[int, str]], budget_chars: int) -> list[Hit]:
    """Pick chunks spread evenly across the paper so they fit in the model's context window.

    The opening (abstract/intro) and the end (conclusion) are always included.
    """
    if not chunks:
        return []
    per_chunk = max(1, sum(len(t) for _, t in chunks) // len(chunks))
    n = max(2, budget_chars // per_chunk)
    if n >= len(chunks):
        picked = range(len(chunks))
    else:
        head = min(4, n // 3)
        tail = min(2, n // 6)
        middle = n - head - tail
        span = range(head, len(chunks) - tail)
        step = len(span) / max(1, middle)
        picked = sorted(
            set(range(head))
            | {span[int(i * step)] for i in range(middle)}
            | set(range(len(chunks) - tail, len(chunks)))
        )
    return [Hit(doc_id, chunks[i][0], chunks[i][1], 1.0) for i in picked]


def multi_paper_hits(doc_chunks: dict[int, list[tuple[int, str]]], budget_chars: int) -> list[Hit]:
    """Split the context budget evenly across papers."""
    per_doc = budget_chars // max(1, len(doc_chunks))
    return [h for doc_id, chunks in doc_chunks.items() for h in paper_hits(doc_id, chunks, per_doc)]


def page_hits(doc_id: int, chunks: list[tuple[int, str]], page: int) -> list[Hit]:
    """Chunks on the page itself, plus its neighbours for context."""
    on_page = [Hit(doc_id, p, t, 1.0) for p, t in chunks if p == page]
    around = [Hit(doc_id, p, t, 0.5) for p, t in chunks if abs(p - page) == 1]
    return on_page + around[:2]

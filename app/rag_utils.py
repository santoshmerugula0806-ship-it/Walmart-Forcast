"""
RAG (Retrieval-Augmented Generation) over the policy knowledge base.

Documents -> split into chunks -> indexed with BM25 (a strong, dependency-light
retriever — no embedding model/API required, so this works fully offline) ->
retriever -> relevant chunks -> answer.

Answer generation has two modes:
  - If ANTHROPIC_API_KEY is set: the retrieved chunks + question are sent to
    Claude, which is instructed to answer ONLY from the provided context.
  - Otherwise: a deterministic template stitches the most relevant chunks
    together with light framing. No hallucination risk either way, since the
    LLM (when used) is grounded strictly in retrieved text.
"""
import os
import re
import glob

from rank_bm25 import BM25Okapi

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KB_DIR = os.path.join(BASE_DIR, "data", "knowledge_base")

_chunks = []          # list of {"doc": filename, "heading": str, "text": str}
_bm25 = None
_tokenized_corpus = None


def _tokenize(text):
    return re.findall(r"[a-z0-9%]+", text.lower())


def _split_into_chunks(doc_name, text):
    """Split a markdown doc into chunks along ## headings."""
    sections = re.split(r"\n(?=## )", text)
    out = []
    for sec in sections:
        sec = sec.strip()
        if not sec:
            continue
        heading_match = re.match(r"##?\s*(.+)", sec)
        heading = heading_match.group(1).strip() if heading_match else doc_name
        out.append({"doc": doc_name, "heading": heading, "text": sec})
    return out


def build_index():
    global _chunks, _bm25, _tokenized_corpus
    _chunks = []
    for path in sorted(glob.glob(os.path.join(KB_DIR, "*.md"))):
        with open(path, encoding="utf-8") as f:
            text = f.read()
        doc_name = os.path.basename(path)
        _chunks.extend(_split_into_chunks(doc_name, text))

    _tokenized_corpus = [_tokenize(c["text"]) for c in _chunks]
    _bm25 = BM25Okapi(_tokenized_corpus) if _tokenized_corpus else None


def retrieve(query, k=4):
    if _bm25 is None:
        build_index()
    if not _chunks:
        return []
    scores = _bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(scores, _chunks), key=lambda x: x[0], reverse=True)
    results = [{"score": round(float(s), 3), **c} for s, c in ranked[:k] if s > 0]
    return results


def _template_answer(question, retrieved):
    if not retrieved:
        return ("I couldn't find anything relevant to that in the inventory policy, "
                "supplier rules, store guidelines, stock thresholds, holiday calendar, "
                "or product information documents. Try rephrasing, or ask about "
                "reorder rules, safety stock, lead times, or holiday planning.")
    lines = [f"Based on **{r['doc']}** — *{r['heading']}*:\n{r['text'].split(chr(10), 1)[-1].strip()[:500]}"
             for r in retrieved[:2]]
    return "\n\n".join(lines)


def _claude_answer(question, retrieved, extra_context=None):
    try:
        import anthropic
    except ImportError:
        return None
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None

    context = "\n\n---\n\n".join(f"[{r['doc']} — {r['heading']}]\n{r['text']}" for r in retrieved)
    if extra_context:
        context += "\n\n---\n\n[Live system data]\n" + extra_context

    system = (
        "You are the Walmart AI Assistant embedded in a demand-forecasting and "
        "inventory web app. Answer the manager's question using ONLY the context "
        "provided below (retrieved policy documents and, when present, live "
        "system data). If the context doesn't contain the answer, say so plainly "
        "rather than guessing. Be concise and concrete — managers are reading "
        "this on a dashboard, not a report."
    )
    try:
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=system,
            messages=[{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        )
        return "".join(b.text for b in resp.content if b.type == "text")
    except Exception as e:
        return None


def answer_question(question, extra_context=None, k=4):
    retrieved = retrieve(question, k=k)
    claude_ans = _claude_answer(question, retrieved, extra_context=extra_context)
    answer = claude_ans if claude_ans else _template_answer(question, retrieved)
    return {
        "question": question,
        "answer": answer,
        "sources": [{"doc": r["doc"], "heading": r["heading"], "score": r["score"]} for r in retrieved],
        "generated_by": "claude" if claude_ans else "template",
    }

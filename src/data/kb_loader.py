"""Load a knowledge base from disk and split it into chunks.

A KB is a directory of .md / .txt files (searched recursively). Each file is a
document whose id is its path relative to the KB root, without extension.
Chunks are paragraph-based: short paragraphs are merged forward, and long ones
are split on sentence boundaries, so chunks stay well under the 256-token
limit of all-MiniLM-L6-v2.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

KB_EXTENSIONS = {".md", ".txt"}


@dataclass(frozen=True)
class KBDocument:
    doc_id: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str


def load_documents(kb_path: str | Path) -> List[KBDocument]:
    root = Path(kb_path)
    if not root.exists():
        raise FileNotFoundError(f"KB path does not exist: {root}")
    # Allow pointing at the synthetic KB root, which keeps documents in docs/.
    if (root / "docs").is_dir():
        root = root / "docs"
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in KB_EXTENSIONS)
    if not files:
        raise ValueError(f"No {sorted(KB_EXTENSIONS)} files found under {root}")

    docs = []
    for path in files:
        raw = path.read_text(encoding="utf-8").strip()
        title = path.stem
        lines = raw.splitlines()
        if lines and lines[0].startswith("#"):
            title = lines[0].lstrip("#").strip()
            raw = "\n".join(lines[1:]).strip()
        doc_id = str(path.relative_to(root).with_suffix(""))
        docs.append(KBDocument(doc_id=doc_id, title=title, text=raw))
    return docs


_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _split_long(paragraph: str, max_words: int) -> List[str]:
    pieces, current, count = [], [], 0
    for sentence in _SENTENCE_SPLIT.split(paragraph):
        n = len(sentence.split())
        if current and count + n > max_words:
            pieces.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += n
    if current:
        pieces.append(" ".join(current))
    return pieces


def chunk_document(doc: KBDocument, min_words: int = 40, max_words: int = 160) -> List[Chunk]:
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", doc.text) if p.strip()]
    merged: List[str] = []
    buffer = ""
    for para in paragraphs:
        buffer = f"{buffer} {para}".strip() if buffer else para
        if len(buffer.split()) >= min_words:
            merged.append(buffer)
            buffer = ""
    if buffer:
        if merged and len(merged[-1].split()) + len(buffer.split()) <= max_words:
            merged[-1] = f"{merged[-1]} {buffer}"
        else:
            merged.append(buffer)

    texts: List[str] = []
    for block in merged:
        texts.extend(_split_long(block, max_words) if len(block.split()) > max_words else [block])
    return [Chunk(chunk_id=f"{doc.doc_id}#{i}", doc_id=doc.doc_id, text=t) for i, t in enumerate(texts)]


def load_chunks(kb_path: str | Path, min_words: int = 40, max_words: int = 160) -> List[Chunk]:
    chunks: List[Chunk] = []
    for doc in load_documents(kb_path):
        chunks.extend(chunk_document(doc, min_words=min_words, max_words=max_words))
    return chunks


def kb_fingerprint(docs: List[KBDocument]) -> str:
    h = hashlib.sha256()
    for doc in sorted(docs, key=lambda d: d.doc_id):
        h.update(doc.doc_id.encode())
        h.update(b"\0")
        h.update(doc.text.encode())
        h.update(b"\0")
    return h.hexdigest()

"""Splits documents into chunks. Each chunk keeps its own text and a copy of the document's metadata."""
from dataclasses import dataclass

from rag.loader import Doc


@dataclass
class Chunk:
    id: str            # "<doc id>#<number>", e.g. "of-210-city#2"
    doc_id: str
    text: str          # the original text, exactly as it will be shown to the model
    meta: dict


def chunk_fixed(doc: Doc, size: int = 350, overlap: int = 50) -> list[Chunk]:
    """Baseline: cut every `size` characters, with `overlap` characters repeated between neighbours."""
    chunks = []
    step = size - overlap
    for start in range(0, max(len(doc.text), 1), step):
        piece = doc.text[start:start + size]
        if piece.strip():
            chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, piece, dict(doc.meta)))
        if start + size >= len(doc.text):
            break
    return chunks


def chunk_by_headings(doc: Doc) -> list[Chunk]:
    """One chunk per "## " section, each starting with a "<title> › <section>" line."""
    sections = []
    heading, lines = None, []
    for line in doc.text.splitlines():
        if line.startswith("## "):
            sections.append((heading, lines))
            heading, lines = line[3:].strip(), []
        elif not line.startswith("# "):
            lines.append(line)
    sections.append((heading, lines))

    chunks = []
    for heading, lines in sections:
        body = "\n".join(lines).strip()
        if heading is None and not body and len(sections) > 1:
            continue
        meta = dict(doc.meta)
        if heading is None:
            path = doc.title
        else:
            path = f"{doc.title} › {heading}"
            meta["section"] = heading
        text = f"{path}\n{body}" if body else path
        chunks.append(Chunk(f"{doc.id}#{len(chunks)}", doc.id, text, meta))
    return chunks

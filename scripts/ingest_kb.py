"""Ingest .md/.txt files into the pgvector knowledge base.

Usage (inside the api container or a configured virtualenv)::

    python -m scripts.ingest_kb kb_docs/

Re-running is safe: chunk ids are content-derived, so unchanged chunks are
upserted in place instead of duplicated. (Chunks of a document you *edit* get
new ids; delete stale ones with ``--replace`` which clears the collection first.)
"""

import argparse
import asyncio
import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from agent.rag import close_vector_store, get_vector_store, ingest_documents, init_vector_store
from core.config import get_settings

logger = logging.getLogger("ingest_kb")

SUPPORTED_SUFFIXES = {".md", ".txt"}


def load_chunks(root: Path, chunk_size: int = 800, chunk_overlap: int = 100) -> list[Document]:
    """Read every supported file under ``root`` and split it into chunks."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: list[Document] = []
    for path in sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_SUFFIXES):
        text = path.read_text(encoding="utf-8")
        source = path.relative_to(root).as_posix()
        doc = Document(page_content=text, metadata={"source": source, "title": path.stem})
        chunks.extend(splitter.split_documents([doc]))
    return chunks


async def main(root: Path, replace: bool) -> int:
    settings = get_settings()
    init_vector_store(settings)
    try:
        if replace:
            await get_vector_store().adelete_collection()
            init_vector_store(settings)
        chunks = load_chunks(root)
        if not chunks:
            logger.warning("No .md/.txt files found under %s", root)
            return 0
        count = await ingest_documents(chunks)
        logger.info("Ingested %d chunks from %s", count, root)
        return count
    finally:
        await close_vector_store()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--replace", action="store_true", help="clear the collection first")
    args = parser.parse_args()
    asyncio.run(main(args.directory, args.replace))

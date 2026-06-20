"""RAG knowledge base implementation.

Users provide their own embedding API key. We store vectors locally.
"""

import hashlib
import json
from pathlib import Path
from typing import List

from ..config import get_config
from ..llm_client import LLMClient, LLMError


def cmd_rag(args):
    """RAG knowledge base operations."""
    action = args.action

    if action == "status":
        return _cmd_rag_status()
    elif action == "index":
        return _cmd_rag_index(args.docs or [])
    elif action == "query":
        return _cmd_rag_query(args.query or "")

    return 0


def _get_db_path() -> Path:
    cfg = get_config()
    path = cfg.get("rag.db_path", "")
    if path:
        return Path(path)
    return Path.home() / ".math-anything" / "rag_db"


def _get_chroma_client():
    try:
        import chromadb
    except ImportError:
        print("Error: chromadb not installed. Run: pip install chromadb")
        return None
    db_path = _get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(db_path))


def _cmd_rag_status():
    client = _get_chroma_client()
    if not client:
        return 1
    try:
        collection = client.get_or_create_collection("materials_science")
        count = collection.count()
        print(f"RAG Database: {_get_db_path()}")
        print(f"Documents indexed: {count}")
    except Exception as e:
        print(f"Error: {e}")
        return 1
    return 0


def _cmd_rag_index(docs: List[str]):
    if not docs:
        print("Error: No documents provided. Use --docs file1.pdf file2.txt")
        return 1

    client = _get_chroma_client()
    if not client:
        return 1

    collection = client.get_or_create_collection("materials_science")

    try:
        llm_client = LLMClient()
    except LLMError as e:
        print(f"LLM Error: {e}")
        return 1

    for doc_path in docs:
        path = Path(doc_path)
        if not path.exists():
            print(f"Skip: {doc_path} not found")
            continue

        print(f"Indexing: {doc_path}...")
        text = _load_document(path)
        if not text:
            continue

        # Chunk text
        chunks = _chunk_text(text, chunk_size=500, overlap=100)
        embeddings = llm_client.embed(chunks)

        ids = [f"{path.name}_{hashlib.md5(c.encode()).hexdigest()[:8]}" for c in chunks]
        metadatas = [{"source": str(path), "chunk": i} for i in range(len(chunks))]

        # Add to collection
        collection.add(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        print(f"  Indexed {len(chunks)} chunks")

    print("Indexing complete.")
    return 0


def _cmd_rag_query(query: str):
    if not query:
        print("Error: No query provided. Use --query 'your question'")
        return 1

    client = _get_chroma_client()
    if not client:
        return 1

    try:
        llm_client = LLMClient()
    except LLMError as e:
        print(f"LLM Error: {e}")
        return 1

    collection = client.get_or_create_collection("materials_science")

    # Embed query
    query_embedding = llm_client.embed([query])[0]

    # Search
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=get_config().get("rag.top_k", 5),
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    if not documents:
        print("No relevant documents found in knowledge base.")
        return 0

    print(f"Query: {query}\n")
    print("Relevant passages:")
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances), 1):
        source = meta.get("source", "unknown") if meta else "unknown"
        print(f"\n[{i}] {source} (distance: {dist:.3f})")
        print(f"    {doc[:300]}...")

    # Optional: ask LLM to synthesize answer
    print("\nSynthesizing answer...")
    context = "\n\n".join(documents[:3])
    prompt = f"""Based on the following context, answer the user's question.
If the context is insufficient, say so.

Context:
{context}

Question: {query}

Answer:"""

    try:
        answer = llm_client.chat(
            [
                {"role": "system", "content": "You are a materials science research assistant."},
                {"role": "user", "content": prompt},
            ]
        )
        print(f"\n{answer}")
    except Exception as e:
        print(f"Synthesis failed: {e}")

    return 0


def _load_document(path: Path) -> str:
    """Load text from various document formats."""
    suffix = path.suffix.lower()
    if suffix == ".txt" or suffix == ".md":
        return path.read_text(encoding="utf-8", errors="replace")
    elif suffix == ".pdf":
        try:
            import pypdf

            reader = pypdf.PdfReader(str(path))
            return "\n".join(page.extract_text() or "" for page in reader.pages)
        except ImportError:
            print("  Warning: pypdf not installed, skipping PDF. Run: pip install pypdf")
            return ""
    elif suffix == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return json.dumps(data, indent=2)
        except Exception:
            return path.read_text(encoding="utf-8", errors="replace")
    else:
        try:
            return path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            print(f"  Warning: Could not read {path}")
            return ""


def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
    """Simple sliding window chunking."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk.strip())
        start = end - overlap
    return chunks

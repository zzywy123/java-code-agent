"""ChromaDB vector store for code chunk embeddings.

Uses ChromaDB's official Python SDK for vector storage and retrieval.
Supports add, query, delete_by_file, and count operations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from agent.config import RAGConfig
from agent.indexing.embedding import EmbeddingService
from agent.models import CodeChunk, SearchResult

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-backed vector store for code chunk embeddings.

    Stores chunk embeddings with metadata for similarity search.
    Uses the embedding service for query embedding.
    """

    def __init__(
        self,
        config: RAGConfig,
        embedding_service: EmbeddingService,
        persist_dir: Path | None = None,
    ) -> None:
        self._config = config
        self._embedding = embedding_service
        self._persist_dir = persist_dir
        self._client: Any = None
        self._collection: Any = None
        self._initialized = False

    def _ensure_initialized(self) -> None:
        """Lazy initialization of ChromaDB client and collection."""
        if self._initialized:
            return

        try:
            import chromadb

            if self._persist_dir:
                self._persist_dir.mkdir(parents=True, exist_ok=True)
                self._client = chromadb.PersistentClient(
                    path=str(self._persist_dir),
                )
            else:
                self._client = chromadb.Client()

            self._collection = self._client.get_or_create_collection(
                name=self._config.chroma_collection,
                metadata={"hnsw:space": "cosine"},
            )
            self._initialized = True
            logger.info(
                "ChromaDB collection '%s' ready, %d vectors",
                self._config.chroma_collection,
                self._collection.count(),
            )
        except ImportError:
            raise RuntimeError(
                "chromadb not installed. Install with: pip install chromadb"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize ChromaDB: {e}")

    def add(self, chunks: list[CodeChunk]) -> int:
        """Add chunks to the vector store.

        Chunks must already have embeddings populated.
        Returns the number of chunks added.
        """
        if not chunks:
            return 0

        self._ensure_initialized()

        ids = []
        embeddings = []
        documents = []
        metadatas = []

        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning("Chunk %s has no embedding, skipping", chunk.chunk_id)
                continue

            ids.append(chunk.chunk_id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.slice.content[:1000])  # Truncate for storage
            metadatas.append({
                "file_path": chunk.slice.file_path,
                "class_name": chunk.slice.class_name,
                "method_name": chunk.slice.method_name,
                "symbol_signature": chunk.slice.symbol_signature,
                "package": chunk.slice.package,
                "start_line": chunk.slice.start_line,
                "end_line": chunk.slice.end_line,
                "file_hash": chunk.file_hash,
            })

        if not ids:
            return 0

        # ChromaDB has a batch limit; split if needed
        batch_size = 5461  # ChromaDB default max batch
        added = 0
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self._collection.upsert(
                ids=ids[i:end],
                embeddings=embeddings[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
            )
            added += end - i

        return added

    def query(
        self,
        query_text: str,
        top_k: int = 10,
        where: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Query the vector store for similar chunks.

        Args:
            query_text: The query text to search for
            top_k: Number of results to return
            where: Optional ChromaDB where filter

        Returns:
            List of SearchResult objects sorted by relevance
        """
        self._ensure_initialized()

        if self._collection.count() == 0:
            return []

        # Embed the query
        query_embedding = self._embedding.embed_query(query_text)
        if not query_embedding:
            return []

        # Query ChromaDB
        query_params: dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": min(top_k, self._collection.count()),
        }
        if where:
            query_params["where"] = where

        try:
            results = self._collection.query(**query_params)
        except Exception as e:
            logger.error("ChromaDB query failed: %s", e)
            return []

        # Parse results
        search_results: list[SearchResult] = []
        if results and results["ids"] and results["ids"][0]:
            for i, chunk_id in enumerate(results["ids"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                # ChromaDB cosine distance = 1 - cosine_similarity
                score = 1.0 - distance

                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                document = results["documents"][0][i] if results["documents"] else ""

                # Reconstruct a minimal CodeChunk for the result
                from agent.models import CodeSlice
                code_slice = CodeSlice(
                    module=metadata.get("module", ""),
                    package=metadata.get("package", ""),
                    class_name=metadata.get("class_name", ""),
                    method_name=metadata.get("method_name", ""),
                    file_path=metadata.get("file_path", ""),
                    start_line=metadata.get("start_line", 0),
                    end_line=metadata.get("end_line", 0),
                    content=document,
                    symbol_signature=metadata.get("symbol_signature", ""),
                )
                chunk = CodeChunk(
                    chunk_id=chunk_id,
                    slice=code_slice,
                    file_hash=metadata.get("file_hash", ""),
                )

                search_results.append(SearchResult(
                    chunk=chunk,
                    score=score,
                    source="vector",
                    rank=i + 1,
                ))

        return search_results

    def delete_by_file(self, file_path: str) -> int:
        """Delete all chunks from a specific file.

        Returns the number of chunks deleted.
        """
        self._ensure_initialized()

        try:
            results = self._collection.get(
                where={"file_path": file_path},
            )
            if results and results["ids"]:
                self._collection.delete(ids=results["ids"])
                return len(results["ids"])
        except Exception as e:
            logger.error("ChromaDB delete failed: %s", e)

        return 0

    def delete_by_ids(self, chunk_ids: list[str]) -> int:
        """Delete chunks by their IDs.

        Returns the number of chunks deleted.
        """
        if not chunk_ids:
            return 0

        self._ensure_initialized()

        try:
            self._collection.delete(ids=chunk_ids)
            return len(chunk_ids)
        except Exception as e:
            logger.error("ChromaDB delete by IDs failed: %s", e)
            return 0

    def count(self) -> int:
        """Return the number of vectors in the store."""
        self._ensure_initialized()
        return self._collection.count()

    def clear(self) -> None:
        """Remove all vectors from the collection."""
        self._ensure_initialized()
        # Delete the collection and recreate it
        self._client.delete_collection(self._config.chroma_collection)
        self._collection = self._client.get_or_create_collection(
            name=self._config.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

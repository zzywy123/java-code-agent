"""BM25 inverted index for code chunk retrieval.

Uses rank_bm25's BM25Okapi algorithm.
Includes a Java-specific tokenizer that handles:
- camelCase splitting (calculateTotal → calculate, Total)
- underscore splitting (my_method → my, method)
- dot splitting (com.example.order → com, example, order)
- Java stop word removal
- Lowercase normalization
"""

from __future__ import annotations

import logging
import re
from typing import Any

from rank_bm25 import BM25Okapi

from agent.models import CodeChunk, SearchResult

logger = logging.getLogger(__name__)

# Java stop words — common keywords that add no retrieval value
_JAVA_STOP_WORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must", "not",
    "and", "or", "but", "if", "else", "for", "while", "do", "switch",
    "case", "break", "continue", "return", "new", "this", "super",
    "class", "interface", "enum", "extends", "implements", "import",
    "package", "public", "private", "protected", "static", "final",
    "abstract", "synchronized", "volatile", "transient", "native",
    "void", "int", "long", "float", "double", "boolean", "char", "byte",
    "short", "string", "null", "true", "false",
    # Common Java API
    "get", "set", "to", "from", "by", "in", "of", "on", "at", "with",
}


def _split_camel_case(identifier: str) -> list[str]:
    """Split a camelCase or PascalCase identifier into tokens.

    Examples:
        calculateTotal → ["calculate", "total"]
        OrderService → ["order", "service"]
        IOException → ["io", "exception"]
        getHTTPResponse → ["get", "http", "response"]
    """
    # Insert boundary before uppercase followed by lowercase
    s1 = re.sub(r"([a-z])([A-Z])", r"\1_\2", identifier)
    # Insert boundary between consecutive uppercase and uppercase+lowercase
    s2 = re.sub(r"([A-Z]+)([A-Z][a-z])", r"\1_\2", s1)
    parts = s2.split("_")
    return [p.lower() for p in parts if p]


def _extract_cjk_chars(text: str) -> list[str]:
    """Extract CJK characters as individual tokens.

    Chinese/Japanese/Korean characters don't have word boundaries,
    so each character is treated as a token for BM25 matching.
    """
    tokens = []
    for ch in text:
        if "一" <= ch <= "鿿" or "㐀" <= ch <= "䶿":
            tokens.append(ch)
    return tokens


def _tokenize_java_text(text: str) -> list[str]:
    """Tokenize Java code text for BM25 indexing.

    Pipeline:
    1. Split on non-alphanumeric characters (preserving CJK)
    2. For each token, split camelCase
    3. Split on underscores and dots
    4. Remove stop words
    5. Normalize to lowercase
    6. Extract CJK characters as individual tokens
    """
    # Split on non-alphanumeric boundaries (but keep CJK chars)
    raw_tokens = re.split(r"[^a-zA-Z0-9_一-鿿㐀-䶿]+", text)

    tokens: list[str] = []
    for raw in raw_tokens:
        if not raw:
            continue
        # Split on underscores
        for underscore_part in raw.split("_"):
            if not underscore_part:
                continue
            # Split on dots
            for dot_part in underscore_part.split("."):
                if not dot_part:
                    continue
                # Split camelCase
                camel_parts = _split_camel_case(dot_part)
                tokens.extend(camel_parts)

    # Extract CJK characters as individual tokens
    cjk_tokens = _extract_cjk_chars(text)
    tokens.extend(cjk_tokens)

    # Remove stop words and very short tokens (but keep single CJK chars)
    filtered = []
    for t in tokens:
        if t in _JAVA_STOP_WORDS:
            continue
        # Keep CJK single characters, skip short Latin tokens
        if len(t) == 1 and "一" <= t <= "鿿":
            filtered.append(t)
        elif len(t) > 1:
            filtered.append(t)
    return filtered


class BM25Index:
    """BM25 inverted index for code chunks.

    Uses Java-specific tokenization for both indexing and querying.
    Rebuilds the underlying BM25Okapi index when chunks are modified.
    """

    def __init__(self) -> None:
        self._chunks: list[CodeChunk] = []
        self._tokenized_corpus: list[list[str]] = []
        self._bm25: BM25Okapi | None = None
        self._dirty = True

    def add(self, chunks: list[CodeChunk]) -> None:
        """Add chunks to the index."""
        for chunk in chunks:
            self._chunks.append(chunk)
            tokens = self._tokenize_chunk(chunk)
            self._tokenized_corpus.append(tokens)
        self._dirty = True

    def remove_by_file(self, file_path: str) -> int:
        """Remove all chunks from a specific file.

        Returns the number of chunks removed.
        """
        indices_to_remove = [
            i for i, c in enumerate(self._chunks)
            if c.slice.file_path == file_path
        ]
        if not indices_to_remove:
            return 0

        # Remove in reverse order to preserve indices
        for i in reversed(indices_to_remove):
            self._chunks.pop(i)
            self._tokenized_corpus.pop(i)

        self._dirty = True
        return len(indices_to_remove)

    def remove_by_ids(self, chunk_ids: set[str]) -> int:
        """Remove chunks by their IDs."""
        indices_to_remove = [
            i for i, c in enumerate(self._chunks)
            if c.chunk_id in chunk_ids
        ]
        for i in reversed(indices_to_remove):
            self._chunks.pop(i)
            self._tokenized_corpus.pop(i)

        self._dirty = True
        return len(indices_to_remove)

    def search(self, query: str, top_k: int = 10) -> list[SearchResult]:
        """Search the index using BM25 scoring.

        Args:
            query: The search query
            top_k: Number of top results to return

        Returns:
            List of SearchResult objects sorted by BM25 score
        """
        if not self._chunks:
            return []

        self._ensure_index()

        query_tokens = _tokenize_java_text(query)
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)

        # Get top-K indices (BM25 can return negative scores for short corpora)
        indexed_scores = list(enumerate(scores))
        indexed_scores.sort(key=lambda x: x[1], reverse=True)

        results: list[SearchResult] = []
        for rank, (idx, score) in enumerate(indexed_scores[:top_k]):
            # Only skip truly zero scores (no overlap at all)
            if score == 0.0 and not any(t in self._tokenized_corpus[idx] for t in query_tokens):
                continue
            results.append(SearchResult(
                chunk=self._chunks[idx],
                score=float(score),
                source="bm25",
                rank=rank + 1,
            ))

        return results

    def count(self) -> int:
        """Return the number of indexed chunks."""
        return len(self._chunks)

    def clear(self) -> None:
        """Remove all chunks from the index."""
        self._chunks.clear()
        self._tokenized_corpus.clear()
        self._bm25 = None
        self._dirty = True

    def _ensure_index(self) -> None:
        """Rebuild the BM25 index if it's dirty."""
        if self._dirty or self._bm25 is None:
            if self._tokenized_corpus:
                self._bm25 = BM25Okapi(self._tokenized_corpus)
            else:
                self._bm25 = None
            self._dirty = False

    def _tokenize_chunk(self, chunk: CodeChunk) -> list[str]:
        """Tokenize a code chunk for BM25 indexing.

        Combines method name, class name, content, and docstring
        into a single token stream.
        """
        parts = [
            chunk.slice.class_name,
            chunk.slice.method_name,
            chunk.slice.symbol_signature,
            chunk.slice.content,
            chunk.slice.docstring,
        ]
        combined = " ".join(p for p in parts if p)
        return _tokenize_java_text(combined)

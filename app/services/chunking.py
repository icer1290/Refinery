"""Text chunking service for RAG system.

Uses LangChain's RecursiveCharacterTextSplitter for intelligent text chunking
that preserves semantic boundaries.
"""

from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.core import get_logger

logger = get_logger(__name__)
settings = get_settings()


@dataclass
class TextChunk:
    """Represents a chunk of text with metadata."""

    text: str
    chunk_number: int
    start_char: int
    end_char: int


class ChunkingService:
    """Service for chunking text into smaller pieces for embedding.

    Uses recursive character text splitting with customizable separators
    to preserve semantic boundaries (paragraphs, sentences, words).
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ):
        """Initialize the chunking service.

        Args:
            chunk_size: Maximum characters per chunk (default from config)
            chunk_overlap: Characters to overlap between chunks (default from config)
        """
        self.chunk_size = chunk_size or settings.rag_chunk_size
        self.chunk_overlap = chunk_overlap or settings.rag_chunk_overlap

        # Initialize text splitter with hierarchical separators
        # The splitter tries separators in order, splitting by larger units first
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=[
                "\n\n\n",  # Major sections
                "\n\n",    # Paragraphs
                "\n",      # Lines
                ". ",      # Sentences
                " ",       # Words
                "",        # Characters (last resort)
            ],
            length_function=len,
            is_separator_regex=False,
        )

        logger.info(
            "ChunkingService initialized",
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def chunk_text(self, text: str) -> List[TextChunk]:
        """Split text into chunks with metadata.

        Args:
            text: The text to chunk

        Returns:
            List of TextChunk objects with text and position metadata
        """
        if not text or not text.strip():
            logger.debug("Empty text provided, returning empty list")
            return []

        # Use splitter to create documents with metadata
        documents = self.splitter.create_documents([text])

        chunks = []
        current_position = 0

        for i, doc in enumerate(documents):
            chunk_text = doc.page_content

            # Find the actual position of this chunk in the original text
            # This handles cases where overlap might cause position shifts
            start_pos = text.find(chunk_text, current_position)
            if start_pos == -1:
                # Fallback: search from beginning (handles edge cases)
                start_pos = text.find(chunk_text)

            if start_pos == -1:
                # Another fallback: use current position estimate
                logger.warning(
                    "Could not find exact chunk position",
                    chunk_number=i,
                    chunk_preview=chunk_text[:50],
                )
                start_pos = current_position

            end_pos = start_pos + len(chunk_text)

            chunks.append(TextChunk(
                text=chunk_text,
                chunk_number=i,
                start_char=start_pos,
                end_char=end_pos,
            ))

            # Update position for next search (accounting for overlap)
            current_position = max(start_pos + 1, current_position + 1)

        logger.info(
            "Text chunked successfully",
            original_length=len(text),
            num_chunks=len(chunks),
            avg_chunk_length=sum(len(c.text) for c in chunks) // max(len(chunks), 1),
        )

        return chunks

    def chunk_text_with_summary_first(
        self,
        text: str,
        summary: str,
        max_summary_chunk: int = 500,
    ) -> List[TextChunk]:
        """Chunk text with summary prepended to first chunk.

        This strategy improves retrieval for long documents by ensuring
        the summary (high-level information) appears in the first chunk,
        making it more likely to be retrieved for relevant queries.

        Args:
            text: The text to chunk
            summary: Summary to prepend to first chunk
            max_summary_chunk: Maximum characters for summary prefix

        Returns:
            List of TextChunk objects with summary in first chunk
        """
        chunks = self.chunk_text(text)

        if not chunks:
            return chunks

        # Truncate summary if too long
        truncated_summary = summary[:max_summary_chunk]
        if len(summary) > max_summary_chunk:
            truncated_summary = truncated_summary.rstrip() + "..."

        # Prepend summary to first chunk
        summary_prefix = f"[摘要] {truncated_summary}\n\n---\n\n"
        first_chunk = chunks[0]

        # Check if prepending would exceed chunk size
        if len(summary_prefix) + len(first_chunk.text) <= self.chunk_size:
            chunks[0] = TextChunk(
                text=summary_prefix + first_chunk.text,
                chunk_number=0,
                start_char=first_chunk.start_char,
                end_char=first_chunk.end_char,
            )
        else:
            # Insert summary as separate chunk if first chunk would be too long
            logger.debug(
                "Summary too long to prepend, inserting as separate chunk",
                summary_length=len(summary_prefix),
                first_chunk_length=len(first_chunk.text),
            )
            chunks.insert(0, TextChunk(
                text=summary_prefix.rstrip(),
                chunk_number=0,
                start_char=0,
                end_char=0,
            ))
            # Renumber remaining chunks
            for i, chunk in enumerate(chunks[1:], start=1):
                chunks[i] = TextChunk(
                    text=chunk.text,
                    chunk_number=i,
                    start_char=chunk.start_char,
                    end_char=chunk.end_char,
                )

        return chunks


# Singleton instance
_chunking_service: ChunkingService | None = None


def get_chunking_service() -> ChunkingService:
    """Get or create chunking service instance."""
    global _chunking_service
    if _chunking_service is None:
        _chunking_service = ChunkingService()
    return _chunking_service
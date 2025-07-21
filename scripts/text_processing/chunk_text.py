"""Utility functions for chunking text into smaller pieces."""


def chunk_text(text, max_tokens=2048):
    """Splits text into chunks to fit api token limits."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0

    for word in words:
        current_length += len(word) + 1
        if current_length > max_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = len(word) + 1
        current_chunk.append(word)

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

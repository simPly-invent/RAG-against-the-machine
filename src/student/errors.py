"""Custom exceptions for the student RAG pipeline.

Every failure mode raises a `StudentError` subclass carrying an actionable
message, instead of leaking a cryptic third-party traceback (bm25s, pydantic)
or — worse — failing silently and returning meaningless results.
"""


class StudentError(Exception):
    """Base class for every error raised by this project."""


# for Index
class IndexNotFoundError(StudentError):
    """The BM25 index or the chunks file is missing on disk."""

    def __init__(self, missing: str) -> None:
        super().__init__(
            f"Index not found: {missing} is missing. "
            "Run `uv run python -m student index` first."
        )


class IndexCorruptedError(StudentError):
    """The chunks file exists but cannot be parsed."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            f"Corrupted chunks file {path}: {reason}. "
            "Rebuild it with `uv run python -m student index`."
        )


class IndexEmptyError(StudentError):
    """The index was built but contains no document."""

    def __init__(self) -> None:
        super().__init__(
            "Index is empty (0 document indexed). "
            "Run `uv run python -m student index` first."
        )


class IndexDesyncError(StudentError):
    """Index and chunks disagree: doc ids would point to the wrong chunks."""

    def __init__(self, num_docs: int, num_chunks: int) -> None:
        super().__init__(
            f"Index out of sync: {num_docs} indexed documents but "
            f"{num_chunks} chunks. Doc ids are corpus positions, so results "
            "would be silently wrong. Rebuild with "
            "`uv run python -m student index`."
        )


class EmptyCorpusError(StudentError):
    """Nothing to index: no readable source file was found."""

    def __init__(self, directory: str) -> None:
        super().__init__(
            f"No indexable file found under {directory}. "
            "Expected a vLLM checkout with .py/.md files."
        )


# CLI arguments
class InvalidKError(StudentError):
    """`k` is outside the acceptable range."""

    def __init__(self, k: int, corpus_size: int | None = None) -> None:
        if corpus_size is None:
            super().__init__(f"k must be at least 1 (got {k}).")
        else:
            super().__init__(
                f"k={k} exceeds the corpus size ({corpus_size} chunks). "
                f"Use k <= {corpus_size}."
            )


class EmptyQueryError(StudentError):
    """The query yields no searchable token."""

    def __init__(self, query: str) -> None:
        super().__init__(
            f"Query has no searchable term: {query!r}. "
            "BM25 would score every chunk at 0 and return arbitrary results."
        )


# Datasets
class DatasetNotFoundError(StudentError):
    """The dataset file does not exist."""

    def __init__(self, path: str) -> None:
        super().__init__(f"Dataset file not found: {path}")


class DatasetInvalidError(StudentError):
    """The dataset exists but does not match the expected schema."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            f"Invalid dataset {path}: {reason}"
        )


# Answer
class SourceFileError(StudentError):
    """A retrieved source points to a file that cannot be read."""

    def __init__(self, file_path: str, reason: str) -> None:
        super().__init__(
            f"Cannot read source file {file_path}: {reason}. "
            "The indexed repository may have moved or changed."
        )


class ModelLoadError(StudentError):
    """The language model could not be loaded."""

    def __init__(self, model_name: str, reason: str) -> None:
        super().__init__(
            f"Could not load model {model_name}: {reason}"
        )

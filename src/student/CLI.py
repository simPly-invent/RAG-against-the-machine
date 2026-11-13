import json
import bm25s
from student.pymodel import (
    MinimalSource,
    MinimalSearchResults,
    StudentSearchResults,
    RagDataset,
    AnsweredQuestion,
    MinimalAnswer,
    StudentSearchResultsAndAnswer,
)
from student.external_LLM import build_context, generate_answer
from student.chunking import chunk_file
from student.errors import (
    IndexNotFoundError,
    IndexCorruptedError,
    IndexEmptyError,
    IndexDesyncError,
    EmptyCorpusError,
    InvalidKError,
    EmptyQueryError,
    DatasetNotFoundError,
    DatasetInvalidError,
)
from pydantic import ValidationError
from pathlib import Path
from tqdm import tqdm

BM25_DIR = Path("data/processed/bm25_index")
CHUNKS_PATH = Path("data/processed/chunks/chunks.json")


def index(max_chunk_size: int = 2000) -> str:
    resized_dataset = []
    data_set = []
    output_dir = Path("data/processed")
    chunk_dir = Path("data/processed/chunks")
    output_dir.mkdir(parents=True, exist_ok=True)
    chunk_dir.mkdir(parents=True, exist_ok=True)
    directory = Path("data/raw/vllm-0.10.1")
    files = [p for p in directory.rglob(
        "*") if p.is_file() and p.suffix in {".py", ".md"}]
    for file_path in tqdm(files, desc="Brut text process"):
        try:
            with open(file_path, "r") as f:
                text = f.read()
        except UnicodeDecodeError:
            continue
        data_set.append({
            "file_path": str(file_path),
            "first_character_index": 0,
            "last_character_index": len(text),
            "text": text
        })
    with tqdm(total=len(data_set), desc="Chunking process") as t:
        for element in data_set:
            resized_dataset += chunk_file(
                str(element["text"]),
                str(element["file_path"]),
                max_chunk_size
            )
            t.update()
    with open(chunk_dir / "chunks.json", "w", encoding="utf-8") as f:
        json_resized_data = json.dumps(
            resized_dataset, indent=2, ensure_ascii=False)
        f.write(json_resized_data)
    corpus = [c["text"] for c in resized_dataset]
    if not corpus:
        raise EmptyCorpusError(str(directory))
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    retriever.save("data/processed/bm25_index")
    return "Ingestion complete! Indices saved under data/processed/"


def search(query: str, k: int = 10) -> list[dict]:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise InvalidKError(k)
    if not BM25_DIR.is_dir():
        raise IndexNotFoundError(str(BM25_DIR))
    if not CHUNKS_PATH.is_file():
        raise IndexNotFoundError(str(CHUNKS_PATH))
    retriever = bm25s.BM25.load(str(BM25_DIR))
    try:
        chunks = json.loads(CHUNKS_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise IndexCorruptedError(str(CHUNKS_PATH), str(exc)) from exc
    if retriever.scores["num_docs"] == 0:
        raise IndexEmptyError()
    if retriever.scores["num_docs"] != len(chunks):
        raise IndexDesyncError(retriever.scores["num_docs"], len(chunks))
    if k > len(chunks):
        raise InvalidKError(k, len(chunks))
    tokens = bm25s.tokenize(query)
    if not tokens.ids[0]:
        raise EmptyQueryError(query)
    results, scores = retriever.retrieve(tokens, k=k)
    try:
        source = [
            MinimalSource(
                file_path=chunks[int(i)]["file_path"],
                first_character_index=chunks[int(i)]["first_character_index"],
                last_character_index=chunks[int(i)]["last_character_index"],
            ).model_dump()
            for i in results[0]
        ]
    except IndexError:
        source = []
    return (source)


def search_dataset(dataset_path: str, k: int = 10,
                   save_directory: str = "data/output/search_results") -> None:
    if not isinstance(k, int) or isinstance(k, bool) or k <= 0:
        raise InvalidKError(k)
    if not Path(dataset_path).is_file():
        raise DatasetNotFoundError(dataset_path)
    try:
        dataset = RagDataset.model_validate_json(
            Path(dataset_path).read_text())
    except ValidationError as exc:
        raise DatasetInvalidError(
            dataset_path, str(exc).splitlines()[0]) from exc
    questions = dataset.rag_questions
    if not BM25_DIR.is_dir():
        raise IndexNotFoundError(str(BM25_DIR))
    if not CHUNKS_PATH.is_file():
        raise IndexNotFoundError(str(CHUNKS_PATH))
    retriever = bm25s.BM25.load(str(BM25_DIR))
    try:
        chunks = json.loads(CHUNKS_PATH.read_text())
    except json.JSONDecodeError as exc:
        raise IndexCorruptedError(str(CHUNKS_PATH), str(exc)) from exc
    if retriever.scores["num_docs"] == 0:
        raise IndexEmptyError()
    if retriever.scores["num_docs"] != len(chunks):
        raise IndexDesyncError(retriever.scores["num_docs"], len(chunks))
    if k > len(chunks):
        raise InvalidKError(k, len(chunks))
    if not questions:
        Path(save_directory).mkdir(parents=True, exist_ok=True)
        (Path(save_directory) / Path(dataset_path).name).write_text(
            StudentSearchResults(search_results=[], k=k)
            .model_dump_json(indent=2))
        return
    tokens = bm25s.tokenize([q.question for q in questions])
    results, _ = retriever.retrieve(tokens, k=k)
    try:
        search_results = [
            MinimalSearchResults(
                question_id=q.question_id,
                question=q.question,
                retrieved_sources=[
                    MinimalSource(
                        file_path=chunks[int(i)]["file_path"],
                        first_character_index=chunks[int(
                            i)]["first_character_index"],
                        last_character_index=chunks[int(
                            i)]["last_character_index"],
                    )
                    for i in results[j]
                    if 0 <= int(i) < len(chunks)
                ] if tokens.ids[j] else [],
            )
            for j, q in enumerate(questions)
        ]
    except IndexError:
        search_results = []
    out = StudentSearchResults(search_results=search_results, k=k)
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    (Path(save_directory) / Path(dataset_path).name).write_text(
        out.model_dump_json(indent=2))


def answer(query: str, k: int = 10) -> str:
    sources = [MinimalSource(**d) for d in search(query, k)]
    context = build_context(sources)
    generated = generate_answer(query, context)
    result = AnsweredQuestion(
        question=query, sources=sources, answer=generated)
    return result.model_dump_json(indent=2)


def answer_dataset(
        student_search_results_path: str,
        save_directory: str = "data/output/search_results_and_answer") -> None:
    if not Path(student_search_results_path).is_file():
        raise DatasetNotFoundError(student_search_results_path)
    try:
        student = StudentSearchResults.model_validate_json(
            Path(student_search_results_path).read_text())
    except ValidationError as exc:
        raise DatasetInvalidError(
            student_search_results_path, str(exc).splitlines()[0]) from exc
    questions = student.search_results
    print(
        f"Loaded {len(questions)} questions "
        f"from {student_search_results_path}")

    answered = []
    for q in tqdm(questions, desc="Answering questions"):
        context = build_context(q.retrieved_sources)
        generated = generate_answer(q.question, context)
        answered.append(MinimalAnswer(
            question_id=q.question_id,
            question=q.question,
            retrieved_sources=q.retrieved_sources,
            answer=generated,
        ))
    print(f"Processed {len(questions)} of {len(questions)} questions")

    out = StudentSearchResultsAndAnswer(search_results=answered, k=student.k)
    Path(save_directory).mkdir(parents=True, exist_ok=True)
    output_path = Path(save_directory) / Path(student_search_results_path).name
    output_path.write_text(out.model_dump_json(indent=2))
    print(f"Saved student_search_results_and_answer to {output_path}")


def _overlap_ratio(correct: MinimalSource, retrieved: MinimalSource) -> float:
    if correct.file_path.lower() != retrieved.file_path.lower():
        return 0.0
    start = max(correct.first_character_index, retrieved.first_character_index)
    end = min(correct.last_character_index, retrieved.last_character_index)
    intersection = max(0, end - start)
    correct_length = (
        correct.last_character_index - correct.first_character_index)
    if correct_length <= 0:
        return 0.0
    return intersection / correct_length


def evaluate(dataset_path: str, search_results_path: str) -> dict:
    for path in (dataset_path, search_results_path):
        if not Path(path).is_file():
            raise DatasetNotFoundError(path)
    try:
        ground_truth = RagDataset.model_validate_json(
            Path(dataset_path).read_text())
    except ValidationError as exc:
        raise DatasetInvalidError(
            dataset_path, str(exc).splitlines()[0]) from exc
    try:
        student = StudentSearchResults.model_validate_json(
            Path(search_results_path).read_text())
    except ValidationError as exc:
        raise DatasetInvalidError(
            search_results_path, str(exc).splitlines()[0]) from exc
    student_by_id = {
        r.question_id: r.retrieved_sources for r in student.search_results}

    per_question: dict[str, float] = {}
    for q in ground_truth.rag_questions:
        if not isinstance(q, AnsweredQuestion) or not q.sources:
            continue
        retrieved = student_by_id.get(q.question_id, [])
        found = sum(
            1 for correct in q.sources
            if any(_overlap_ratio(correct, r) >= 0.05 for r in retrieved)
        )
        per_question[q.question_id] = found / len(q.sources)

    mean_recall = sum(per_question.values()) / \
        len(per_question) if per_question else 0.0
    return {"recall_at_k": mean_recall, "per_question": per_question}

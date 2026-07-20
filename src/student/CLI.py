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
from pathlib import Path
from tqdm import tqdm


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
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    retriever.save("data/processed/bm25_index")
    return "Ingestion complete! Indices saved under data/processed/ ✅"


def search(query: str, k: int = 10) -> list[dict]:
    retriever = bm25s.BM25.load("data/processed/bm25_index")
    chunks = json.loads(Path("data/processed/chunks/chunks.json").read_text())
    if retriever.scores["num_docs"] == 0:
        raise ValueError("Index vide : lance `index` avant `search`.")
    if retriever.scores["num_docs"] != len(chunks):
        raise ValueError(
            f"Index desynchronise : {retriever.scores['num_docs']} docs "
            f"indexes mais {len(chunks)} chunks. Relance `index`."
        )
    results, scores = retriever.retrieve(bm25s.tokenize(query), k=k)
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
    dataset = RagDataset.model_validate_json(Path(dataset_path).read_text())
    questions = dataset.rag_questions
    retriever = bm25s.BM25.load("data/processed/bm25_index")
    chunks = json.loads(Path("data/processed/chunks/chunks.json").read_text())
    if retriever.scores["num_docs"] == 0:
        raise ValueError("Index vide : lance `index` avant `search_dataset`.")
    if retriever.scores["num_docs"] != len(chunks):
        raise ValueError(
            f"Index desynchronise : {retriever.scores['num_docs']} docs "
            f"indexes mais {len(chunks)} chunks. Relance `index`."
        )
    results, _ = retriever.retrieve(
        bm25s.tokenize([q.question for q in questions]), k=k)
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
                ],
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
    student = StudentSearchResults.model_validate_json(
        Path(student_search_results_path).read_text())
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
    ground_truth = RagDataset.model_validate_json(
        Path(dataset_path).read_text())
    student = StudentSearchResults.model_validate_json(
        Path(search_results_path).read_text())
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


# if __name__ == "__main__":
    # fire.Fire(index)
    # fire.Fire(search)
    # fire.Fire(search_dataset)
    # fire.Fire(answer)
    # fire.Fire(answer_dataset)
    # fire.Fire(evaluate)

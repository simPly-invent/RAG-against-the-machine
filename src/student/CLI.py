import json
import bm25s
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
    files = [p for p in directory.rglob("*") if p.is_file() and p.suffix in {".py", ".md"}]
    for file_path in tqdm(files, desc="Brut text process"):
        if file_path.is_file():
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
                    element.get("text"),
                    element.get("file_path"),
                    max_chunk_size
                    )
            t.update()
    try:
        with open(chunk_dir / "chunks.json", "w", encoding="utf-8") as f:
            json_resized_data = json.dumps(resized_dataset, indent=2, ensure_ascii=False)
            f.write(json_resized_data)
    except Exception as e:
            return(f"{e} dead bro")
    corpus = [c["text"] for c in resized_dataset]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(corpus))
    query = "does the fish purr like a cat?"
    retriever.retrieve(bm25s.tokenize(query), k=2)
    retriever.save("data/processed/bm25_index")
    return("Ingestion complete! Indices saved under data/processed/ ✅")


def search(query: str, k: int = 10) -> list[dict]:
    return(search)


def search_dataset(search_dataset="c'est le search data_set"):
    return(search_dataset)


def answer(answer="answer"):
    return(answer)


def answer_dataset(answer_dataset="answer_dataset"):
    return(answer_dataset)


def evaluate(evaluate="evaluate"):
    return(evaluate)


# if __name__ == "__main__":
    # fire.Fire(index)
    # fire.Fire(search)
    # fire.Fire(search_dataset)
    # fire.Fire(answer)
    # fire.Fire(answer_dataset)
    # fire.Fire(evaluate)
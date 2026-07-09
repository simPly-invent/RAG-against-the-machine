import fire
from student import CLI

if __name__ == "__main__":
    fire.Fire({
        "index": CLI.index,
        "search": CLI.search,
        "search_dataset": CLI.search_dataset,
        "answer": CLI.answer,
        "answer_dataset": CLI.answer_dataset,
        "evaluate": CLI.evaluate
    })
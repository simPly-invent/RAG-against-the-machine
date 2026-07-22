import sys
import fire
from . import CLI
from .errors import StudentError

if __name__ == "__main__":
    try:
        fire.Fire({
            "index": CLI.index,
            "search": CLI.search,
            "search_dataset": CLI.search_dataset,
            "answer": CLI.answer,
            "answer_dataset": CLI.answer_dataset,
            "evaluate": CLI.evaluate
        })
    except StudentError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)

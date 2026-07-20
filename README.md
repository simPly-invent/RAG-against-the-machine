*This project has been created as part of the 42 curriculum by mobenais.*

# RAG against the machine

A **Retrieval-Augmented Generation** system that answers natural-language
questions about the [vLLM](https://github.com/vllm-project/vllm) codebase
(version `0.10.1`). It ingests the repository, builds a BM25 index over smart
chunks, retrieves the most relevant passages for a question, and generates a
grounded answer with a local LLM.

Pipeline: **ingest → index → search → answer → evaluate**.

## Description

The system indexes every `.py` and `.md` file of `vllm-0.10.1`, splits them
into character-bounded chunks, and stores a BM25 index alongside the chunks.
Given a question, it tokenizes the query, retrieves the top-k chunks, and
returns each result as a `MinimalSource` (`file_path`,
`first_character_index`, `last_character_index`). Answers are produced by
`Qwen/Qwen3-0.6B` fed with the retrieved context.

Every command is exposed through a [Python Fire](https://github.com/google/python-fire)
CLI, all data models are validated with [Pydantic](https://docs.pydantic.dev/),
and progress is reported with [tqdm](https://tqdm.github.io/).

## Instructions

### Requirements

- **Python 3.10** (pinned via `.python-version`)
- [`uv`](https://docs.astral.sh/uv/) as the package manager

### Install

```bash
make install        # uv sync
```

### Build the index (required before any search)

```bash
uv run python -m student index --max_chunk_size 2000
```

This reads `data/raw/vllm-0.10.1/`, writes the chunks to
`data/processed/chunks/chunks.json` and the BM25 index to
`data/processed/bm25_index/`.

## Resources & system architecture

```
src/student/
├── __main__.py       # single fire.Fire({...}) mapping the 6 CLI commands
├── CLI.py            # index / search / search_dataset / answer / answer_dataset / evaluate
├── chunking.py       # chunk_file(): Python-aware vs fixed-window splitting
├── external_LLM.py   # Qwen3-0.6B loading, context building, answer generation
└── pymodel.py        # imposed Pydantic models

data/
├── raw/vllm-0.10.1/                     # repository to index
├── processed/bm25_index/                # BM25 index (5 files, generated)
├── processed/chunks/chunks.json         # chunks (generated)
└── output/
    ├── search_results/                  # search_dataset outputs
    └── search_results_and_answer/       # answer_dataset outputs
```

Data flow: `index` produces `chunks.json` + the BM25 index. `search` /
`search_dataset` load both, retrieve doc ids (positions in the corpus), and map
them back to chunks. `answer` / `answer_dataset` add the LLM generation step.
`evaluate` compares retrieved sources against the ground truth.

Key libraries: `bm25s` (BM25), `langchain-text-splitters` (Python-aware
chunking), `transformers` (Qwen3-0.6B), `fire`, `pydantic`, `tqdm`.

## Chunking strategy

Chunking is dispatched on file type in `chunk_file()`:

- **Python** (`.py`) → LangChain
  `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)` with
  `chunk_overlap=200` and `add_start_index=True`. Splitting on Python
  structural separators (classes, functions, blocks) keeps semantically
  coherent chunks, and `add_start_index` gives exact character offsets.
- **Markdown / other** → a fixed-size sliding window of `max_chunk_size`.

The chunk size is capped at **2000 characters** and is configurable through
`--max_chunk_size`. Each chunk records `file_path`, `first_character_index` and
`last_character_index`, so a result can be traced back to an exact byte range.

> The corpus order is preserved end to end: BM25 doc ids are **positions** in
> the corpus, so chunks are never reordered or filtered between indexing and
> search.

## Retrieval method

BM25 via [`bm25s`](https://github.com/xhluca/bm25s):

1. At index time, `corpus = [chunk["text"] for chunk in chunks]` is tokenized
   and indexed, then persisted to `data/processed/bm25_index/`.
2. At search time, the index and chunks are loaded, the query is tokenized with
   the **same** tokenizer, and `retrieve(query_tokens, k)` returns the top-k
   doc ids, which are mapped back to `MinimalSource` objects.

A consistency guard raises if the index and `chunks.json` are out of sync
(`num_docs != len(chunks)`), turning a silent wrong-results failure into a loud,
actionable error.

## Performance analysis

Measured on the public `AnsweredQuestions` datasets, **recall@5**:

| Corpus | recall@5 | Threshold | Status |
|--------|----------|-----------|--------|
| Docs   | **82 %** | ≥ 80 %    | ✅ pass |
| Code   | **53 %** | ≥ 50 %    | ✅ pass |

Recall@k counts a ground-truth source as *found* when a retrieved source
overlaps it by ≥ 5 % (same file, intersecting character range). The per-question
score is `found / total_correct_sources`, averaged over questions.

- Corpus size: ~14.3k chunks for the full repository at `max_chunk_size=2000`.
- The LLM is loaded lazily and cached (`lru_cache`), so retrieval-only commands
  never pay the model-loading cost — this keeps cold start low and lets
  `answer_dataset` reuse a single model instance across all questions.

## Design decisions

- **Lazy, cached LLM loading** — `load_model()` is `@lru_cache`d and only called
  from `generate_answer()`, so `index`/`search` never load Qwen.
- **Output filename mirrors the input dataset name** — `search_dataset` and
  `answer_dataset` write to `<save_directory>/<input dataset name>`.
- **Character indices over token indices** — results are byte ranges, matching
  the ground-truth format and the evaluation overlap metric.
- **Fail loud, not silent** — degenerate inputs (empty index, index/chunks
  desync) raise explicit errors instead of returning misleading results.
- **Dev tooling separated from deliverables** — `tester/` and generated data are
  excluded from lint and from version control.

## Challenges faced

- **Empty / stopword-only queries.** With zero tokens, BM25 scores every chunk
  at 0 and returns arbitrary top-k results that *look* relevant. Detecting the
  empty-token case avoids reporting garbage.
- **Index / chunks desynchronization.** Doc ids are corpus positions; if
  `chunks.json` is regenerated without rebuilding the index, results stay
  in-range but become wrong. The `num_docs == len(chunks)` guard catches it.
- **Tokenizer behavior on identifiers.** The default token pattern keeps
  `snake_case` identifiers as single tokens, which limits code recall — a custom
  token pattern is the main lever for further improvement.
- **Toolchain.** Pinning Python 3.10 and keeping the code 3.10-compatible
  (e.g. avoiding multi-line f-string expressions introduced by aggressive
  auto-formatting, which are only valid from 3.12).

## Example usage

```bash
# Index the repository
uv run python -m student index --max_chunk_size 2000

# Single query
uv run python -m student search "How to configure the OpenAI server?" --k 10

# Batch search from a dataset (output name == input dataset name)
uv run python -m student search_dataset \
    --dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results

# Answer a single question
uv run python -m student answer "How to configure the OpenAI server?" --k 10

# Answer a batch from previous search results
uv run python -m student answer_dataset \
    --student_search_results_path data/output/search_results/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer

# Evaluate recall against a ground-truth dataset
uv run python -m student evaluate \
    --dataset_path datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
    --search_results_path data/output/search_results/dataset_docs_public.json
```

### Makefile targets

| Target | Description |
|--------|-------------|
| `make install` | Install dependencies via `uv` |
| `make run ARGS="..."` | Run the CLI (`uv run python -m student $ARGS`) |
| `make debug ARGS="..."` | Run the CLI under `pdb` |
| `make lint` | `flake8` + `mypy` (imposed rules) |
| `make clean` | Remove caches and generated files |

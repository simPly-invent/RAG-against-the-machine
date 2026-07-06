# CLAUDE.md — RAG against the machine

Fichier de contexte persistant pour Claude. À lire au début de chaque session.

## Objectif du projet

Construire un système **RAG (Retrieval-Augmented Generation)** qui répond à des questions
sur le codebase **vLLM** (fourni dans `vllm-0.10.1/`). Pipeline : ingest → index → search → answer → evaluate.

## Contraintes techniques (imposées par le sujet)

- **Python 3.10** obligatoire
- Gestionnaire de paquets : **`uv`** (obligatoire)
- CLI : **Python Fire** (obligatoire)
- Modèles : validation via **Pydantic**
- Progress bars : **tqdm**
- LLM : **Qwen/Qwen3-0.6B** (défaut, d'autres modèles compatibles OK)
- Lint : **flake8** + **mypy** (règles imposées, voir Makefile ci-dessous)
- Librairies recommandées : `transformers`, `dspy`, `fire`, `tqdm`, `langchain`, `bm25s`, `chromadb`

## Composants à livrer

1. **Knowledge Base Ingestion**
   - Lire tous les fichiers utiles de `vllm-0.10.1/`
   - Chunking intelligent différent pour Python vs Markdown
   - Chunk size max **2000 chars**, configurable via CLI
   - Index TF-IDF ou BM25 (BM25 recommandé, `bm25s`)
   - Indexation ≤ **5 min**

2. **Retrieval System**
   - Top-k sur index
   - Chaque résultat : `file_path`, `first_character_index`, `last_character_index`
   - Batch depuis JSON datasets
   - Cible : **recall@5 ≥ 80% docs, ≥ 50% code**

3. **Answer Generation**
   - LLM Qwen3-0.6B avec contexte retrieved
   - Respect token limits
   - Output JSON conforme aux modèles Pydantic
   - Answer : self-contained, source-grounded, faithful, relevant

4. **Evaluation**
   - Métrique **recall@k** (overlap ≥ 5% entre source retrieved et source correcte = "found")
   - Score par question = `number_found / total_number_of_correct_sources`

## CLI obligatoire (Fire)

- `index` : indexer le repo
- `search` : query unique
- `search_dataset` : batch depuis JSON
- `answer` : answer une query
- `answer_dataset` : answers en batch
- `evaluate` : recall@k vs ground truth

Exemples du sujet :
```
uv run python -m student index --max_chunk_size 2000
uv run python -m student answer "How to configure OpenAI server?" --k 10
uv run python -m student search_dataset --dataset_path ... --k 10 --save_directory ...
uv run python -m student answer_dataset --student_search_results_path ... --save_directory ...
```

## Makefile — cibles requises

- `install` : dépendances (via `uv`)
- `run` : script principal
- `debug` : via pdb
- `clean` : `__pycache__`, `.mypy_cache`
- `lint` : `flake8 . && mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`
- `lint-strict` (optionnel) : `flake8 . && mypy . --strict`

## Modèles Pydantic imposés

```python
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int

class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str

class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str

class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]

class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    answer: str

class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int

class StudentSearchResultsAndAnswer(StudentSearchResults):
    search_results: List[MinimalAnswer]
```
Peut être étendu si besoin.

## Performances cibles

- Indexation ≤ 5 min
- Cold start ≤ 60 s (1er retrieval + chargement modèle)
- 1000 questions ≤ 90 s (warm)
- Recall@5 : ≥ 80% docs, ≥ 50% code
- Dépasser les seuils = bonus points

## Livrables Git

- `src/` avec le code
- `pyproject.toml` + `uv.lock`
- `README.md` (English) avec : première ligne italique "*This project has been created as part of the 42 curriculum by <login>*", sections Description / Instructions / Resources + System architecture / Chunking strategy / Retrieval method / Performance analysis / Design decisions / Challenges faced / Example usage
- `Makefile`
- `.gitignore`
- **PAS** de gros fichiers (repo vLLM, modèle, outputs générés)

## Bonus (points en plus)

- Query expansion (synonymes, rewriting)
- Embeddings sémantiques
- Caching (index, query)
- Retrieval hybride (BM25 + dense)
- Inférence locale via vLLM

## État du dossier local

```
/home/mobenais/Documents/RAG-against-the-machine/
├── rag.pdf                    # sujet
├── vllm-0.10.1/               # repo à indexer
├── datasets_public/public/
│   ├── AnsweredQuestions/     # ground truth pour eval
│   │   ├── dataset_code_public.json
│   │   └── dataset_docs_public.json
│   └── UnansweredQuestions/   # à traiter
│       ├── dataset_code_public.json
│       └── dataset_docs_public.json
└── moulinette_pkg/            # moulinette d'éval (fedora + ubuntu)
```

## Avancement

- [x] Lecture du sujet
- [x] Inspection dossier
- [ ] Setup projet `uv` + structure `src/`
- [ ] Modèles Pydantic
- [ ] Ingestion + chunking (Python / Markdown)
- [ ] Index BM25
- [ ] CLI Fire (index, search, search_dataset, answer, answer_dataset, evaluate)
- [ ] LLM Qwen3-0.6B integration
- [ ] Recall@k evaluation
- [ ] Perf tuning (viser recall@5 > seuils)
- [ ] Makefile + lint clean
- [ ] README.md complet
- [ ] Bonus

## Notes de session

_À mettre à jour à chaque session avec les décisions/blocages/à-faire._

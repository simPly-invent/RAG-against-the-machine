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
├── data/
│   ├── raw/vllm-0.10.1/       # repo à indexer (déplacé ici : c'est ce que
│   │                          #   la moulinette attend, cf. sujet V.6.1)
│   └── processed/
│       ├── bm25_index/        # écrit par `index` (dossier, 5 fichiers)
│       └── chunks/            # sujet = dossier au PLURIEL
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
- [x] Setup projet `uv` + structure `src/`
- [x] Modèles Pydantic
- [x] Ingestion + chunking (Python / Markdown)
- [x] Index BM25 (`bm25s`, persisté dans `data/processed/bm25_index/`)
- [x] CLI Fire (index, search, search_dataset, answer, answer_dataset, evaluate)
- [ ] `search` : premier vrai retrieval  ← **prochaine étape**
- [ ] LLM Qwen3-0.6B integration
- [ ] Recall@k evaluation
- [ ] Perf tuning (viser recall@5 > seuils)
- [ ] Makefile + lint clean
- [ ] README.md complet
- [ ] Bonus

## Découvertes clés du sujet (à ne pas oublier)

- **Arborescence attendue par la moulinette** :
  - `data/raw/vllm-0.10.1/` (repo à indexer — **déplacé, c'est fait**)
  - `data/processed/` → `bm25_index/` + `chunks/`
  - `data/output/search_results/` et `data/output/search_results_and_answer/`
  - Le sujet ne *dit* jamais « mets le repo dans `data/raw/` » : il le **montre**
    en V.6.1 « Files before » (`ls -l data/raw` → `vllm-0.10.1`). C'est l'état de
    l'environnement à la correction, pas un choix. Idem `chunks` : le `ls` de V.6.2
    montre un `d` en tête de ligne → c'est un **dossier**, pas un `chunks.json`.
- **`search` n'a aucun exemple d'invocation dans le sujet** (seule commande des 6
  dans ce cas). Proto déduit par symétrie avec `answer "..." --k 10` :
  `def search(query: str, k: int = 10) -> list[dict]` — query positionnel, `k`
  optionnel, retourne des `MinimalSource` (3 champs, **sans** `text` ni `index`).
- V.2.5 : « CLI arguments will be tested thoroughly with edge cases ». Pour `search` :
  index absent, `k` ≤ 0, `k` > nombre de chunks (bm25s lève), query vide ou
  100% stopwords (tokenize renvoie 0 token → cousin du `ValueError: max() … empty`).
- **Le nom du fichier de sortie == le nom du dataset d'entrée**
  (`dataset_docs_public.json` en entrée → `dataset_docs_public.json` en sortie).
- L'`evaluate` **officiel** n'est pas le nôtre : la moulinette lance
  `uv run python -m moulinette evaluate_student_search_results --student_answer_path ... --dataset_path ... --k 10`.
  Notre commande `student evaluate` doit exister quand même (exigée au chap. V.2.5).
- La ground truth préfixe les `file_path` par `data/raw/vllm-0.10.1/...` → nos
  `file_path` doivent être **comparables** (normalisation nécessaire).

## État du code

- `src/student/__main__.py` : **un seul** `fire.Fire({...})` mappant les 6 commandes.
  (`python -m student` exécute `__main__.py`, pas `CLI.py`.)
- `pyproject.toml` : `[build-system]` hatchling + `packages = ["src/student"]`
  (sans ça : `No module named student`).
- `src/student/chunking.py` :
  - `chunk_file()` = aiguilleur sur `Path(...).suffix`
  - `.py` → `chunk_code()` : LangChain `RecursiveCharacterTextSplitter.from_language(Language.PYTHON)`,
    `chunk_overlap=200`, **`add_start_index=True`** (indispensable pour les indices)
  - autres → `chunk_text()` : fenêtre fixe de `max_chunk_size`
  - Vérifié : 0 indice faux, aucun chunk > 2000.
- `src/student/CLI.py` : `index(max_chunk_size: int = 2000)` lit `.py`/`.md` sous
  `data/raw/vllm-0.10.1`, chunk, persiste les chunks, puis construit et sauvegarde
  l'index BM25. Les 5 autres commandes sont encore des stubs.
- Dépendances ajoutées : `tqdm`, `langchain`, `langchain-text-splitters`, `bm25s`.

## Testeur maison

`tester/check.py` (outil de dev, non soumis). Lancer **avec uv** :
`uv run python tester/check.py [--full] [--step <nom>]`.
Statuts : `OK` / `KO` / `~~` / `..` / `[]` (conception restant à faire).
L'étape `design` est **auto-vérifiée** : elle analyse `chunks.json` (structure,
extensions, indices, `max_chunk_size`) et la source de chaque fonction CLI.

Ses propres défauts (à corriger, il resservira) :
- `CHUNKS_JSON` est un chemin **en dur** → se désynchronise dès qu'on bouge la sortie.
- `_design_index` ne distingue pas « 0 chunk » de « chunks du mauvais type » : sur une
  liste vide il annonce `PASS — 0 chunks` puis `KO — aucun dict trouvé`. Ajouter un
  `if not chunks: FAIL "chunks.json est vide"` juste après `_flatten`.

## À corriger (dette connue)

- [x] ~~chemin absolu en dur~~ → relatif (`data/raw/vllm-0.10.1`), reste à en faire un
  argument CLI (`repo_path`).
- [x] ~~`max_chunk_size` pas branché~~ → fait.
- `CLI.py` : le `try/except` autour de l'écriture des chunks fait `return` d'une string
  → Fire imprime et sort avec le code **0**. Une indexation ratée passe pour un succès.
  Supprimer le `try/except`, laisser l'exception remonter.
- `CLI.py` : code de debug résiduel dans `index` (query « does the fish purr like a
  cat? », `retrieve`, `results`/`scores` jamais lus → `F841`).
- `CLI.py` : `"data/processed/bm25_index"` réécrit en dur alors que `output_dir` existe.
- `CLI.py` : `if file_path.is_file()` redondant (déjà filtré dans la compréhension).
- `CLI.py` : deux sorties de chunks coexistent (`processed/chunks.json` et un
  sous-dossier). Le sujet veut `data/processed/chunks/`. Trancher.
- mypy `--disallow-untyped-defs` : `resized_dataset = []` non annoté → `list[dict]`.
  Les 5 stubs (`search`, …) n'ont ni annotation ni retour typé.
- **venv en Python 3.12** alors que le sujet impose **3.10** (`requires-python = ">=3.10"`
  l'autorise, et pas de `.python-version`). → `uv python pin 3.10` + recréer le venv.
- Chemins encodés à 3 endroits (`CLI.py`, `check.py`, sujet) → envisager un
  `src/student/paths.py` avec `RAW_DIR` / `PROCESSED_DIR` / `CHUNKS_DIR` / `BM25_DIR`.
- Branche `if len(doc.page_content) > max_chunk_size` dans `chunk_code` : **code mort**
  (le dernier séparateur LangChain est `""`, donc la taille n'est jamais dépassée).

## Notes de session

### Session du 2026-07-09

Pièges Python rencontrés (à retenir) :
- `Path("~/...")` n'expanse **pas** le `~` → la boucle ne trouve rien, sans erreur.
- `from chunking import x` échoue : il faut `from student.chunking import x`.
- `liste += dict` **itère les clés** du dict → utiliser `.append()` pour ajouter un
  dict, et `+=` seulement pour concaténer une liste.
- Fire **affiche automatiquement la valeur de retour** d'une commande → pas besoin de
  `print`. Mais persister ≠ retourner : `index` doit **écrire sur disque**.
- tqdm : utiliser `desc="..."` plutôt qu'un `print(..., end="")` (tqdm écrit sur stderr,
  et un print sans `\n` n'est pas flush).

**BM25 : fait.** `index` construit `corpus = [c["text"] for c in chunks]` →
`bm25s.tokenize(corpus)` → `BM25().index(tokens)` → `.save("data/processed/bm25_index")`.
Prochaine étape : `search` = `BM25.load(...)` → `retrieve(query_tokens, k)` → `doc_id`
→ `chunks[doc_id]`.

⚠️ **L'ordre de `chunks` doit rester identique à celui du `corpus`** : les `doc_id`
renvoyés sont des **positions** dans le corpus. Réordonner/filtrer entre l'index et
le search casse le recall silencieusement.

Ce qu'on a appris sur `bm25s` (vérifié dans le code source / à l'exécution) :
- `BM25(...)` **n'a pas** de `**kwargs`. Le seul kwarg de corpus s'appelle `corpus=`,
  et il est optionnel (il sert juste à ce que `retrieve` renvoie les objets, pas les ids).
- `tokenize()` attend `str | List[str]`. Lui passer la `list[dict]` de chunks → erreur.
- `save(save_dir)` : `save_dir` est un **dossier**, jamais un fichier. bm25s fait
  `Path(save_dir).mkdir(parents=True, exist_ok=True)` lui-même, puis écrit 5 fichiers
  dedans. Donner `bm25_index.txt` crée donc un *dossier* nommé `bm25_index.txt`.
- `load(save_dir, mmap=False)` : passer `mmap=True` mappe les `.npy` au lieu de les
  charger → levier direct pour la contrainte **cold start ≤ 60 s**. Pour la phase perf.
- `retrieve()` renvoie `(results, scores)` de shape `(n_queries, k)` → query unique =
  `results[0]`.
- ⚠️ Le `token_pattern` par défaut est `(?u)\b\w\w+\b`, et **`\w` inclut l'underscore**.
  Donc `get_supported_mm_limits` reste **un seul token** (vérifié), et le camelCase est
  juste lowercasé. C'est LE levier pour le recall code : `token_pattern` custom, appliqué
  **identiquement** à l'indexation et à la recherche.

Pièges Python / outillage (suite) :
- Corpus vide → bm25s lève `ValueError: max() iterable argument is empty`, 15 lignes et
  3 libs plus loin que la cause. **Le vrai signal est le `0it` de la barre tqdm.** Quand
  une barre affiche `0it`, s'arrêter là et ne pas lire le traceback.
- Fire : si un flag ne correspond à aucun paramètre, la fonction est appelée **sans lui**,
  puis Fire tente d'appliquer le flag au **retour**. D'où le `rfind | rjust | rsplit …`
  (méthodes de `str`) et le vrai message noyé plus haut : `Could not consume arg`.
  Autrement dit : la commande *s'exécute* mais sort en code ≠ 0.
- Un `return` d'une string dans un `except` = code de sortie **0** = échec invisible.

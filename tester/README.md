# testeur/ — checklist exécutable (outil de dev, non soumis)

`check.py` vérifie **étape par étape** si le projet respecte le sujet.
On le relance après chaque avancée : tant qu'une case est `KO`, l'étape n'est
pas passée. Objectif = tout vert.

> ⚠️ Ce testeur **ne remplace pas** la moulinette officielle. Il aide juste à
> avancer dans la bonne direction et à ne rien oublier.

## Lancer

Toujours avec **`uv`** : le script importe `pydantic` (tes modèles), qui n'est
installé que dans le venv du projet.

```bash
# Checklist rapide (structure, modèles, métrique, commandes CLI, roadmap)
uv run python testeur/check.py

# + checks lents : indexation réelle + un search réel
uv run python testeur/check.py --full

# Une seule étape
uv run python testeur/check.py --step models
uv run python testeur/check.py --step design
```

> ❌ `./testeur/check.py` seul échoue (`ModuleNotFoundError`) : il utiliserait le
> Python système, sans les dépendances du projet. Passe toujours par `uv run`.

> 💡 Dans l'aide, `--step {structure,models,...}` liste les **choix possibles**.
> Ce n'est pas à recopier tel quel : on choisit **une** valeur, ex. `--step cli`.

Valeurs de `--step` : `structure`, `models`, `selftest`, `cli`, `design`,
`index`, `search`, `search_dataset`, `recall`, `answers`.

## Évaluer le recall@k (comme la moulinette, mais en local)

Une fois que `search_dataset` produit un JSON de résultats :

```bash
uv run python testeur/check.py --recall \
  --search-results data/output/search_results/dataset_docs_public.json \
  --ground-truth  datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
  --kind docs      # compare recall@5 au seuil 80% (docs) / 50% (code)
```

Le calcul réplique la règle du sujet : une source est *found* si le
chevauchement de caractères ≥ 5 % ; score par question =
`nb_found / nb_sources_correctes`, moyenné sur les questions. Les `file_path`
sont normalisés (le préfixe `data/raw/...` est ignoré) avant comparaison.

## Vérifier le format des réponses générées

```bash
uv run python testeur/check.py --step answers \
  --answers data/output/search_results_and_answer/dataset_docs_public.json
```

Sans `--answers`, l'étape inspecte automatiquement le dossier standard
`data/output/search_results_and_answer/`.

## Légende

| Icône | Sens |
|-------|------|
| `OK ` | check passé (vérifié automatiquement) |
| `KO ` | échec (si obligatoire → code de sortie 1) |
| `~~ ` | avertissement (non bloquant) |
| `.. ` | ignoré (check lent, sortie absente, ou argument manquant) |
| `[] ` | **conception** : étape pas encore détectée dans ton code / tes artefacts |

## Ce qui est vérifié

| Étape | Contenu |
|-------|---------|
| `structure` | Makefile, `pyproject.toml`, `uv.lock`, package `student` + `__main__.py`, `.gitignore`, README |
| `models` | Présence et champs des 8 modèles Pydantic imposés |
| `selftest` | Auto-test de la métrique recall (overlap 5 %, normalisation des chemins) |
| `cli` | Les 6 commandes Fire : `index`, `search`, `search_dataset`, `answer`, `answer_dataset`, `evaluate` |
| `design` | **Roadmap auto-vérifiée** : analyse `data/processed/chunks.json` (structure, extensions, indices, `max_chunk_size`) et la source de chaque fonction CLI pour cocher ce qui est réellement fait |
| `index` | Artefacts dans `data/processed/` ; avec `--full` : exécution + durée ≤ 5 min |
| `search` (`--full`) | Sortie JSON valide, sources bien formées (`file_path`, indices) |
| `search_dataset` | Sortie `data/output/search_results/`, nommée comme le dataset, conforme `StudentSearchResults` |
| `recall` | recall@k vs ground truth, comparé aux seuils du sujet |
| `answers` | Sortie `data/output/search_results_and_answer/`, conforme `StudentSearchResultsAndAnswer` |

## Rappels du sujet encodés dans le testeur

- Repo à indexer : `data/raw/vllm-0.10.1/` (fallback local : `vllm-0.10.1/`).
- Index persisté dans `data/processed/` (`bm25_index/`, `chunks/`).
- **Nom du fichier de sortie == nom du dataset d'entrée**
  (`dataset_docs_public.json` → `dataset_docs_public.json`).
- Éval officielle : `uv run python -m moulinette evaluate_student_search_results ...`
- Cibles recall@5 : **≥ 80 % (docs)**, **≥ 50 % (code)**.

## Workflow conseillé

1. `uv run python testeur/check.py` → regarde les `KO` et la roadmap `[]`.
2. Code une fonction (commence par `index`).
3. Relance : les `[]`/`..` concernés passent au vert.
4. Recommence jusqu'à ce que tout soit `OK`.

## Note lint

Ce dossier est un outil de dev. Si `make lint` (flake8 `.`) le remonte et que
tu ne veux pas le nettoyer, exclus-le via `extend-exclude = testeur` dans la
config flake8, ou ajoute `testeur/` au `.gitignore`.

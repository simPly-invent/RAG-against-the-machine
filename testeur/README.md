# testeur/ — checklist exécutable (outil de dev, non soumis)

`check.py` vérifie **étape par étape** si le projet respecte le sujet.
On le relance après chaque avancée : tant qu'une case est ❌/KO, l'étape n'est
pas passée. Objectif = tout vert.

## Lancer

```bash
# Checklist rapide (structure, modèles Pydantic, présence des commandes CLI)
python testeur/check.py

# + checks lents : indexation réelle + un search réel
python testeur/check.py --full

# Une seule étape
python testeur/check.py --step models
python testeur/check.py --step structure
```

## Évaluer le recall@k (comme la moulinette, mais en local)

Une fois que `search_dataset` produit un JSON de résultats :

```bash
python testeur/check.py --step recall \
  --search-results data/output/search_results/dataset_docs_public.json \
  --ground-truth  datasets_public/public/AnsweredQuestions/dataset_docs_public.json \
  --kind docs      # compare recall@5 au seuil 80% (docs) / 50% (code)
```

Le calcul réplique la règle du sujet : une source est *found* si le
chevauchement de caractères ≥ 5 % ; score par question =
`nb_found / nb_sources_correctes`, moyenné sur les questions.

## Vérifier le format des réponses générées

```bash
python testeur/check.py --step answers \
  --answers data/output/search_results_and_answer/dataset_docs_public.json
```

## Légende

| Icône | Sens |
|-------|------|
| `OK`  | check passé |
| `KO`  | échec (si obligatoire → code de sortie 1) |
| `~~`  | avertissement (non bloquant) |
| `..`  | ignoré (check lent ou argument manquant) |

## Note lint

Ce dossier est un outil de dev. Si `make lint` (flake8 `.`) le remonte et que
tu ne veux pas le nettoyer, exclus-le via `extend-exclude = testeur` dans la
config flake8, ou ajoute `testeur/` au `.gitignore`.

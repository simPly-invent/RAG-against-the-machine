# Makefile — RAG against the machine
#
# Toutes les cibles imposées par le sujet (chap. III.2) :
#   install / run / debug / clean / lint  (+ lint-strict optionnel)
#
# Le module principal est `student`, invoqué via `uv run python -m student`
# comme dans les exemples du sujet.

# Module Python exposant la CLI Fire
MODULE := student

# Arguments passés à la CLI, ex : make run ARGS="index --max_chunk_size 2000"
ARGS ?=

.DEFAULT_GOAL := help

# Utilise bash pour un comportement homogène
SHELL := /bin/bash

.PHONY: help install run debug clean lint lint-strict test

## help : liste les cibles disponibles
help:
	@echo "Cibles disponibles :"
	@echo "  make install       Installe les dépendances (uv)"
	@echo "  make run ARGS=...   Lance la CLI          (uv run python3 -m $(MODULE) \$$ARGS)"
	@echo "  make debug ARGS=... Lance la CLI sous pdb"
	@echo "  make lint          flake8 + mypy (règles imposées)"
	@echo "  make lint-strict   flake8 + mypy --strict"
	@echo "  make test          Lance la suite de tests (pytest)"
	@echo "  make clean         Supprime caches et fichiers temporaires"

## install : installe les dépendances du projet via uv
install:
	uv sync

## run : exécute le script principal ; passer les arguments via ARGS=
run:
	uv run python -m $(MODULE) $(ARGS)

## debug : exécute le script principal sous le débogueur pdb
debug:
	uv run python -m pdb -m $(MODULE) $(ARGS)

## lint : flake8 + mypy avec les flags imposés par le sujet
lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

## lint-strict : version renforcée (optionnelle mais recommandée)
lint-strict:
	uv run flake8 .
	uv run mypy . --strict

## test : lance les tests unitaires (non notés, aide au dev)
test:
	uv run pytest -q

## clean : supprime les caches Python / mypy / pytest
clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	rm -rf .mypy_cache .pytest_cache .ruff_cache
	rm -rf data .venv

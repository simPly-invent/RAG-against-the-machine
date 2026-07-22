MODULE := student

ARGS ?=

.DEFAULT_GOAL := help

SHELL := /bin/bash

.PHONY: help install run debug clean lint lint-strict test

## help : comme le man je suis conscient que c'est pas habituelle mais j'aime bien comme ca 
help:
	@echo "Cibles disponibles :"
	@echo "  make install       Installe les dépendances (uv)"
	@echo "  make run ARGS=...   Lance la CLI          (uv run python3 -m $(MODULE) \$$ARGS)"
	@echo "  make debug ARGS=... Lance la CLI sous pdb"
	@echo "  make lint          flake8 + mypy (règles imposées)"
	@echo "  make lint-strict   flake8 + mypy --strict"
	@echo "  make test          Lance la suite de tests (pytest)"
	@echo "  make clean         Supprime caches et fichiers temporaires"

install:
	uv sync

run:
	uv run python -m $(MODULE) $(ARGS)

debug:
	uv run python -m pdb -m $(MODULE) $(ARGS)

lint:
	uv run flake8 .
	uv run mypy . --warn-return-any --warn-unused-ignores \
		--ignore-missing-imports --disallow-untyped-defs --check-untyped-defs


lint-strict:
	uv run flake8 .
	uv run mypy . --strict

test:
	uv run python src/student/__main__.py index
	uv run python src/student/__main__.py search --query "link between google and gemini"
	uv run python src/student/__main__.py search_dataset --dataset_path datasets_public/public/UnansweredQuestions/dataset_docs_public.json
	uv run python src/student/__main__.py answer --query "link between google and gemini"
	uv run python src/student/__main__.py answer_dataset --student_search_results_path data/output/search_results/dataset_docs_public.json
	uv run python -m student evaluate --dataset_path datasets_public/public/AnsweredQuestions/dataset_docs_public.json --search_results_path data/output/search_results/dataset_docs_public.json



clean:
	find . -type d -name '__pycache__' -prune -exec rm -rf {} +
	find . -type f -name '*.py[co]' -delete
	rm -rf .mypy_cache 
	rm -rf data/processed .venv data/output

#!/usr/bin/env python3
"""Testeur de progression — RAG against the machine.

Outil de DEV (non noté, non soumis) : vérifie étape par étape si le projet
respecte le sujet. On le relance après chaque avancée pour voir ce qui passe.

Usage :
    python testeur/check.py                 # checklist rapide (structure, modèles, CLI)
    python testeur/check.py --full          # + indexation + retrieval réels (lent)
    python testeur/check.py --step models   # une seule étape
    python testeur/check.py --recall \\
        --search-results data/output/search_results/dataset_docs_public.json \\
        --ground-truth datasets_public/public/AnsweredQuestions/dataset_docs_public.json \\
        --kind docs                         # évalue recall@k comme la moulinette

Codes de sortie : 0 si aucune étape obligatoire n'échoue, 1 sinon.
"""
from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

# --- Localisation du projet -------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
MODULE = "student"  # module CLI attendu par le sujet : `python -m student ...`

# Seuils de recall@5 imposés par le sujet (chap. VI.1.2)
RECALL5_DOCS_MIN = 0.80
RECALL5_CODE_MIN = 0.50
OVERLAP_MIN = 0.05  # 5% de chevauchement = source "found"

# --- Affichage --------------------------------------------------------------
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


PASS, FAIL, WARN, SKIP = "PASS", "FAIL", "WARN", "SKIP"
_ICON = {PASS: _c("32", "OK "), FAIL: _c("31", "KO "),
         WARN: _c("33", "~~ "), SKIP: _c("90", ".. ")}


@dataclass
class Result:
    status: str
    label: str
    detail: str = ""
    mandatory: bool = True


@dataclass
class Step:
    key: str
    title: str
    run: Callable[["Ctx"], list[Result]]


@dataclass
class Ctx:
    """Contexte partagé entre les checks (options CLI)."""
    full: bool = False
    args: argparse.Namespace = field(default_factory=argparse.Namespace)


# --- Utilitaires ------------------------------------------------------------
def run_cli(cli_args: list[str], timeout: int = 600) -> tuple[int, str, str, float]:
    """Lance `uv run python -m student <cli_args>` ; renvoie (rc, out, err, secondes)."""
    cmd = ["uv", "run", "python", "-m", MODULE, *cli_args]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=ROOT, capture_output=True,
                              text=True, timeout=timeout)
    except FileNotFoundError:
        return 127, "", "commande `uv` introuvable", 0.0
    except subprocess.TimeoutExpired:
        return 124, "", f"timeout > {timeout}s", float(timeout)
    return proc.returncode, proc.stdout, proc.stderr, time.perf_counter() - start


def import_models() -> Optional[object]:
    """Importe le module des modèles Pydantic, où qu'il soit."""
    for p in (str(SRC), str(ROOT)):
        if p not in sys.path:
            sys.path.insert(0, p)
    for name in (f"{MODULE}.models", f"{MODULE}", "models"):
        try:
            return importlib.import_module(name)
        except Exception:
            continue
    return None


def norm_path(path: str) -> str:
    """Normalise un file_path pour comparer output étudiant et ground truth.

    La vérité terrain préfixe les chemins par `data/raw/`. On coupe tout ce
    qui précède `vllm-0.10.1` pour comparer sur une base commune.
    """
    p = path.replace("\\", "/")
    marker = "vllm-0.10.1"
    idx = p.find(marker)
    return p[idx:] if idx != -1 else p.lstrip("./")


def overlap_ratio(r: dict, g: dict) -> float:
    """Fraction de la source correcte couverte par la source retrouvée."""
    if norm_path(r.get("file_path", "")) != norm_path(g.get("file_path", "")):
        return 0.0
    lo = max(int(r["first_character_index"]), int(g["first_character_index"]))
    hi = min(int(r["last_character_index"]), int(g["last_character_index"]))
    inter = max(0, hi - lo)
    span = max(1, int(g["last_character_index"]) - int(g["first_character_index"]))
    return inter / span


# ===========================================================================
# ÉTAPE 0 — Structure & outillage
# ===========================================================================
def step_structure(ctx: Ctx) -> list[Result]:
    out: list[Result] = []

    mk = ROOT / "Makefile"
    if mk.is_file():
        txt = mk.read_text(errors="ignore")
        missing = [t for t in ("install", "run", "debug", "clean", "lint")
                   if f"{t}:" not in txt]
        if missing:
            out.append(Result(FAIL, "Makefile : cibles", f"manquantes : {missing}"))
        else:
            out.append(Result(PASS, "Makefile : cibles install/run/debug/clean/lint"))
    else:
        out.append(Result(FAIL, "Makefile", "absent"))

    out.append(Result(PASS if (ROOT / "pyproject.toml").is_file() else FAIL,
                       "pyproject.toml"))
    out.append(Result(PASS if (ROOT / "uv.lock").is_file() else WARN,
                       "uv.lock", "généré par `uv sync`/`uv lock`", mandatory=False))

    pkg = None
    for cand in (SRC / MODULE, ROOT / MODULE):
        if (cand / "__init__.py").is_file():
            pkg = cand
            break
    if pkg is None:
        out.append(Result(FAIL, f"package `{MODULE}`",
                          f"attendu : src/{MODULE}/__init__.py"))
    else:
        rel = pkg.relative_to(ROOT)
        has_main = (pkg / "__main__.py").is_file()
        out.append(Result(PASS, f"package `{MODULE}` ({rel})"))
        out.append(Result(PASS if has_main else FAIL, f"{rel}/__main__.py",
                          "requis pour `python -m student`"))

    gi = ROOT / ".gitignore"
    if gi.is_file():
        g = gi.read_text(errors="ignore")
        need = [p for p in ("vllm-0.10.1", "__pycache__") if p not in g]
        out.append(Result(PASS if not need else WARN, ".gitignore",
                          f"ne couvre pas : {need}" if need else "", mandatory=False))
    else:
        out.append(Result(WARN, ".gitignore", "absent", mandatory=False))

    rd = ROOT / "README.md"
    if rd.is_file():
        first = rd.read_text(errors="ignore").strip().splitlines()
        first_line = first[0] if first else ""
        ok_first = first_line.startswith("*") and "42 curriculum" in first_line
        out.append(Result(PASS if ok_first else WARN, "README : 1re ligne italique 42",
                          "" if ok_first else "format attendu : *...42 curriculum by <login>*",
                          mandatory=False))
    else:
        out.append(Result(WARN, "README.md", "absent (requis au livrable)",
                          mandatory=False))
    return out


# ===========================================================================
# ÉTAPE 1 — Modèles Pydantic
# ===========================================================================
def step_models(ctx: Ctx) -> list[Result]:
    mod = import_models()
    if mod is None:
        return [Result(FAIL, "import des modèles Pydantic",
                      f"module `{MODULE}.models` introuvable")]
    expected = {
        "MinimalSource": ["file_path", "first_character_index", "last_character_index"],
        "UnansweredQuestion": ["question_id", "question"],
        "AnsweredQuestion": ["sources", "answer"],
        "RagDataset": ["rag_questions"],
        "MinimalSearchResults": ["question_id", "question", "retrieved_sources"],
        "MinimalAnswer": ["answer"],
        "StudentSearchResults": ["search_results", "k"],
        "StudentSearchResultsAndAnswer": ["search_results"],
    }
    out: list[Result] = []
    for cls_name, fields in expected.items():
        cls = getattr(mod, cls_name, None)
        if cls is None:
            out.append(Result(FAIL, f"modèle {cls_name}", "classe absente"))
            continue
        model_fields = getattr(cls, "model_fields", None)
        if model_fields is None:
            out.append(Result(WARN, f"modèle {cls_name}",
                              "n'est pas un BaseModel pydantic v2", mandatory=False))
            continue
        miss = [f for f in fields if f not in model_fields]
        out.append(Result(PASS if not miss else FAIL, f"modèle {cls_name}",
                          f"champs manquants : {miss}" if miss else ""))
    return out


# ===========================================================================
# ÉTAPE 2 — CLI Fire (présence des commandes)
# ===========================================================================
def step_cli(ctx: Ctx) -> list[Result]:
    rc, out_s, err_s, _ = run_cli(["--", "--help"], timeout=120)
    blob = (out_s + err_s).lower()
    if rc == 127:
        return [Result(FAIL, "CLI : `uv run python -m student`", err_s)]
    if rc == 124:
        return [Result(FAIL, "CLI : help", "timeout")]
    cmds = ["index", "search", "search_dataset", "answer",
            "answer_dataset", "evaluate"]
    res: list[Result] = []
    for cmd in cmds:
        res.append(Result(PASS if cmd in blob else FAIL, f"commande `{cmd}`"))
    return res


# ===========================================================================
# ÉTAPE 3 — Indexation (lent : --full)
# ===========================================================================
def step_index(ctx: Ctx) -> list[Result]:
    if not ctx.full:
        return [Result(SKIP, "indexation réelle", "relance avec --full", mandatory=False)]
    rc, _o, err, secs = run_cli(["index", "--max_chunk_size", "2000"], timeout=600)
    if rc != 0:
        return [Result(FAIL, "index : exécution", err.strip()[-300:] or f"rc={rc}")]
    res = [Result(PASS, "index : exécution sans crash")]
    res.append(Result(PASS if secs <= 300 else FAIL,
                      f"index : durée {secs:.0f}s", "seuil 300s (5 min)"))
    return res


# ===========================================================================
# ÉTAPE 4 — Format d'un search (validité Pydantic + indices)
# ===========================================================================
def step_search_format(ctx: Ctx) -> list[Result]:
    if not ctx.full:
        return [Result(SKIP, "format de search", "relance avec --full", mandatory=False)]
    rc, out_s, err, _ = run_cli(
        ["search", "How to configure the OpenAI server?", "--k", "5"], timeout=120)
    if rc != 0:
        return [Result(FAIL, "search : exécution", err.strip()[-300:] or f"rc={rc}")]
    try:
        data = json.loads(out_s[out_s.index("{"):out_s.rindex("}") + 1])
    except Exception:
        return [Result(WARN, "search : sortie JSON",
                      "pas de JSON parsable en stdout", mandatory=False)]
    srcs = _first_sources(data)
    if srcs is None:
        return [Result(WARN, "search : structure", "pas de retrieved_sources",
                      mandatory=False)]
    bad = [s for s in srcs
           if not {"file_path", "first_character_index", "last_character_index"} <= set(s)
           or int(s["last_character_index"]) < int(s["first_character_index"])]
    return [Result(PASS if not bad else FAIL, "search : sources bien formées",
                  f"{len(bad)} source(s) invalide(s)" if bad else
                  f"{len(srcs)} sources OK")]


def _first_sources(data: dict) -> Optional[list]:
    if "retrieved_sources" in data:
        return data["retrieved_sources"]
    sr = data.get("search_results")
    if isinstance(sr, list) and sr and isinstance(sr[0], dict):
        return sr[0].get("retrieved_sources")
    return None


# ===========================================================================
# ÉTAPE 5 — Recall@k (évaluation type moulinette, indépendante)
# ===========================================================================
def evaluate_recall(search_results_path: Path, ground_truth_path: Path,
                    ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[int, float]:
    """Réplique le recall@k du sujet : found si overlap >= 5%, score par
    question = number_found / total_correct_sources, moyenné sur les questions.
    """
    sr = json.loads(search_results_path.read_text())
    gt = json.loads(ground_truth_path.read_text())
    student = {r["question_id"]: r.get("retrieved_sources", [])
               for r in sr.get("search_results", [])}
    questions = gt.get("rag_questions", gt.get("questions", []))
    scores: dict[int, list[float]] = {k: [] for k in ks}
    for q in questions:
        correct = q.get("sources", [])
        qid = q.get("question_id")
        retrieved = student.get(qid, [])
        if not correct:
            continue
        for k in ks:
            topk = retrieved[:k]
            found = sum(1 for g in correct
                        if any(overlap_ratio(r, g) >= OVERLAP_MIN for r in topk))
            scores[k].append(found / len(correct))
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in scores.items()}


def step_recall(ctx: Ctx) -> list[Result]:
    a = ctx.args
    if not getattr(a, "search_results", None) or not getattr(a, "ground_truth", None):
        return [Result(SKIP, "recall@k",
                      "fournir --search-results et --ground-truth", mandatory=False)]
    srp, gtp = Path(a.search_results), Path(a.ground_truth)
    if not srp.is_file() or not gtp.is_file():
        return [Result(FAIL, "recall@k", "fichier(s) introuvable(s)")]
    try:
        rec = evaluate_recall(srp, gtp)
    except Exception as exc:  # noqa: BLE001
        return [Result(FAIL, "recall@k : calcul", str(exc))]
    out = [Result(PASS, f"recall@{k} = {rec[k]:.3f}", mandatory=False)
           for k in sorted(rec)]
    kind = getattr(a, "kind", None)
    if kind in ("docs", "code"):
        thr = RECALL5_DOCS_MIN if kind == "docs" else RECALL5_CODE_MIN
        ok = rec.get(5, 0.0) >= thr
        out.append(Result(PASS if ok else FAIL,
                          f"recall@5 {kind} >= {thr:.0%}",
                          f"obtenu {rec.get(5, 0.0):.3f}"))
    return out


# ===========================================================================
# ÉTAPE 6 — Format des réponses générées
# ===========================================================================
def step_answer_format(ctx: Ctx) -> list[Result]:
    a = ctx.args
    path = getattr(a, "answers", None)
    if not path:
        return [Result(SKIP, "format des réponses",
                      "fournir --answers <json>", mandatory=False)]
    p = Path(path)
    if not p.is_file():
        return [Result(FAIL, "answers", "fichier introuvable")]
    try:
        data = json.loads(p.read_text())
        rows = data["search_results"]
    except Exception as exc:  # noqa: BLE001
        return [Result(FAIL, "answers : JSON", str(exc))]
    empty = sum(1 for r in rows if not str(r.get("answer", "")).strip())
    res = [Result(PASS if empty == 0 else WARN,
                 f"answers : {len(rows)} réponses",
                 f"{empty} vide(s)" if empty else "", mandatory=False)]
    return res


# ===========================================================================
STEPS: list[Step] = [
    Step("structure", "Étape 0 — Structure & outillage", step_structure),
    Step("models", "Étape 1 — Modèles Pydantic", step_models),
    Step("cli", "Étape 2 — CLI Fire (commandes)", step_cli),
    Step("index", "Étape 3 — Indexation (--full)", step_index),
    Step("search", "Étape 4 — Format d'un search (--full)", step_search_format),
    Step("recall", "Étape 5 — Recall@k", step_recall),
    Step("answers", "Étape 6 — Format des réponses", step_answer_format),
]


def main() -> int:
    ap = argparse.ArgumentParser(description="Testeur de progression du projet RAG.")
    ap.add_argument("--full", action="store_true",
                    help="exécute aussi les checks lents (index, search réels)")
    ap.add_argument("--step", choices=[s.key for s in STEPS],
                    help="ne lancer qu'une étape")
    ap.add_argument("--search-results", help="JSON de search_dataset (étape recall)")
    ap.add_argument("--ground-truth", help="JSON AnsweredQuestions (étape recall)")
    ap.add_argument("--kind", choices=["docs", "code"],
                    help="type de dataset pour comparer aux seuils recall@5")
    ap.add_argument("--answers", help="JSON de answer_dataset (étape answers)")
    args = ap.parse_args()

    ctx = Ctx(full=args.full, args=args)
    steps = [s for s in STEPS if not args.step or s.key == args.step]

    total = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0}
    hard_fail = False
    print(_c("1", "\n=== Testeur RAG — état d'avancement ===\n"))
    for st in steps:
        print(_c("1;36", st.title))
        for r in st.run(ctx):
            total[r.status] += 1
            if r.status == FAIL and r.mandatory:
                hard_fail = True
            line = f"  {_ICON[r.status]} {r.label}"
            if r.detail:
                line += _c("90", f"  — {r.detail}")
            print(line)
        print()

    print(_c("1", "--- Bilan ---"))
    print(f"  {_c('32', 'OK')}={total[PASS]}  "
          f"{_c('31', 'KO')}={total[FAIL]}  "
          f"{_c('33', '~~')}={total[WARN]}  "
          f"{_c('90', '..')}={total[SKIP]}")
    verdict = ("étape(s) obligatoire(s) en échec" if hard_fail
               else "aucune étape obligatoire en échec")
    print(f"  Verdict : {_c('31', verdict) if hard_fail else _c('32', verdict)}\n")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())

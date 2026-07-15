#!/usr/bin/env python3
"""Testeur de progression — RAG against the machine.

Outil de DEV (non noté, non soumis) : vérifie étape par étape si le projet
respecte le sujet. On le relance après chaque avancée pour voir ce qui passe.


lance le make file avec tester-strict normalement ca devrait faire full
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

# Arborescence attendue par la moulinette (chap. V.6)
PROCESSED = ROOT / "data/processed"                       # bm25_index/ + chunks/
OUT_SEARCH = ROOT / "data/output/search_results"          # sorties search_dataset
OUT_ANSWER = ROOT / "data/output/search_results_and_answer"  # sorties answer_dataset
# Le nom du fichier de sortie doit == le nom du dataset d'entrée
DATASET_NAMES = {"dataset_docs_public.json", "dataset_code_public.json"}

# Seuils de recall@5 imposés par le sujet (chap. VI.1.2)
RECALL5_DOCS_MIN = 0.80
RECALL5_CODE_MIN = 0.50
OVERLAP_MIN = 0.05  # 5% de chevauchement = source "found"

# --- Affichage --------------------------------------------------------------
_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text


PASS, FAIL, WARN, SKIP, TODO = "PASS", "FAIL", "WARN", "SKIP", "TODO"
_ICON = {PASS: _c("32", "OK "), FAIL: _c("31", "KO "),
         WARN: _c("33", "~~ "), SKIP: _c("90", ".. "),
         TODO: _c("35", "[] ")}


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
    for name in (f"{MODULE}.pymodel", f"{MODULE}", "model"):
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


def _validate_files(files: list[Path], model_name: str, label: str) -> list[Result]:
    """Valide chaque JSON contre un modèle Pydantic imposé."""
    mod = import_models()
    model = getattr(mod, model_name, None) if mod else None
    if model is None:
        return [Result(WARN, f"{label} : modèle {model_name}",
                      "modèle introuvable", mandatory=False)]
    bad = None
    for p in files:
        ok, res = safe(lambda p=p: model.model_validate(json.loads(p.read_text())))
        if not ok:
            bad = f"{p.name} : {str(res)[:120]}"
            break
    return [Result(PASS if bad is None else FAIL,
                  f"{label} : conforme à {model_name}",
                  bad or f"{len(files)} fichier(s) OK", mandatory=False)]


def safe(fn: Callable[[], object]) -> tuple[bool, object]:
    """Exécute fn sans planter -> (succès, résultat_ou_exception)."""
    try:
        return True, fn()
    except Exception as exc:  # noqa: BLE001
        return False, exc


def _check_output_dir(dirpath: Path, model_name: str, label: str,
                      check_k: bool = False) -> list[Result]:
    """Vérifie un dossier de sortie : présence, nom == dataset, conformité modèle."""
    if not dirpath.exists():
        return [Result(SKIP, label,
                      f"{dirpath.relative_to(ROOT)} absent (pas encore produit)",
                      mandatory=False)]
    files = list(dirpath.rglob("*.json"))
    if not files:
        return [Result(SKIP, label, "aucun .json produit", mandatory=False)]

    out: list[Result] = []
    names = {p.name for p in files}
    good = names & DATASET_NAMES
    out.append(Result(PASS if good else WARN, f"{label} : nom == dataset d'entrée",
                      f"ok : {sorted(good)}" if good else f"noms inattendus : {sorted(names)}",
                      mandatory=False))
    out += _validate_files(files, model_name, label)
    if check_k:
        missing = [p.name for p in files
                   if "k" not in (json.loads(p.read_text()) if p.is_file() else {})]
        out.append(Result(PASS if not missing else WARN, f"{label} : champ k présent",
                          f"absent dans : {missing}" if missing else "", mandatory=False))
    return out


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
    proc_ok = PROCESSED.exists() and any(p.is_file() for p in PROCESSED.rglob("*"))
    res: list[Result] = [Result(
        PASS if proc_ok else SKIP, "index : artefacts data/processed/",
        "présents" if proc_ok else "pas encore générés (bm25_index/, chunks/)",
        mandatory=False)]
    if not ctx.full:
        res.append(Result(SKIP, "indexation réelle", "relance avec --full",
                          mandatory=False))
        return res
    rc, _o, err, secs = run_cli(["index", "--max_chunk_size", "2000"], timeout=600)
    if rc != 0:
        res.append(Result(FAIL, "index : exécution", err.strip()[-300:] or f"rc={rc}"))
        return res
    res.append(Result(PASS, "index : exécution sans crash"))
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
    data = _extract_json(out_s)
    if data is None:
        return [Result(WARN, "search : sortie JSON",
                      "pas de JSON parsable en stdout", mandatory=False)]
    srcs = _first_sources(data)
    if srcs is None:
        return [Result(WARN, "search : structure",
                      "attendu : liste de MinimalSource (3 champs)",
                      mandatory=False)]
    bad = [s for s in srcs
           if not {"file_path", "first_character_index", "last_character_index"} <= set(s)
           or int(s["last_character_index"]) < int(s["first_character_index"])]
    return [Result(PASS if not bad else FAIL, "search : sources bien formées",
                  f"{len(bad)} source(s) invalide(s)" if bad else
                  f"{len(srcs)} sources OK")]


def _extract_json(out_s: str):
    """Récupère le JSON de stdout, en document unique OU en JSON-lines.

    `search_dataset` émet un objet StudentSearchResults `{...}`. `search`
    retourne une liste de MinimalSource, et Fire imprime alors **un objet par
    ligne** (JSON-lines) : le slice « premier `{` → dernier `}` » avalerait
    toutes les lignes d'un coup et échouerait. D'où le fallback ligne à ligne.
    """
    starts = [i for i in (out_s.find("["), out_s.find("{")) if i != -1]
    if not starts:
        return None
    start = min(starts)
    close = "]" if out_s[start] == "[" else "}"
    end = out_s.rfind(close)
    if end >= start:
        try:
            return json.loads(out_s[start:end + 1])
        except Exception:
            pass

    docs = []
    for line in out_s.splitlines():
        line = line.strip()
        if not line or line[0] not in "[{":
            continue
        try:
            docs.append(json.loads(line))
        except Exception:
            return None
    return docs or None


def _first_sources(data) -> Optional[list]:
    # search : la valeur de retour EST déjà la liste de MinimalSource.
    if isinstance(data, list):
        return data
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
# ÉTAPE — search_dataset : sortie batch conforme
# ===========================================================================
def step_search_dataset(ctx: Ctx) -> list[Result]:
    return _check_output_dir(OUT_SEARCH, "StudentSearchResults",
                             "search_dataset", check_k=True)


# ===========================================================================
# ÉTAPE — answer_dataset : réponses batch conformes
# ===========================================================================
def step_answer_dataset(ctx: Ctx) -> list[Result]:
    a = ctx.args
    path = getattr(a, "answers", None)
    if path:  # fichier explicite fourni en option
        p = Path(path)
        if not p.is_file():
            return [Result(FAIL, "answers", "fichier introuvable")]
        files = [p]
    else:  # sinon, dossier standard du sujet
        if not OUT_ANSWER.exists():
            return [Result(SKIP, "answer_dataset",
                          f"{OUT_ANSWER.relative_to(ROOT)} absent (pas encore produit)",
                          mandatory=False)]
        files = list(OUT_ANSWER.rglob("*.json"))
        if not files:
            return [Result(SKIP, "answer_dataset", "aucun .json produit",
                          mandatory=False)]

    out = _validate_files(files, "StudentSearchResultsAndAnswer", "answer_dataset")
    for p in files:
        ok, data = safe(lambda p=p: json.loads(p.read_text()))
        rows = data.get("search_results", []) if ok and isinstance(data, dict) else []
        empty = sum(1 for r in rows if not str(r.get("answer", "")).strip())
        out.append(Result(PASS if empty == 0 else WARN,
                          f"answer_dataset : {p.name} ({len(rows)} réponses)",
                          f"{empty} vide(s)" if empty else "", mandatory=False))
    return out


# ===========================================================================
# ÉTAPE — Auto-test de la métrique (valide le testeur lui-même)
# ===========================================================================
def step_selftest(ctx: Ctx) -> list[Result]:
    def src(fp: str, a: int, b: int) -> dict:
        return {"file_path": fp, "first_character_index": a,
                "last_character_index": b}
    cases = [
        ("overlap identique = 1.0",
         overlap_ratio(src("a.py", 0, 100), src("a.py", 0, 100)) == 1.0),
        ("fichier différent = 0.0",
         overlap_ratio(src("a.py", 0, 100), src("b.py", 0, 100)) == 0.0),
        ("préfixe data/raw ignoré",
         overlap_ratio(src("vllm-0.10.1/x.py", 0, 100),
                       src("data/raw/vllm-0.10.1/x.py", 0, 100)) == 1.0),
        ("overlap 2% < seuil 5%",
         overlap_ratio(src("a.py", 98, 100), src("a.py", 0, 100)) < OVERLAP_MIN),
    ]
    return [Result(PASS if ok else FAIL, f"métrique : {label}") for label, ok in cases]


# ===========================================================================
# ÉTAPE — Conception : décomposition des fonctions, vérifiée automatiquement
# ===========================================================================
CHUNKS_JSON = PROCESSED / "chunks" / "chunks.json"
MAX_CHUNK = 2000
_SRC_CACHE: dict[str, str] = {}


def _src_text() -> str:
    """Concatène le code du package `student` (pour repérer tqdm, bm25, Qwen...)."""
    if "all" not in _SRC_CACHE:
        parts = []
        for p in (SRC / MODULE).rglob("*.py"):
            ok, txt = safe(lambda p=p: p.read_text(encoding="utf-8", errors="ignore"))
            if ok:
                parts.append(str(txt))
        _SRC_CACHE["all"] = "\n".join(parts).lower()
    return _SRC_CACHE["all"]


def _uses(needle: str) -> bool:
    return needle.lower() in _src_text()


def _read_chunks() -> tuple[Optional[list], str]:
    """Charge chunks.json (avec garde-fou de taille) -> (data, message d'erreur)."""
    if not CHUNKS_JSON.is_file():
        return None, "data/processed/chunks.json absent — lance `index`"
    size = CHUNKS_JSON.stat().st_size
    if size > 400_000_000:
        return None, f"chunks.json trop gros ({size / 1e6:.0f} Mo) pour l'analyse"
    ok, data = safe(lambda: json.loads(CHUNKS_JSON.read_text(encoding="utf-8")))
    if not ok:
        return None, f"JSON invalide : {str(data)[:80]}"
    return data if isinstance(data, list) else None, (
        "" if isinstance(data, list) else "la racine n'est pas une liste")


def _flatten(data: list) -> tuple[list, bool]:
    """Aplatit une liste de listes ; renvoie (chunks, était_imbriqué)."""
    if data and isinstance(data[0], list):
        return [c for sub in data for c in sub], True
    return data, False


def _design_index(out: list[Result]) -> None:
    data, err = _read_chunks()
    if data is None:
        out.append(Result(SKIP, "[index] chunks.json analysable", err, mandatory=False))
        return

    chunks, nested = _flatten(data)
    out.append(Result(
        FAIL if nested else PASS, "[index] chunks.json = liste plate de chunks",
        "liste de listes ! utilise `resized += chunk_text(...)` (ou .extend) "
        "au lieu de .append" if nested else f"{len(chunks)} chunks",
        mandatory=False))

    dicts = [c for c in chunks if isinstance(c, dict)]
    if not dicts:
        out.append(Result(FAIL, "[index] chunks = dictionnaires", "aucun dict trouvé",
                          mandatory=False))
        return

    need = {"file_path", "first_character_index", "last_character_index"}
    keys_ok = all(need <= set(c) for c in dicts)
    out.append(Result(PASS if keys_ok else FAIL,
                      "[index] chaque chunk garde file_path + first/last",
                      "" if keys_ok else f"clés attendues : {sorted(need)}",
                      mandatory=False))

    exts = {Path(str(c.get("file_path", ""))).suffix for c in dicts}
    bad = exts - {".py", ".md"}
    out.append(Result(PASS if not bad else WARN, "[index] fichiers filtrés (.py, .md)",
                      f"autres extensions : {sorted(bad)[:6]}" if bad else "",
                      mandatory=False))

    prefixed = all("vllm-0.10.1" in str(c.get("file_path", "")) for c in dicts)
    out.append(Result(PASS if prefixed else FAIL,
                      "[index] file_path comparable à la ground truth",
                      "" if prefixed else "doit contenir vllm-0.10.1/...",
                      mandatory=False))

    if keys_ok:
        spans = [c["last_character_index"] - c["first_character_index"] for c in dicts]
        too_big = [s for s in spans if s > MAX_CHUNK]
        out.append(Result(PASS if not too_big else FAIL,
                          f"[index] respecte max_chunk_size ({MAX_CHUNK})",
                          f"{len(too_big)} chunk(s) trop grands (max {max(spans)})"
                          if too_big else f"max observé : {max(spans)}",
                          mandatory=False))

        coherent = all(0 <= c["first_character_index"] < c["last_character_index"]
                       for c in dicts)
        out.append(Result(PASS if coherent else FAIL,
                          "[index] indices cohérents (0 <= first < last)",
                          mandatory=False))

        # Heuristique : si presque tous les spans valent pile MAX_CHUNK,
        # c'est une fenêtre fixe -> le chunking "intelligent" reste à faire.
        exact = sum(1 for s in spans if s == MAX_CHUNK)
        fixed = exact > 0.5 * len(spans)
        out.append(Result(TODO if fixed else PASS,
                          "[index] chunking intelligent (Python: def/class, MD: sections)",
                          "fenêtre fixe détectée — à raffiner pour le recall"
                          if fixed else "découpage non uniforme (bon signe)",
                          mandatory=False))

    bm25 = any(PROCESSED.glob("*bm25*")) or _uses("bm25") or _uses("tfidf")
    out.append(Result(PASS if bm25 else TODO,
                      "[index] index BM25 (ou TF-IDF) construit et persisté",
                      "" if bm25 else "rien dans data/processed/, ni dans le code",
                      mandatory=False))

    out.append(Result(PASS if _uses("tqdm") else TODO,
                      "[index] barre de progression tqdm", mandatory=False))


def _func_src(name: str) -> str:
    """Source de LA fonction CLI demandée (pas de tout le package)."""
    key = f"fn:{name}"
    if key not in _SRC_CACHE:
        import inspect
        for p in (str(SRC), str(ROOT)):
            if p not in sys.path:
                sys.path.insert(0, p)
        ok, mod = safe(lambda: importlib.import_module(f"{MODULE}.CLI"))
        fn = getattr(mod, name, None) if ok else None
        ok2, src = safe(lambda: inspect.getsource(fn)) if fn else (False, "")
        _SRC_CACHE[key] = str(src).lower() if ok2 else ""
    return _SRC_CACHE[key]


def _is_stub(name: str) -> bool:
    """Un stub = un corps d'une seule instruction (`return x`)."""
    lines = [ln.strip() for ln in _func_src(name).splitlines()[1:]
             if ln.strip() and not ln.strip().startswith("#")]
    return len(lines) <= 1


_PROBES: dict[str, list[tuple[str, Callable[[str], bool]]]] = {
    "search": [
        ("charge l'index depuis data/processed/",
         lambda s: "chunks.json" in s or "processed" in s),
        ("scorer et prendre le top-k (--k)",
         lambda s: "score" in s or "top" in s or "sort" in s),
        ("mapper chaque hit -> MinimalSource", lambda s: "minimalsource" in s),
        ("gérer les cas dégénérés (query vide, index absent)",
         lambda s: "except" in s),
    ],
    "answer": [
        ("faire un search pour récupérer le contexte", lambda s: "search" in s),
        ("charger Qwen/Qwen3-0.6B", lambda s: "qwen" in s),
        ("construire le prompt (contexte + question)", lambda s: "prompt" in s),
    ],
    "evaluate": [
        ("normaliser les file_path", lambda s: "vllm-0.10.1" in s or "norm" in s),
        ("overlap de caractères >= 5%", lambda s: "0.05" in s or "overlap" in s),
        ("recall@k agrégé", lambda s: "recall" in s),
    ],
}


def _design_rest(out: list[Result]) -> None:
    for func, checks in _PROBES.items():
        stub = _is_stub(func)
        for i, (label, pred) in enumerate(checks):
            done = (not stub) and pred(_func_src(func))
            note = "fonction encore vide (stub)" if stub and i == 0 else ""
            out.append(Result(PASS if done else TODO, f"[{func}] {label}", note,
                              mandatory=False))


def step_design(ctx: Ctx) -> list[Result]:
    out: list[Result] = []
    _design_index(out)
    _design_rest(out)
    return out


# ===========================================================================
STEPS: list[Step] = [
    Step("structure", "Étape 0 — Structure & outillage", step_structure),
    Step("models", "Étape 1 — Modèles Pydantic", step_models),
    Step("selftest", "Étape 1b — Auto-test de la métrique", step_selftest),
    Step("cli", "Étape 2 — CLI Fire (commandes)", step_cli),
    Step("design", "Conception — fonctions à coder (roadmap)", step_design),
    Step("index", "Étape 3 — Indexation (artefacts + --full)", step_index),
    Step("search", "Étape 4 — Format d'un search (--full)", step_search_format),
    Step("search_dataset", "Étape 4b — search_dataset (sortie batch)",
         step_search_dataset),
    Step("recall", "Étape 5 — Recall@k", step_recall),
    Step("answers", "Étape 6 — answer_dataset (réponses batch)",
         step_answer_dataset),
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

    total = {PASS: 0, FAIL: 0, WARN: 0, SKIP: 0, TODO: 0}
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
          f"{_c('90', '..')}={total[SKIP]}  "
          f"{_c('35', '[]')}={total[TODO]}")
    verdict = ("étape(s) obligatoire(s) en échec" if hard_fail
               else "aucune étape obligatoire en échec")
    (f"  Verdict : {_c('31', verdict) if hard_fail else _c('32', verdict)}\n")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    sys.exit(main())

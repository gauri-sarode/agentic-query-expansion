"""EXPLORATORY partial-intervention probes (configs/partial_probe_protocol.json).

  time     latency per query on 50 validation queries per corpus: probe(k),
           remaining(100-k), full(100), and the 8 t0 features (TripClick, NQ;
           ArguAna's from docs/arguana_feature_latency.txt). Label-free.
           Writes results/probes/latency.json.
  analyze  fits probe models on validation (ArguAna: train/val), scores test
           once, builds gain-vs-cost curves and the registered comparisons.
           Writes docs/partial_probe_summary.json (rounded aggregates).

Usage: PYTHONPATH=. python scripts/60_partial_probe.py {time|analyze}
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from src.data import load_queries  # noqa: E402

PROBES = Path("results/probes")
SIZES = (5, 10, 20)
PRIMARY = 10
BUDGETS = (0.0, 0.10, 0.25, 0.50, 0.75, 1.00)
TEST_BUDGETS = (0.10, 0.25, 0.50, 0.75)
FULL = 100


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _h(q: str, salt: str = "nq57") -> str:
    return hashlib.sha256(f"{q}:{salt}".encode()).hexdigest()


# ---------------------------------------------------------------- latency

def time_corpus(corpus: str) -> dict:
    from src.observability.telemetry import compute_retrieval_telemetry
    from src.retrieval import bm25_index
    from src.retrieval.rerank import _model

    iv48 = _load("iv48", "scripts/48_intervention_runs.py")
    qids, queries, db = iv48.setup(corpus, "val")
    bm = json.loads(Path(f"results/interventions/{corpus}_val_bm25.json").read_text())
    sample = sorted(qids, key=_h)[:50]
    model = _model()

    def ce(q, ids):
        texts = bm25_index.get_texts(ids, db)
        model.predict([(queries[q], texts[d]) for d in ids if d in texts])

    pipes = {f"probe_{k}": (lambda q, k=k: ce(q, [d for d, _ in bm[q][:k]])) for k in SIZES}
    pipes |= {f"rest_{k}": (lambda q, k=k: ce(q, [d for d, _ in bm[q][k:FULL]])) for k in SIZES}
    pipes["full"] = lambda q: ce(q, [d for d, _ in bm[q][:FULL]])
    pipes["telemetry"] = lambda q: compute_retrieval_telemetry(queries[q], [tuple(x) for x in bm[q][:FULL]], db)
    for q in sample[:3]:  # warm caches and the model
        for f in pipes.values():
            f(q)
    names, t = list(pipes), {n: [] for n in pipes}
    for i, q in enumerate(sample):
        for n in names[i % len(names):] + names[: i % len(names)]:  # rotate order per query
            s = time.perf_counter()
            pipes[n](q)
            t[n].append(time.perf_counter() - s)
    return {n: {"median_s": float(np.median(v)), "p95_s": float(np.percentile(v, 95))} for n, v in t.items()}


def time_all() -> None:
    out = {"_note": "Medians over 50 validation queries per corpus, M4 16 GB, warm caches, rotated pipeline order."}
    for corpus in ("nq", "tripclick"):
        out[corpus] = time_corpus(corpus)
        print(corpus, json.dumps(out[corpus]), flush=True)
    # ArguAna: CE timings measured the same way on its validation split; telemetry from the frozen file.
    out["arguana"] = time_arguana()
    print("arguana", json.dumps(out["arguana"]), flush=True)
    (PROBES / "latency.json").write_text(json.dumps(out, indent=2))


def time_arguana() -> dict:
    from src.retrieval import bm25_index
    from src.retrieval.rerank import _model

    split = json.loads(Path("configs/arguana_split.json").read_text())
    val = split["val"] if isinstance(split.get("val"), list) else [q for q, s in split["assignment"].items() if s == "val"]
    bm = {q: v["ranking"] for q, v in json.loads(Path("results/arguana/bm25.json").read_text()).items()}
    queries, db, model = load_queries("arguana"), "data/arguana_bm25.sqlite3", _model()
    sample = sorted(val, key=_h)[:50]

    def ce(q, ids):
        texts = bm25_index.get_texts(ids, db)
        model.predict([(queries[q], texts[d]) for d in ids if d in texts])

    pipes = {f"probe_{k}": (lambda q, k=k: ce(q, [d for d, _ in bm[q][:k]])) for k in SIZES}
    pipes |= {f"rest_{k}": (lambda q, k=k: ce(q, [d for d, _ in bm[q][k:FULL]])) for k in SIZES}
    pipes["full"] = lambda q: ce(q, [d for d, _ in bm[q][:FULL]])
    for q in sample[:3]:
        for f in pipes.values():
            f(q)
    names, t = list(pipes), {n: [] for n in pipes}
    for i, q in enumerate(sample):
        for n in names[i % len(names):] + names[: i % len(names)]:
            s = time.perf_counter()
            pipes[n](q)
            t[n].append(time.perf_counter() - s)
    out = {n: {"median_s": float(np.median(v)), "p95_s": float(np.percentile(v, 95))} for n, v in t.items()}
    line = [x for x in Path("docs/arguana_feature_latency.txt").read_text().splitlines() if x.startswith("telemetry")][0]
    out["telemetry"] = {"median_s": float(line.split()[2]) / 1000, "source": "docs/arguana_feature_latency.txt (test)"}
    return out


# ---------------------------------------------------------------- features

def probe_features(bm_rank: list, probe: list, k: int, nqc: float, qlen: float) -> list[float]:
    s = np.array([x for _, x in probe[:k]], dtype=float)
    if len(s) == 0:
        return [0.0] * 7 + [nqc, qlen]
    first = s[0]
    rho = spearmanr(np.arange(len(s)), -s).statistic if len(s) > 2 and s.std() > 0 else 0.0
    return [s.max(), s.mean(), s.std(), first, s.max() - first, float(np.argmax(s)), float(np.nan_to_num(rho)), nqc, qlen]


def corpus_data(corpus: str) -> dict:
    """val (fit pool) and test: Delta, F, probe features per k, comparator scores."""
    da = _load("da50", "scripts/50_decision_analyses.py")
    rn = _load("rn54", "scripts/54_baseline_runner.py")
    if corpus == "arguana":
        fr = pd.read_csv("results/arguana/arguana_frozen.csv").set_index("query_id")
        pr = pd.read_csv("results/arguana/runner/test_predictions.csv")
        pr = pr[pr.action == "rerank"].set_index("query_id")
        probes = json.loads((PROBES / "arguana_all.json").read_text())
        bm = {q: v["ranking"] for q, v in json.loads(Path("results/arguana/bm25.json").read_text()).items()}
        out = {}
        for sp in ("train", "val", "test"):
            ids = [q for q in fr.index if fr.loc[q, "split"] == sp]
            if sp == "test":
                ids = list(pr.index)
            f = fr.loc[ids]
            out[sp] = {"qids": ids, "delta": (f.rerank_ndcg - f.baseline_ndcg).to_numpy(),
                       "F": (f.baseline_ndcg <= 1e-12).to_numpy(),
                       "X": {k: np.array([probe_features(bm[q], probes[q], k, f.loc[q, "z_nqc"], f.loc[q, "z_qlen"])
                                          for q in ids]) for k in SIZES}}
        T = out["test"]
        assert np.allclose(T["delta"], pr["delta"].to_numpy())
        T["comparators"] = {"failure": pr["failure"].to_numpy(), "nqc": pr["nqc"].to_numpy(),
                            "B1_gbdt": pr["B1_gbdt"].to_numpy()}
        return {"fit": out["train"], "select": out["val"], "test": T}

    iv = da.iv
    inp = iv.corpus_inputs(corpus)
    queries = load_queries("nq") if corpus == "nq" else None
    S = {s["name"]: s for s in da.intervention_settings()}[f"rerank_{corpus}"]
    parts = {}
    for sp in ("val", "test"):
        qids, qrels = inp[sp]["qids"], inp[sp]["qrels"]
        bm = iv._runs(corpus, sp, "bm25")
        rr = iv._runs(corpus, sp, "rerank")
        b = iv.ndcg(qids, {q: [x for x, _ in bm[q]] for q in qids}, qrels)
        d = iv.ndcg(qids, {q: [x for x, _ in rr[q]] for q in qids}, qrels) - b
        probes = json.loads((PROBES / f"{corpus}_{sp}.json").read_text())
        if queries is None:
            queries = load_queries("tripclick-tail-val" if sp == "val" else "tripclick-tail")
        nq_ = np.array([da.nqc(bm[q]) for q in qids])
        ql = np.array([len(queries[q].split()) for q in qids], dtype=float)
        parts[sp] = {"qids": qids, "delta": d, "F": b <= 1e-12, "pre": np.hstack([inp[sp]["pre"], np.c_[nq_, ql]]),
                     "X": {k: np.array([probe_features(bm[q], probes[q], k, n, l) for q, n, l in zip(qids, nq_, ql)])
                           for k in SIZES}}
        if corpus == "tripclick" and sp == "val":
            queries = None
    V, T = parts["val"], parts["test"]
    assert np.allclose(T["delta"], S["delta"])
    tr = np.zeros(len(V["qids"]), bool)
    tr[sorted(range(len(V["qids"])), key=lambda i: _h(V["qids"][i]))[: int(0.7 * len(V["qids"]))]] = True
    fit = {"delta": V["delta"][tr], "X": {k: v[tr] for k, v in V["X"].items()}, "pre": V["pre"][tr]}
    sel = {"delta": V["delta"][~tr], "X": {k: v[~tr] for k, v in V["X"].items()}, "pre": V["pre"][~tr]}
    if corpus == "nq":
        b1 = np.load("results/nq_stronger_baselines/rerank_test_scores.npy")[0]
    else:  # TripClick B1 under the same rule; local only, never released
        b1, _ = rn.cat_gbdt(fit["pre"], fit["delta"], sel["pre"], sel["delta"], T["pre"])
    T["comparators"] = {"failure": S["det"], "nqc": -S["nqc"], "B1_gbdt": b1}
    return {"fit": fit, "select": sel, "test": T}


# ---------------------------------------------------------------- analysis

def analyze() -> None:
    rn = _load("rn54", "scripts/54_baseline_runner.py")
    da = _load("da50", "scripts/50_decision_analyses.py")
    lat = json.loads((PROBES / "latency.json").read_text())
    res = {"_note": "EXPLORATORY; protocol configs/partial_probe_protocol.json (frozen before computing). "
                    "All test sets were inspected before; not independent confirmation."}
    ps = {}
    for corpus in ("nq", "tripclick", "arguana"):
        D = corpus_data(corpus)
        T, d = D["test"], D["test"]["delta"]
        L = {n: v["median_s"] for n, v in lat[corpus].items()}
        scores = dict(T["comparators"])
        hp = {}
        for k in SIZES:
            scores[f"probe_{k}"], hp[f"probe_{k}"] = rn.cat_gbdt(D["fit"]["X"][k], D["fit"]["delta"],
                                                                 D["select"]["X"][k], D["select"]["delta"], T["X"][k])
        np.save(PROBES / f"{corpus}_test_probe_scores.npy", np.vstack([scores[f"probe_{k}"] for k in SIZES]))
        scores["oracle"] = d

        def cost(policy: str, b: float) -> float:
            if policy.startswith("probe_"):
                k = int(policy.split("_")[1])
                return L[f"probe_{k}"] + b * L[f"rest_{k}"]
            tel = L["telemetry"] if policy in ("failure", "B1_gbdt") else 0.0
            return tel + b * L["full"]

        curves = {}
        for m, s in scores.items():
            pts = []
            for b in BUDGETS:
                r = rn.metrics(d, s, b)
                pts.append({"budget": b, "gain": r["gain"], "cost_s": cost(m, b) if m != "oracle" else b * L["full"],
                            "harm_rate_selected": r["harm_rate_selected"],
                            "oracle_regret": rn.metrics(d, d, b)["gain"] - r["gain"]})
            curves[m] = pts
        curves["random"] = [{"budget": b, "gain": b * d.mean(), "cost_s": b * L["full"]} for b in BUDGETS]
        quality = {}
        for m, s in scores.items():
            if m == "oracle":
                continue
            quality[m] = {"spearman": float(spearmanr(s, d).statistic),
                          "auc_helped": float(roc_auc_score(d > 1e-9, s))}
        comps = {}
        for other in ("failure", "B1_gbdt"):
            for k in SIZES:
                key = f"probe_{k}_minus_{other}"
                comps[key] = rn.bootstrap_pair(d, scores[f"probe_{k}"], scores[other], budgets=TEST_BUDGETS, nboot=10000)
                if k == PRIMARY:
                    ps[f"{corpus}:{key}"] = comps[key]["signflip_p"]

        # Frontier (protocol: "at equal or lower cost"): the best probe-10 gain on its
        # piecewise-linear curve at cost <= each pre-action point's cost; a linear
        # segment peaks at an endpoint, so check the points below c and the value at c.
        pc = curves[f"probe_{PRIMARY}"]
        xs, ys = [p["cost_s"] for p in pc], [p["gain"] for p in pc]

        def best_at_or_below(c: float) -> float:
            if c < xs[0]:
                return np.nan
            return max([y for x, y in zip(xs, ys) if x <= c] + [float(np.interp(c, xs, ys))])

        frontier = {}
        for m in ("failure", "nqc", "B1_gbdt", "random"):
            frontier[m] = [{"budget": p["budget"], "cost_s": p["cost_s"], "gain": p["gain"],
                            "probe_gain_at_equal_cost": float(np.interp(p["cost_s"], xs, ys, left=np.nan, right=ys[-1])),
                            "probe_best_gain_at_or_below_cost": best_at_or_below(p["cost_s"])}
                           for p in curves[m]]
        res[corpus] = {"n_test": int(len(d)), "latency_s": L, "hyperparameters": hp, "curves": curves,
                       "prediction_quality": quality, "comparisons_10_75": comps, "equal_cost": frontier}
        print(corpus, "done", json.dumps({m: round(np.mean([p["gain"] for p in c if 0 < p["budget"] < 1]) * 1e3, 1)
                                          for m, c in curves.items()}), flush=True)
    res["holm_primary"] = da.holm(ps)

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else round(float(x), 4)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        if isinstance(x, (np.bool_,)):
            return bool(x)
        return x
    Path("docs/partial_probe_summary.json").write_text(json.dumps(rnd(res), indent=2) + "\n")


if __name__ == "__main__":
    {"time": time_all, "analyze": analyze}[sys.argv[1]]()

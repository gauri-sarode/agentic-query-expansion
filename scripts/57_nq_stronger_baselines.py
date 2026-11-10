"""EXPLORATORY (configs/nq_stronger_baselines.json, frozen before running):
the ArguAna runner's B1 (gradient boosting) and B2 (semantic kNN) on NQ,
fit on NQ validation, scored once on NQ test against Table 3's failure
routing. Writes rounded aggregates to docs/nq_stronger_baselines.json and
per-query predictions to results/nq_stronger_baselines/ (gitignored).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.data import load_queries  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


da = _load("da50", "scripts/50_decision_analyses.py")
rn = _load("runner54", "scripts/54_baseline_runner.py")
ra, iv = da.ra, da.iv
BUDGETS = (0.10, 0.25, 0.50, 0.75, 1.00)
SECONDARY = (0.10, 0.25, 0.50, 0.75)
OUT = Path("results/nq_stronger_baselines")


def split_fit_pool(qids: list[str]) -> tuple[np.ndarray, np.ndarray]:
    order = sorted(range(len(qids)), key=lambda i: hashlib.sha256(f"{qids[i]}:nq57".encode()).hexdigest())
    n_tr = int(0.7 * len(qids))
    tr = np.zeros(len(qids), bool)
    tr[order[:n_tr]] = True
    return tr, ~tr


def features(t0: np.ndarray, qids: list[str], bm: dict, queries: dict) -> np.ndarray:
    extra = np.array([[da.nqc(bm[q]), len(queries[q].split())] for q in qids], dtype=float)
    return np.hstack([np.asarray(t0, dtype=float), extra])


def embed(qids: list[str], queries: dict) -> np.ndarray:
    from src.retrieval.dense_index import _get_model
    return _get_model().encode([queries[q] for q in qids], batch_size=32, normalize_embeddings=True,
                               show_progress_bar=False).astype(np.float32)


def nq_inputs() -> dict:
    """Validation fit pools and test labels per intervention, as Table 3."""
    queries = load_queries("nq")
    inp = iv.corpus_inputs("nq")
    sel = json.loads((iv.RUNS / "prf_selected.json").read_text())["nq"]
    d, t, lam = sel["fb_docs"], sel["fb_terms"], sel["lam"]
    bms, delta = {}, {}
    for sp in ("val", "test"):
        qids, qrels = inp[sp]["qids"], inp[sp]["qrels"]
        bm = bms[sp] = iv._runs("nq", sp, "bm25")
        rr = iv._runs("nq", sp, "rerank")
        ex = iv._runs("nq", sp, f"prf_d{d}_t{t}")
        b = iv.ndcg(qids, {q: [x for x, _ in bm[q]] for q in qids}, qrels)
        fused = {q: iv.prf_fuse(bm[q], ex[q]["ranking"], lam) for q in qids}
        delta[sp] = {"rerank": iv.ndcg(qids, {q: [x for x, _ in rr[q]] for q in qids}, qrels) - b,
                     "prf": iv.ndcg(qids, {q: [x for x, _ in fused[q]] for q in qids}, qrels) - b}
    vq, tq = inp["val"]["qids"], inp["test"]["qids"]
    Xv = features(inp["val"]["pre"], vq, bms["val"], queries)
    Xt = features(inp["test"]["pre"], tq, bms["test"], queries)
    Ev, Et = embed(vq, queries), embed(tq, queries)

    settings = {S["name"]: S for S in da.llm_qe_settings() + da.intervention_settings()}
    out = {}
    for name in ("rerank", "prf"):
        S = settings[f"{name}_nq"]
        assert np.allclose(S["delta"], delta["test"][name])
        out[name] = {"qids": vq, "X": Xv, "E": Ev, "y": delta["val"][name],
                     "Xt": Xt, "Et": Et, "dt": S["delta"], "det": S["det"], "benefit": S["benefit_pre"]}
    cal = ra._load("results/nq_val_calibration.json")
    acted = da.replay_acted(np.array(cal["X_t0"]), cal["t0_feature_names"], cal["val_qids"], queries)
    assert int(acted.sum()) == len(cal["y_delta_ndcg"])
    assert list(cal["val_qids"]) == list(vq)
    S = settings["llm_qe_nq"]
    out["qe"] = {"qids": [q for q, a in zip(vq, acted) if a], "X": Xv[acted], "E": Ev[acted],
                 "y": np.array(cal["y_delta_ndcg"]), "Xt": Xt, "Et": Et, "dt": S["delta"],
                 "det": S["det"], "benefit": S["benefit_pre"]}
    return out


def main() -> None:
    data = nq_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    res, ps = {"_note": "EXPLORATORY; protocol configs/nq_stronger_baselines.json (frozen before running). "
                        "NQ test was inspected before this analysis."}, {}
    for action, D in data.items():
        tr, se = split_fit_pool(D["qids"])
        b1, b1_hp = rn.cat_gbdt(D["X"][tr], D["y"][tr], D["X"][se], D["y"][se], D["Xt"])
        b2, b2_hp = rn.semantic_knn(D["E"][tr], D["y"][tr], D["E"][se], D["y"][se], D["Et"])
        dt, fail = D["dt"], D["det"]
        np.save(OUT / f"{action}_test_scores.npy", np.vstack([b1, b2]))
        scorers = {"failure": fail, "B1_gbdt": b1, "B2_semantic_knn": b2, "benefit_linear": D["benefit"], "oracle": dt}
        r = {"n_train": int(tr.sum()), "n_select": int(se.sum()), "n_test": int(len(dt)),
             "hyperparameters": {"B1_gbdt": b1_hp, "B2_semantic_knn": b2_hp},
             "per_budget": {m: [rn.metrics(dt, s, b)["gain"] for b in BUDGETS] for m, s in scorers.items()},
             "harm_rate_selected": {m: [rn.metrics(dt, s, b)["harm_rate_selected"] for b in BUDGETS]
                                    for m, s in scorers.items() if m != "oracle"}}
        r["per_budget"]["random"] = [b * dt.mean() for b in BUDGETS]
        avg = {m: float(np.mean(v)) for m, v in r["per_budget"].items()}
        r["budget_averaged"] = avg
        for m, s in (("B1_gbdt", b1), ("B2_semantic_knn", b2)):
            r[f"{m}_minus_failure"] = rn.bootstrap_pair(dt, s, fail, budgets=BUDGETS, nboot=10000)
            r[f"{m}_minus_failure_10_75"] = rn.bootstrap_pair(dt, s, fail, budgets=SECONDARY, nboot=10000)
            head = avg["oracle"] - avg["failure"]
            r[f"{m}_headroom_share"] = (avg[m] - avg["failure"]) / head if head > 0 else None
            ps[f"{action}:{m}"] = r[f"{m}_minus_failure"]["signflip_p"]
        res[action] = r
        print(action, "done", json.dumps(r["budget_averaged"]), flush=True)
    res["holm_primary"] = da.holm(ps)

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return round(float(x), 4)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [rnd(v) for v in x]
        if isinstance(x, np.bool_):
            return bool(x)
        if isinstance(x, np.integer):
            return int(x)
        return x
    Path("docs/nq_stronger_baselines.json").write_text(json.dumps(rnd(res), indent=2) + "\n")


if __name__ == "__main__":
    main()

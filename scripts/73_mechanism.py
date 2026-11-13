"""EXPLORATORY mechanism analysis, as frozen in configs/mechanism_protocol.json
(committed before computing). Tests whether rescue of failing queries is
governed by reachability (a relevant document at BM25 ranks 11-100), which
a failure score built from the top of the ranking does not observe.

Writes rounded aggregates only to docs/mechanism.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.data import load_qrels  # noqa: E402


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ss = _load("ss69", "scripts/69_score_sufficiency.py")
auc = _load("cc65", "scripts/65_core_claim_audit.py").auc
EPS, NBOOT, SEED = 1e-9, 10000, 42


def settings():
    """(name, S, delta, rankings, relevant-sets) per setting, test order."""
    da = _load("da50", "scripts/50_decision_analyses.py")
    iv = da.iv
    by = {s["name"]: s for s in da.intervention_settings()}
    out = []
    for corpus in ("nq", "tripclick"):
        inp = iv.corpus_inputs(corpus)
        qids, qrels = inp["test"]["qids"], inp["test"]["qrels"]
        bm = iv._runs(corpus, "test", "bm25")
        rk = [[d for d, _ in bm[q]][:100] for q in qids]
        rel = [{d for d, g in qrels.get(q, {}).items() if g > 0} for q in qids]
        for a in ("rerank", "prf"):
            S = by[f"{a}_{corpus}"]
            out.append((f"{a}_{corpus}", S["det"], S["delta"], S["fail"].astype(bool), rk, rel))
    for corpus, qn in (("arguana", "arguana"), ("fiqa", "fiqa-test")):
        qrels = load_qrels(qn)
        bm = json.loads(Path(f"results/{corpus}/bm25.json").read_text())
        pred = pd.read_csv(f"results/{corpus}/runner/test_predictions.csv")
        for a in ("rerank", "prf"):
            g = pred[pred.action == a]
            qids = list(g["query_id"].astype(str))
            rk = [[d for d, _ in bm[q]["ranking"]][:100] for q in qids]
            rel = [{d for d, x in qrels.get(q, {}).items() if x > 0} for q in qids]
            out.append((f"{a}_{corpus}", g["failure"].to_numpy(), g["delta"].to_numpy(),
                        g["original_ndcg"].to_numpy() <= EPS, rk, rel))
    return out


def ci(stat, n, rng):
    v = np.array([stat(rng.integers(0, n, n)) for _ in range(NBOOT)], dtype=float)
    v = v[~np.isnan(v)]
    return [float(np.quantile(v, .025)), float(np.quantile(v, .975))] if len(v) else None


def nanmean(x):
    return float(x.mean()) if len(x) else float("nan")


def analyse(s, d, F, rk, rel) -> dict:
    first = np.array([next((i + 1 for i, x in enumerate(r) if x in R), 0) for r, R in zip(rk, rel)])
    top10 = (first >= 1) & (first <= 10)
    assert np.array_equal(~top10, F), "failure label does not match 'no relevant in top 10'"
    reach = F & (first >= 11)
    helped, harmed = d > EPS, d < -EPS
    rng = np.random.default_rng(SEED)
    n = len(d)
    out = {"n": int(n), "n_failing": int(F.sum())}

    def cell(mask_fn):
        def stat(i):
            m = mask_fn(i)
            return nanmean(helped[i][m]) if m.any() else np.nan
        return stat

    fi, ri = F, reach
    out["P_reachable_given_F"] = {"estimate": float(reach[F].mean()),
                                  "ci": ci(lambda i: nanmean(ri[i][fi[i]]), n, rng)}
    for nm, m in (("reachable", reach), ("unreachable", F & ~reach)):
        out[f"failing_{nm}"] = {"n": int(m.sum()), "P_helped": nanmean(helped[m]), "mean_delta": nanmean(d[m]),
                                "P_helped_ci": ci(cell(lambda i, m=m: m[i]), n, rng)}
    out["decomposition"] = {"P_helped_given_F": float(helped[F].mean()),
                            "via_reachable": float(reach[F].mean() * nanmean(helped[reach])),
                            "via_unreachable": float((F & ~reach)[F].mean() * nanmean(helped[F & ~reach]))}
    out["AUC_S_reachable_among_failing"] = {
        "estimate": auc(reach[F], s[F]),
        "ci": ci(lambda i: auc(ri[i][fi[i]], s[i][fi[i]]), n, rng)}
    # Helped rate by S tertile within reachable failing queries.
    def tertiles(mask):
        cut = np.quantile(s[mask], [1 / 3, 2 / 3])
        t = np.searchsorted(cut, s, side="right")
        return [{"n": int((mask & (t == j)).sum()), "P_helped": nanmean(helped[mask & (t == j)]),
                 "mean_delta": nanmean(d[mask & (t == j)]),
                 "P_helped_ci": ci(cell(lambda i, mm=(mask & (t == j)): mm[i]), n, rng)} for j in range(3)]
    out["reachable_failing_by_S_tertile"] = tertiles(reach) if reach.sum() >= 30 else None
    # Healthy categories by first relevant rank.
    cats = {"rank1": first == 1, "rank2_3": (first >= 2) & (first <= 3), "rank4_10": (first >= 4) & (first <= 10)}
    out["healthy"] = {}
    for nm, m in cats.items():
        out["healthy"][nm] = {"n": int(m.sum()), "P_helped": nanmean(helped[m]), "P_harmed": nanmean(harmed[m]),
                              "mean_delta": nanmean(d[m]),
                              "mean_delta_ci": ci(lambda i, m=m: nanmean(d[i][m[i]]), n, rng),
                              "AUC_S_this_vs_other_healthy": auc(m[~F], s[~F]) if 0 < m.sum() < (~F).sum() else None}
    out["healthy_by_S_tertile"] = tertiles(~F)
    return out


def main() -> None:
    res = {"_note": "EXPLORATORY (configs/mechanism_protocol.json, committed before computing). reachable = failing "
                    "query with a judged-relevant document at BM25 ranks 11-100. Aggregates only."}
    for nm, s, d, F, rk, rel in settings():
        r = analyse(s, d, F, rk, rel)
        res[nm] = r
        print(f"{nm:18s} P(reach|F)={r['P_reachable_given_F']['estimate']:.2f} "
              f"P(h|reach)={r['failing_reachable']['P_helped']:.2f} P(h|unreach)={r['failing_unreachable']['P_helped']:.2f} "
              f"AUC_S(reach|F)={r['AUC_S_reachable_among_failing']['estimate']:.2f} {r['AUC_S_reachable_among_failing']['ci']} "
              f"tertiles={[round(t['P_helped'], 2) for t in r['reachable_failing_by_S_tertile']] if r['reachable_failing_by_S_tertile'] else None} "
              f"healthy dD={[round(v['mean_delta'], 3) for v in r['healthy'].values()]}", flush=True)

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return None if np.isnan(x) else round(float(x), 4)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        return x
    Path("docs/mechanism.json").write_text(json.dumps(rnd(res), indent=2) + "\n")
    print("done")


if __name__ == "__main__":
    main()

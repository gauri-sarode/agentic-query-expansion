"""EXPLORATORY (post-review, added after the scripts/83 smoke run): (1) inner
out-of-fold balance of the scripts/83 amended split per corpus; (2) the
probe frontier at equal median latency: Zp reranking a share b after paying
the Probe-10 cost, against Z0_cheap reranking b + c_probe / c_rerank (capped
at 1), budget-averaged over b in {10, 25, 50, 75}%, paired bootstrap
(10,000, seed 42). Same data, models and folds as scripts/83. Aggregates
only to docs/frontier_cost_check.json.
"""
import importlib.util
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("o83", "scripts/83_observability_frontier.py")
o83 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o83)
a81 = o83.a81
NBOOT, SEED, AVG = 10000, 42, a81.AVG


def gain_at(score, d, b):
    n, order = len(d), np.argsort(-score, kind="stable")
    return float(d[order[: int(np.floor(min(b, 1.0) * n))]].sum() / n)


def main() -> None:
    costs = o83.costs()
    res = {"_note": __doc__.split("\n\n")[0].replace("\n", " "), "costs": costs}
    for corpus in ("nq", "tripclick", "arguana", "fiqa", "dbpedia", "hotpotqa"):
        folds, _ = o83.nq_tripclick(corpus) if corpus in ("nq", "tripclick") else o83.beir(corpus)
        sizes = [np.bincount([o83.hfold(q) for q in tr.index], minlength=5).tolist() for tr, _ in folds]
        z0, zp, tests = [], [], []
        f0 = a81.CHEAP_BASE + a81.DEEP
        fp = f0 + o83.PROBE
        for tr, te in folds:
            z0.append(lgb.LGBMRegressor(**a81.LGB).fit(tr[f0].to_numpy(), tr["delta"].to_numpy()).predict(te[f0].to_numpy()))
            zp.append(lgb.LGBMRegressor(**a81.LGB).fit(tr[fp].to_numpy(), tr["delta"].to_numpy()).predict(te[fp].to_numpy()))
            tests.append(te)
        d = pd.concat(tests)["delta"].to_numpy()
        z0, zp = np.concatenate(z0), np.concatenate(zp)
        extra = costs[corpus]["probe_10"] / costs[corpus]["rerank"]

        def diff(i):
            return float(np.mean([gain_at(zp[i], d[i], b) - gain_at(z0[i], d[i], b + extra) for b in AVG]))
        rng = np.random.default_rng(SEED)
        draws = np.array([diff(rng.integers(0, len(d), len(d))) for _ in range(NBOOT)])
        est = diff(np.arange(len(d)))
        res[corpus] = {"inner_fold_sizes": sizes, "probe_cost_in_rerank_shares": round(extra, 4),
                       "Zp_minus_Z0cheap_equal_latency": {"estimate": round(est, 4),
                                                           "ci": [round(float(np.quantile(draws, q)), 4) for q in (.025, .975)],
                                                           "p": round(a81.pval(draws), 4)}}
        print(corpus, res[corpus], flush=True)
    Path("docs/frontier_cost_check.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()

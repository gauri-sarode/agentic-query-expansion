"""EXPLORATORY (post-review, after the test passes): raw failure-score routing
for reranking, i.e. rerank the top b of test queries by the learned failure
gate's score with no calibration, minus random routing (b * mean Delta),
averaged over b in {10, 25, 50, 75}%. Paired bootstrap over test queries
(10,000 resamples, seed 42). Compares with the pre-registered P2, which
routes by cross-fitted binned E[Delta | score]. Aggregates only to
docs/raw_routing_check.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

AVG, NBOOT, SEED = (0.10, 0.25, 0.50, 0.75), 10000, 42


def preds(corpus: str) -> pd.DataFrame:
    if corpus == "dbpedia":
        p = pd.concat([pd.read_csv(f"results/dbpedia/runner_fold{k}/test_predictions.csv") for k in range(5)])
    else:
        p = pd.read_csv(f"results/{corpus}/runner/test_predictions.csv")
    return p[p.action == "rerank"]


def raw_minus_random(s: np.ndarray, d: np.ndarray) -> float:
    n, order = len(d), np.argsort(-s, kind="stable")
    return float(np.mean([d[order[: int(np.floor(b * n))]].sum() / n - b * d.mean() for b in AVG]))


def main() -> None:
    res = {"_note": __doc__.split("\n\n")[0].replace("\n", " ")}
    for corpus in ("fiqa", "dbpedia", "hotpotqa"):
        p = preds(corpus)
        s, d = p["failure"].to_numpy(), p["delta"].to_numpy()
        rng = np.random.default_rng(SEED)
        boot = [raw_minus_random(s[i], d[i]) for i in (rng.integers(0, len(d), len(d)) for _ in range(NBOOT))]
        res[corpus] = {"n": len(d), "raw_minus_random": round(raw_minus_random(s, d), 4),
                       "ci": [round(float(np.quantile(boot, q)), 4) for q in (.025, .975)]}
        print(corpus, res[corpus], flush=True)
    Path("docs/raw_routing_check.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()

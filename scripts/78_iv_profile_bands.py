"""Bootstrap bands for the failure-score decile profile (P(F), P(Delta>0))
of reranking on the independent test sets (DBPedia, cross-fitted folds
pooled; HotpotQA), with the scripts/70 procedure (2,000 resamples within
decile, seed 42). Point estimates equal docs/{corpus}_summary.json profile.
Reads the runners' test predictions; writes aggregates only to
docs/iv_profile.json, for scripts/71.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

EPS, K, FIG_BOOT, SEED = 1e-9, 10, 2000, 42


def preds(corpus: str) -> pd.DataFrame:
    if corpus == "dbpedia":
        p = pd.concat([pd.read_csv(f"results/dbpedia/runner_fold{k}/test_predictions.csv") for k in range(5)])
    else:
        p = pd.read_csv("results/hotpotqa/runner/test_predictions.csv")
    return p[p.action == "rerank"]


def profile(s, d, F):
    bins = np.searchsorted(np.quantile(s, np.linspace(0, 1, K + 1)[1:-1]), s, side="right")
    rng = np.random.default_rng(SEED)
    out = []
    for j in range(K):
        f, h = F[bins == j].astype(float), (d[bins == j] > EPS).astype(float)
        bi = [rng.integers(0, len(f), len(f)) for _ in range(FIG_BOOT)]
        out.append({"n": int(len(f)), "p_fail": float(f.mean()), "p_helped": float(h.mean()),
                    "p_fail_ci": [float(np.quantile([f[i].mean() for i in bi], q)) for q in (.025, .975)],
                    "p_helped_ci": [float(np.quantile([h[i].mean() for i in bi], q)) for q in (.025, .975)]})
    return out


def main() -> None:
    res = {"_note": "Reranking, test. Deciles of the learned failure gate's score; 95% bootstrap bands. Aggregates only."}
    for corpus in ("dbpedia", "hotpotqa"):
        p = preds(corpus)
        prof = profile(p["failure"].to_numpy(), p["delta"].to_numpy(), p["original_ndcg"].to_numpy() <= EPS)
        ref = json.loads(Path(f"docs/{corpus}_summary.json").read_text())["per_action"]["rerank"]["profile"]
        assert all(abs(a["p_fail"] - b["p_fail"]) < 1e-3 and abs(a["p_helped"] - b["p_helped"]) < 1e-3
                   for a, b in zip(prof, ref)), corpus
        res[f"rerank_{corpus}"] = {"figure_profile": [{k: (round(v, 4) if isinstance(v, float) else
                                                         [round(x, 4) for x in v] if isinstance(v, list) else v)
                                                        for k, v in b.items()} for b in prof]}
        print(corpus, [round(b["p_helped"], 2) for b in prof])
    Path("docs/iv_profile.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()

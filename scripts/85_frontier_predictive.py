"""EXPLORATORY (post-review): predictive value of each scripts/83 information
set, separate from its decision value: helped AUC (Delta > 0) and Spearman
with Delta on test, for Z0_free (score list + query length), Z0_full
(telemetry + score list), Zp (Z0_free + Probe-10), Z1 (Zp + post-action).
Same data, models and folds as scripts/83. Point estimates; aggregates only
to docs/frontier_predictive.json.
"""
import importlib.util
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("o83", "scripts/83_observability_frontier.py")
o83 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o83)
a81 = o83.a81
SETS = {"Z0_free": a81.CHEAP_BASE + a81.DEEP, "Z0_full": a81.BASE + a81.DEEP,
        "Zp": a81.CHEAP_BASE + a81.DEEP + o83.PROBE, "Z1": a81.CHEAP_BASE + a81.DEEP + o83.PROBE + o83.POST}


def main() -> None:
    res = {"_note": __doc__.split("\n\n")[0].replace("\n", " ")}
    for corpus in ("nq", "tripclick", "arguana", "fiqa", "dbpedia", "hotpotqa"):
        folds, _ = o83.nq_tripclick(corpus) if corpus in ("nq", "tripclick") else o83.beir(corpus)
        preds, tests = {k: [] for k in SETS}, []
        for tr, te in folds:
            for k, f in SETS.items():
                preds[k].append(lgb.LGBMRegressor(**a81.LGB).fit(tr[f].to_numpy(), tr["delta"].to_numpy()).predict(te[f].to_numpy()))
            tests.append(te)
        d = pd.concat(tests)["delta"].to_numpy()
        res[corpus] = {k: {"helped_auc": round(a81.auc(d > 1e-9, np.concatenate(v)), 4),
                           "spearman": round(float(spearmanr(np.concatenate(v), d).statistic), 4)} for k, v in preds.items()}
        print(corpus, res[corpus], flush=True)
    Path("docs/frontier_predictive.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()

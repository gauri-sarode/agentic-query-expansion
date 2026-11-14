"""EXPLORATORY (post-review): paired contrast of the cheap-feature gates
(scripts/81 RCBR_cheap, Mono_cheap: BM25 score list + query length) against
the deployed failure gate S, budget-averaged routing gain, paired bootstrap
(10,000, seed 42); plus the measured per-query cost of computing the cheap
features and evaluating the cheap model. Same data, models and folds as
scripts/81. Aggregates only to docs/cheap_gate_check.json.
"""
import importlib.util
import json
import sys
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("a81", "scripts/81_actionability.py")
a81 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(a81)


def main() -> None:
    res = {"_note": __doc__.split("\n\n")[0].replace("\n", " ")}
    for corpus in ("nq", "tripclick", "arguana", "fiqa", "dbpedia", "hotpotqa"):
        folds = a81.nq_tripclick(corpus) if corpus in ("nq", "tripclick") else a81.beir(corpus)
        note, rc, mc, tests = [], [], [], []
        feats = a81.CHEAP_BASE + a81.DEEP
        for tr, te in folds:
            p = a81.fit_predict(tr, te, feats, note)
            rc.append(p["rcbr"]); mc.append(p["mono"]); tests.append(te)
        te = pd.concat(tests)
        d = te["delta"].to_numpy()
        sc = {"RCBR_cheap": np.concatenate(rc), "Mono_cheap": np.concatenate(mc), "S": te["S"].to_numpy()}
        out, pt = a81.boot_contrasts(sc, d, [("RCBR_cheap", "S"), ("Mono_cheap", "S")])
        res[corpus] = {"routing_gain": pt, "contrasts": out}
        print(corpus, {k: (round(v["estimate"], 4), [round(x, 4) for x in v["ci"]]) for k, v in out.items()}, flush=True)
    # cost of the cheap path on HotpotQA: features from a stored score list + one model evaluation per query
    bm = {q: v["ranking"] for q, v in json.loads(Path("results/hotpotqa/bm25.json").read_text()).items()}
    tr, te = a81.beir("hotpotqa")[0]
    model = lgb.LGBMRegressor(**a81.LGB).fit(tr[a81.CHEAP_BASE + a81.DEEP].to_numpy(), tr["delta"].to_numpy())
    qs = list(te.index[:500])
    t = time.perf_counter()
    for q in qs:
        x = np.array([list(te.loc[q, a81.CHEAP_BASE]) + a81.deep_features(bm[q])])
        model.predict(x)
    res["hotpotqa_cheap_path_seconds_per_query"] = (time.perf_counter() - t) / len(qs)
    print("cheap path s/query", res["hotpotqa_cheap_path_seconds_per_query"])
    Path("docs/cheap_gate_check.json").write_text(json.dumps(json.loads(json.dumps(res), parse_float=lambda v: round(float(v), 5)), indent=2) + "\n")


if __name__ == "__main__":
    main()

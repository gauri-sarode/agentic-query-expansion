"""EXPLORATORY budget sensitivity of pre-action routing (written after every
test result was read). The registered budget average includes the 100%
budget, where every policy equals intervening on every query; this
recomputes the routing contrasts averaged over 10-75% only, plus per-budget
points, from cached traces and predictions. No model is refit.

TripClick/NQ: scripts/50 D2 with BUDGETS = 10-75%. ArguAna: the verbatim
runner's paired bootstrap on its own test predictions (exact budgets).
Writes rounded aggregates to docs/budget_sensitivity_summary.json.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, ".")


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BUDGETS = (0.10, 0.25, 0.50, 0.75)


def main() -> None:
    out = {"_note": "EXPLORATORY: budget average over 10-75% (the registered average also includes 100%, "
                    "where all policies coincide). Computed after test results were read; not pre-registered."}

    da = _load("da50", "scripts/50_decision_analyses.py")
    da.BUDGETS = BUDGETS
    settings = da.llm_qe_settings() + da.intervention_settings()
    d2 = {S["name"]: da.d2(S) for S in settings}
    for other in ("benefit_pre", "nqc"):
        ps = {k: v[f"{other}_minus_failure"]["p"] for k, v in d2.items() if f"{other}_minus_failure" in v}
        d2[f"holm_{other}_minus_failure"] = da.holm(ps)
    out["tripclick_nq"] = d2
    print("tripclick/nq done", flush=True)

    rn = _load("runner54", "scripts/54_baseline_runner.py")
    pred = pd.read_csv("results/arguana/runner/test_predictions.csv")
    arg = {}
    for action, g in pred.groupby("action"):
        delta, fail = g["delta"].to_numpy(), g["failure"].to_numpy()
        res = {"per_budget": {m: [rn.metrics(delta, g[m].to_numpy(), b)["gain"] for b in BUDGETS]
                              for m in ("failure", "nqc", "B1_gbdt", "B2_semantic_knn", "B3_quantile_LCB", "oracle")}}
        res["per_budget"]["random"] = [b * delta.mean() for b in BUDGETS]
        for m in ("B1_gbdt", "B2_semantic_knn", "B3_quantile_LCB"):
            bp = rn.bootstrap_pair(delta, g[m].to_numpy(), fail, budgets=BUDGETS, nboot=10000)
            res[f"{m}_minus_failure"] = bp
        arg[action] = res
        print(action, "done", flush=True)
    out["arguana"] = arg

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return round(float(x), 4)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple, np.ndarray)):
            return [rnd(v) for v in x]
        return x
    Path("docs/budget_sensitivity_summary.json").write_text(json.dumps(rnd(out), indent=2) + "\n")


if __name__ == "__main__":
    main()

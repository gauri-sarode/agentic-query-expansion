"""EXPLORATORY per-budget uncertainty for the routing contrasts already
reported in aggregate: benefit minus failure routing (Table 3 settings,
TripClick/NQ) and ArguAna B1 minus failure routing, at each budget
separately. Existing scores only, no fitting; runner bootstrap_pair
with a single budget, 10^4 resamples. Descriptive: no multiplicity
correction (the registered five-budget averages remain the tests).
Writes rounded aggregates to docs/per_budget_ci.json.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pandas as pd


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BUDGETS = (0.10, 0.25, 0.50, 0.75)


def contrast(rn, d, a, b) -> dict:
    out = {}
    for k in BUDGETS:
        r = rn.bootstrap_pair(d, a, b, budgets=[k])
        out[f"{int(k * 100)}"] = {x: round(v, 4) for x, v in r.items()}
    return out


def main() -> None:
    rn = _load("rn54", "scripts/54_baseline_runner.py")
    da = _load("da50", "scripts/50_decision_analyses.py")
    res = {"_note": "EXPLORATORY; per-budget B-F, runner bootstrap_pair (one budget each), no multiplicity correction."}
    for S in da.llm_qe_settings() + da.intervention_settings():
        res[S["name"]] = contrast(rn, S["delta"], S["benefit_pre"], S["det"])
    pred = pd.read_csv("results/arguana/runner/test_predictions.csv")
    for action, g in pred.groupby("action"):
        res[f"{action}_arguana_B1"] = contrast(rn, g["delta"].to_numpy(), g["B1_gbdt"].to_numpy(), g["failure"].to_numpy())
    Path("docs/per_budget_ci.json").write_text(json.dumps(res, indent=2) + "\n")
    for k, v in res.items():
        if k != "_note":
            print(k, {b: (r["estimate"], r["ci_low"], r["ci_high"]) for b, r in v.items()})


if __name__ == "__main__":
    main()

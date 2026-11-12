"""EXPLORATORY post-review sensitivity of the probe cost comparisons to the
latency statistic: recomputes probe-10 versus NQC at equal measured cost
with P95 instead of median latencies (stored test scores, no refitting),
and writes the rounded latency aggregates (median, P95) per corpus.
Writes docs/probe_latency.json.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NBOOT = 2000
SEED = 20261112


def main() -> None:
    pp = _load("pp60", "scripts/60_partial_probe.py")
    rn = pp._load("rn54", "scripts/54_baseline_runner.py")
    lat = json.loads((pp.PROBES / "latency.json").read_text())
    res = {"_note": "EXPLORATORY post-review; single machine, 50 validation queries per corpus; "
                    "ArguAna telemetry median from docs/arguana_feature_latency.txt (no P95).",
           "latency_s": {c: {n: {"median": round(v["median_s"], 3), "p95": round(v["p95_s"], 3) if "p95_s" in v else None}
                             for n, v in lat[c].items()} for c in ("nq", "tripclick", "arguana")}}
    for corpus in ("nq", "tripclick", "arguana"):
        T = pp.corpus_data(corpus)["test"]
        d, nqc = T["delta"], T["comparators"]["nqc"]
        probe = np.load(pp.PROBES / f"{corpus}_test_probe_scores.npy")[pp.SIZES.index(pp.PRIMARY)]
        out = {}
        for stat in ("median_s", "p95_s"):
            L = {n: v[stat] for n, v in lat[corpus].items() if stat in v}
            eq = {b: min(1.0, (L["probe_10"] + b * L["rest_10"]) / L["full"]) for b in pp.TEST_BUDGETS}

            def diff(ix):
                return np.mean([rn.metrics(d[ix], probe[ix], b)["gain"] - rn.metrics(d[ix], nqc[ix], eq[b])["gain"]
                                for b in pp.TEST_BUDGETS])
            rng = np.random.default_rng(SEED)
            draws = np.array([diff(rng.integers(0, len(d), len(d))) for _ in range(NBOOT)])
            out[stat] = {"estimate": round(float(diff(np.arange(len(d)))), 4),
                         "ci_low": round(float(np.quantile(draws, .025)), 4),
                         "ci_high": round(float(np.quantile(draws, .975)), 4),
                         "nqc_fraction_at_equal_cost": {str(b): round(v, 3) for b, v in eq.items()}}
        res[corpus] = {"probe10_minus_nqc_equal_cost": out}
        print(corpus, json.dumps(out), flush=True)
    Path("docs/probe_latency.json").write_text(json.dumps(res, indent=2) + "\n")


if __name__ == "__main__":
    main()

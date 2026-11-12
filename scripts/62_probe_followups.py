"""EXPLORATORY, post-review follow-ups to scripts/60 on stored test scores
(no refitting of probe models): (1) probe-10 versus NQC routing, both at
matched full-rerank fraction (10-75%) and at equal measured cost (NQC's
fraction set so its cost equals probe-10's at each of 10/25/50/75%);
(2) bootstrap CIs for Spearman(score, Delta). Writes rounded aggregates
to docs/probe_followups.json.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


NBOOT = 10000
SEED = 20261112


def main() -> None:
    pp = _load("pp60", "scripts/60_partial_probe.py")
    rn = pp._load("rn54", "scripts/54_baseline_runner.py")
    lat = json.loads((pp.PROBES / "latency.json").read_text())
    res = {"_note": "EXPLORATORY post-review follow-ups on stored probe-10 test scores; not independent confirmation."}
    for corpus in ("nq", "tripclick", "arguana"):
        T = pp.corpus_data(corpus)["test"]
        d = T["delta"]
        probe = np.load(pp.PROBES / f"{corpus}_test_probe_scores.npy")[pp.SIZES.index(pp.PRIMARY)]
        nqc, L = T["comparators"]["nqc"], {n: v["median_s"] for n, v in lat[corpus].items()}
        # NQC fraction at the cost of probe-10 at fraction b
        eq = {b: min(1.0, (L["probe_10"] + b * L["rest_10"]) / L["full"]) for b in pp.TEST_BUDGETS}

        def eq_cost_diff(ix):
            return np.mean([rn.metrics(d[ix], probe[ix], b)["gain"] - rn.metrics(d[ix], nqc[ix], eq[b])["gain"]
                            for b in pp.TEST_BUDGETS])

        rng = np.random.default_rng(SEED)
        n, full = len(d), np.arange(len(d))
        draws = np.array([eq_cost_diff(rng.integers(0, n, n)) for _ in range(NBOOT)])
        out = {"matched_fraction_10_75": rn.bootstrap_pair(d, probe, nqc, budgets=pp.TEST_BUDGETS, nboot=NBOOT),
               "equal_cost": {"estimate": float(eq_cost_diff(full)),
                              "ci_low": float(np.quantile(draws, .025)), "ci_high": float(np.quantile(draws, .975)),
                              "nqc_fraction_at_equal_cost": eq},
               "spearman": {}}
        scorers = {"probe_10": probe, **T["comparators"]}
        idx = [rng.integers(0, n, n) for _ in range(2000)]
        for m, s in scorers.items():
            bs = np.array([spearmanr(s[i], d[i]).statistic for i in idx])
            out["spearman"][m] = {"estimate": float(spearmanr(s, d).statistic),
                                  "ci_low": float(np.nanquantile(bs, .025)), "ci_high": float(np.nanquantile(bs, .975))}
        bs = np.array([spearmanr(probe[i], d[i]).statistic - spearmanr(T["comparators"]["B1_gbdt"][i], d[i]).statistic
                       for i in idx])
        out["spearman"]["probe_10_minus_B1"] = {"ci_low": float(np.nanquantile(bs, .025)),
                                                "ci_high": float(np.nanquantile(bs, .975))}
        res[corpus] = out
        print(corpus, json.dumps(out, default=float)[:900], flush=True)

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return round(float(x), 4)
        if isinstance(x, dict):
            return {str(k): rnd(v) for k, v in x.items()}
        return x
    Path("docs/probe_followups.json").write_text(json.dumps(rnd(res), indent=2) + "\n")


if __name__ == "__main__":
    main()

"""Best failure-restricted policy at positive cost (post-review, on the
frozen traces behind Table 1; no fitting). A policy that may act only
on failing queries does best by intervening exactly where F=1 and
Delta>kappa, earning E[F (Delta-kappa)_+]; the oracle's remaining gain,
E[(1-F)(Delta-kappa)_+], is a ceiling on every failure-restricted policy
at any kappa >= 0. Reports its share of the oracle at kappa in KAPPAS
with paired-bootstrap CIs. Writes docs/restricted_oracle.json.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def shares(d: np.ndarray, F: np.ndarray, ra, kappas) -> dict:
    out = {}
    for k in kappas:
        g = np.maximum(d - k, 0)
        fr, hl = g * F, g * ~F
        boot = [fr[i].sum() / g[i].sum() for i in ra._boot_idx(len(d)) if g[i].sum() > 0]
        out[str(k)] = {"oracle": float(g.mean()), "restricted_best": float(fr.mean()),
                       "healthy_gap": float(hl.mean()),
                       "share_restricted": float(fr.sum() / g.sum()) if g.sum() > 0 else None,
                       "share_ci": list(ra._ci(boot)),
                       "all_failures_policy": float(((d - k) * F).mean())}
    return out


def main() -> None:
    da = _load("da50", "scripts/50_decision_analyses.py")
    ra = da.ra
    res = {"_note": "Post-review; frozen inspected traces except ArguAna (pre-registered test); no fitting."}
    for S in da.llm_qe_settings() + da.intervention_settings() + da.dense_settings():
        res[S["name"]] = shares(S["delta"], S["fail"].astype(bool), ra, da.KAPPAS)
    pred = pd.read_csv("results/arguana/runner/test_predictions.csv")
    for action, g in pred.groupby("action"):
        d = g["delta"].to_numpy()
        res[f"{action}_arguana"] = shares(d, g["original_ndcg"].to_numpy() <= 1e-9, ra, da.KAPPAS)

    def rnd(x):
        if isinstance(x, (float, np.floating)):
            return round(float(x), 4)
        if isinstance(x, dict):
            return {k: rnd(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [rnd(v) for v in x]
        return x
    Path("docs/restricted_oracle.json").write_text(json.dumps(rnd(res), indent=2) + "\n")
    for n, v in res.items():
        if n.startswith("_"):
            continue
        print(f"{n:28s}", " ".join(f"k={k}: share={x['share_restricted']:.3f} [{x['share_ci'][0]:.2f},{x['share_ci'][1]:.2f}] orc={x['oracle']:.4f} allF={x['all_failures_policy']:.4f}" for k, x in v.items()))


if __name__ == "__main__":
    main()

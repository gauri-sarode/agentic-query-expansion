"""verify(previous_state, new_state) -> score. Qrel-free: judges whether
the new retrieval state is healthier than the previous one using
telemetry only -- this is what makes the verifier usable at inference
time, when qrels aren't available.

v1 implementation: loads a trained model artifact
(models/verifier_tripclick_tail_val_v1.joblib, produced by
scripts/10_train_verifier_model.py) rather than hardcoding copy-pasted
coefficients -- retraining on new calibration data means re-running that
script, not hand-editing this file. The artifact is gitignored, not
committed -- it's fit on TripClick data, and TripClick's terms prohibit
publicly sharing statistical models derived from the dataset without
permission. Run scripts/10 locally to (re)produce it.

Selected per configs/experimental_protocol.yaml: entirely on TripClick
TAIL-**val** (n=1147), never on test. Fit as a LinearRegression
(fit_intercept=False) on the 10 telemetry-delta features, chosen because
nothing beat it on val -- neither a fitted GBDT (r=+0.001), nor adding 3
dense-embedding features (r=-0.015), nor reverting to the project's
original v0 hand-picked-weight baseline (r=+0.020, not significant,
p=0.498; see git history around commit e668e0e for those literal
weights). This model's own 5-fold CV estimate on val is r=-0.014 --
weaker than chance-adjacent -- and we deploy it anyway because it is the
methodologically honest choice: the only artifact actually fit on the
correct (post retrieval-fix) val distribution, versus alternatives that
were either untested on this pipeline or fit on stale pre-fix data. Do
not read this as "the verifier works" -- it is the least-bad option
among several that don't, and that is itself part of RQ1's reportable
finding (see paper's RQ1 section for the frozen TAIL-test evaluation,
the actual headline number).
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from src.agent.state import AgentState
from src.observability.telemetry import ObservabilityState

_MODEL_PATH = Path(__file__).resolve().parent.parent.parent / "models" / "verifier_tripclick_tail_val_v1.joblib"


@lru_cache(maxsize=1)
def _load_model() -> dict:
    if not _MODEL_PATH.exists():
        raise FileNotFoundError(
            f"{_MODEL_PATH} not found -- run `python scripts/10_train_verifier_model.py` "
            "to train and save it from a calibration run's output."
        )
    return joblib.load(_MODEL_PATH)


def _features(prev_o: ObservabilityState, new_o: ObservabilityState) -> dict[str, float]:
    return {
        "score_margin_gain": new_o.retrieval.score_margin - prev_o.retrieval.score_margin,
        "coverage_gain": new_o.retrieval.query_term_coverage - prev_o.retrieval.query_term_coverage,
        "coherence_gain": new_o.retrieval.result_coherence - prev_o.retrieval.result_coherence,
        "stability": new_o.retrieval.ranking_stability,
        "drift": new_o.agent.rewrite_semantic_drift,
        "disagreement": new_o.retrieval.reranker_bm25_disagreement,
        "topk_overlap": new_o.retrieval.topk_overlap,
        "reranker_top_score": new_o.retrieval.reranker_top_score,
        "reranker_score_margin": new_o.retrieval.reranker_score_margin,
        "query_length_ratio": new_o.agent.query_length_ratio,
    }


def verify(previous_state: AgentState, new_state: AgentState) -> float:
    """Returns a verifier score; higher means the new state is more
    likely to be a retrieval-quality improvement over previous_state.
    Requires both states' O_t to be populated. Score scale is small
    (roughly -0.1 to +0.03 in the calibration sample) -- do not reuse the
    v0 thresholds (~-1 to +0.05) against this verifier; see
    src/agent/controller.py's RecoveryThresholds, recalibrated to match.
    """
    prev_o, new_o = previous_state.O_t, new_state.O_t
    if prev_o is None or new_o is None:
        raise ValueError("verify() requires both states to have O_t computed")

    artifact = _load_model()
    features = _features(prev_o, new_o)
    x = np.array([[features[name] for name in artifact["feature_names"]]])
    return float(artifact["model"].predict(x)[0])

from src.eval.bootstrap import paired_bootstrap
from src.eval.metrics import helped_unchanged_harmed
from src.eval.slo import detection_recall, recovery_rate, rollback_precision


def test_helped_unchanged_harmed_partitions_all_queries():
    baseline = {"q1": {"ndcg_cut_10": 0.5}, "q2": {"ndcg_cut_10": 0.5}, "q3": {"ndcg_cut_10": 0.5}}
    treatment = {"q1": {"ndcg_cut_10": 0.7}, "q2": {"ndcg_cut_10": 0.5}, "q3": {"ndcg_cut_10": 0.2}}
    result = helped_unchanged_harmed(baseline, treatment)
    assert result == {"helped": 1 / 3, "unchanged": 1 / 3, "harmed": 1 / 3}


def test_paired_bootstrap_detects_consistent_improvement():
    a = [0.1] * 50
    b = [0.5] * 50
    result = paired_bootstrap(a, b, n_resamples=1000)
    assert result["mean_diff"] > 0
    assert result["significant"]


def test_slo_ratio_helpers():
    assert detection_recall(8, 10) == 0.8
    assert rollback_precision(3, 4) == 0.75
    assert recovery_rate(0, 0) != recovery_rate(0, 0)  # nan != nan

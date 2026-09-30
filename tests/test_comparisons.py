"""Inference tests assert statistical invariants, not just output structure."""
import numpy as np
import pandas as pd
import pytest

from perturb_bench.comparisons import paired_comparison, build_comparisons, paired_conditions
from perturb_bench.metrics import METRIC_DIRECTION


def scores(n_drugs=8, n_rep=3):
    rows = []
    for context in ["A", "B", "C"]:
        for drug in range(n_drugs):
            for dose in [1, 10]:
                for partition in range(n_rep):
                    # Cluster-correlated advantages, with variation within conditions.
                    advantage = (drug - 2) / 20 + partition / 100 + (context == "C") / 50
                    for baseline, value in [("ridge", 0.2 + advantage), ("per_drug_mean_delta", 0.2)]:
                        row = dict(split="held_out_context", fold=context,
                                   cond_id=f"{context}/{drug}/{dose}", compound=f"drug{drug}",
                                   dose=dose, time=24, cell_line=context, partition=partition,
                                   baseline=baseline)
                        row.update({metric: value for metric in METRIC_DIRECTION})
                        rows.append(row)
    return pd.DataFrame(rows)


def test_repeated_identical_partitions_do_not_narrow_interval():
    original = scores()
    duplicate = original.copy()
    duplicate.partition += original.partition.max() + 1
    a = paired_comparison(original, "pearson_r")
    b = paired_comparison(pd.concat([original, duplicate]), "pearson_r")
    for key in ["difference", "ci_low", "ci_high"]:
        assert a[key] == pytest.approx(b[key], abs=1e-15)
    assert a["n_conditions"] == b["n_conditions"] == 48
    assert b["n_usable_partition_pairs"] == 2 * a["n_usable_partition_pairs"]


def test_identical_predictions_zero_difference():
    data = scores()
    data["pearson_r"] = 0.3
    result = paired_comparison(data, "pearson_r")
    assert (result["difference"], result["ci_low"], result["ci_high"]) == (0, 0, 0)
    assert result["evidence"] == "inconclusive"


def test_swapping_baselines_reverses_difference_and_interval():
    a = paired_comparison(scores(), "pearson_r", seed=16)
    b = paired_comparison(scores(), "pearson_r", "per_drug_mean_delta", "ridge", seed=16)
    assert a["difference"] == pytest.approx(-b["difference"])
    assert a["ci_low"] == pytest.approx(-b["ci_high"])
    assert a["ci_high"] == pytest.approx(-b["ci_low"])


def test_missing_and_infinite_scores_use_shared_partitions():
    data = scores(n_drugs=2)
    first = data.cond_id.iloc[0]
    data.loc[(data.cond_id == first) & (data.baseline == "ridge") & (data.partition == 0), "pearson_r"] = np.nan
    data.loc[(data.cond_id == first) & (data.baseline == "per_drug_mean_delta") & (data.partition == 1), "pearson_r"] = np.inf
    pairs, counts = paired_conditions(data, "pearson_r", "ridge", "per_drug_mean_delta")
    observed = pairs[pairs.cond_id == first].iloc[0]
    expected = data[(data.cond_id == first) & (data.partition == 2)]
    assert observed.a == expected[expected.baseline == "ridge"].pearson_r.iloc[0]
    assert counts["n_excluded_partition_pairs"] == 2
    data.loc[data.cond_id == first, "pearson_r"] = np.nan
    result = paired_comparison(data, "pearson_r")
    assert result["n_excluded_conditions"] == 1


def test_equal_context_weight_not_pooled_condition_weight():
    data = scores()
    data = data[~((data.cell_line == "C") & (data.dose == 10))].copy()
    data["pearson_r"] = 0.0
    data.loc[(data.cell_line == "C") & (data.baseline == "ridge"), "pearson_r"] = 0.6
    result = paired_comparison(data, "pearson_r")
    assert result["difference"] == pytest.approx(0.2)
    assert result["ci_low"] == pytest.approx(0.2)
    assert result["ci_high"] == pytest.approx(0.2)


def test_cluster_bootstrap_keeps_same_drug_together_across_contexts():
    data = scores()
    data["pearson_r"] = 0.0
    for drug in range(8):
        mask = (data.compound == f"drug{drug}") & (data.baseline == "ridge")
        data.loc[mask & (data.cell_line == "A"), "pearson_r"] = drug / 10
        data.loc[mask & (data.cell_line == "B"), "pearson_r"] = -drug / 10
    result = paired_comparison(data, "pearson_r")
    # Independent resampling of contexts would destroy this cancellation.
    assert result["ci_low"] == pytest.approx(0, abs=1e-15)
    assert result["ci_high"] == pytest.approx(0, abs=1e-15)


def test_fewer_than_two_drugs_suppresses_inference():
    result = paired_comparison(scores(n_drugs=1), "pearson_r")
    assert np.isnan(result["ci_low"])
    assert result["status"] == "insufficient_data"
    assert result["evidence"] == "inconclusive"


def test_duplicate_identifiers_fail_loudly():
    data = scores()
    with pytest.raises(ValueError, match="duplicate"):
        paired_comparison(pd.concat([data, data.iloc[:1]]), "pearson_r")


def test_direction_and_primary_endpoint_and_determinism():
    data = scores()
    data.loc[data.baseline == "ridge", list(METRIC_DIRECTION)] = 1.0
    a = build_comparisons(data, seed=2)
    b = build_comparisons(data.sample(frac=1, random_state=3), seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert (a.endpoint == "primary").sum() == 1
    assert a[a.endpoint == "primary"].iloc[0].metric == "pearson_r"
    assert set(a[a.metric == "pearson_r"].evidence) == {"ridge"}
    assert set(a[a.metric == "mae"].evidence) == {"per_drug_mean_delta"}


def test_entire_context_missing_does_not_change_estimand():
    data = scores()
    data.loc[data.cell_line == "B", "pearson_r"] = np.nan
    result = paired_comparison(data, "pearson_r")
    assert np.isnan(result["difference"])
    assert result["n_contexts"] == 3
    assert result["status"] == "insufficient_data"


def test_extra_doses_are_not_independent_drugs():
    data = scores()
    copies = [data]
    for i in range(1, 6):
        extra = data.copy()
        extra.cond_id += f"_copy{i}"
        extra.dose += i * 100
        copies.append(extra)
    a = paired_comparison(data, "pearson_r")
    b = paired_comparison(pd.concat(copies), "pearson_r")
    assert b["n_conditions"] == 6 * a["n_conditions"]
    assert a["n_drugs"] == b["n_drugs"]
    assert a["ci_low"] == pytest.approx(b["ci_low"])
    assert a["ci_high"] == pytest.approx(b["ci_high"])


def test_sparse_context_coverage_suppresses_interval():
    data = scores(n_drugs=20)
    data = data[~((data.cell_line == "C") & ~data.compound.isin(["drug0", "drug1"]))]
    result = paired_comparison(data, "pearson_r")
    assert result["status"] == "insufficient_cluster_coverage"
    assert np.isnan(result["ci_low"])


def test_absent_baseline_and_empty_input_are_inconclusive():
    data = scores()
    for subset in [data[data.baseline == "ridge"], data.iloc[:0]]:
        result = paired_comparison(subset, "pearson_r")
        assert result["status"] == "insufficient_data"
        assert result["n_conditions"] == 0

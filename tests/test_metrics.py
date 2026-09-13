import numpy as np
import pandas as pd
import pytest

from perturb_bench.metrics import (
    METRIC_DIRECTION,
    aggregate,
    bootstrap_ci_mean,
    normalize_score,
    score_conditions,
)


def test_pearson_r_self_and_negation():
    rng = np.random.default_rng(0)
    v = rng.normal(size=50)
    control = np.zeros((1, 50))
    de_mask = np.ones((1, 50), dtype=bool)

    df_self = score_conditions(
        pred=v[None, :], true=v[None, :], control=control, de_mask=de_mask, representation="delta"
    )
    assert df_self["pearson_r"].iloc[0] == pytest.approx(1.0, abs=1e-9)

    df_neg = score_conditions(
        pred=(-v)[None, :], true=v[None, :], control=control, de_mask=de_mask, representation="delta"
    )
    assert df_neg["pearson_r"].iloc[0] == pytest.approx(-1.0, abs=1e-9)


def test_normalize_score_higher_better():
    # floor < ceiling (e.g. pearson_r)
    assert normalize_score(0.2, floor=0.2, ceiling=0.9) == pytest.approx(0.0)
    assert normalize_score(0.9, floor=0.2, ceiling=0.9) == pytest.approx(1.0)
    mid = normalize_score(0.55, floor=0.2, ceiling=0.9)
    assert 0.0 < mid < 1.0


def test_normalize_score_lower_better():
    # ceiling < floor (e.g. mae, pert_discrimination): same formula must still work.
    assert normalize_score(1.5, floor=1.5, ceiling=0.1) == pytest.approx(0.0)
    assert normalize_score(0.1, floor=1.5, ceiling=0.1) == pytest.approx(1.0)
    mid = normalize_score(0.8, floor=1.5, ceiling=0.1)
    assert 0.0 < mid < 1.0


def test_normalize_score_degenerate_returns_nan():
    result = normalize_score(0.5, floor=0.3, ceiling=0.3)
    assert np.isnan(result)


def test_pert_discrimination_perfect_prediction():
    rng = np.random.default_rng(1)
    n_cond, n_genes = 8, 20
    true = rng.normal(size=(n_cond, n_genes)) * 3.0
    pred = true.copy()  # perfect: pred_i == true_i exactly, and all distinct from each other
    control = np.zeros((n_cond, n_genes))
    de_mask = np.ones((n_cond, n_genes), dtype=bool)

    df = score_conditions(pred=pred, true=true, control=control, de_mask=de_mask, representation="delta")
    mean_rank = df["pert_discrimination"].mean()
    assert mean_rank == pytest.approx(0.0, abs=1e-9)


def test_pert_discrimination_identical_predictions_is_chance_not_perfect():
    # This is the B1 (no_change) case: every predicted profile is identical (all zeros,
    # or any constant vector), so every condition is equidistant from every true
    # profile. A naive argsort-based rank would hand this a ~0 (great) score by
    # accident via tie-breaking order; with average-rank tie handling it must come
    # out at ~0.5 (chance), not ~0.
    rng = np.random.default_rng(2)
    n_cond, n_genes = 10, 15
    true = rng.normal(size=(n_cond, n_genes)) * 2.0
    pred = np.zeros_like(true)  # identical (all-zero) prediction for every condition
    control = np.zeros((n_cond, n_genes))
    de_mask = np.ones((n_cond, n_genes), dtype=bool)

    df = score_conditions(pred=pred, true=true, control=control, de_mask=de_mask, representation="delta")
    mean_rank = df["pert_discrimination"].mean()
    assert mean_rank == pytest.approx(0.5, abs=1e-6)


def test_absolute_vs_delta_inflation_demonstration():
    """The inflation demonstration from SPEC §7/§11 as a test: a large shared baseline
    expression dominates pearson_r under the "absolute" representation even for a
    completely uninformative (zero) prediction, while the "delta" representation
    correctly reports this prediction as uncorrelated with the true effect.
    """
    rng = np.random.default_rng(3)
    n_cond, n_genes = 12, 200

    # Large shared baseline expression profile (same shape across conditions, big values)
    baseline_shape = rng.normal(loc=5.0, scale=2.0, size=n_genes)
    control = np.tile(baseline_shape, (n_cond, 1))

    # True delta: small, condition-specific perturbation signal
    true_delta = rng.normal(loc=0.0, scale=0.3, size=(n_cond, n_genes))

    # B1-style prediction: zero delta everywhere
    pred_delta = np.zeros((n_cond, n_genes))

    de_mask = np.ones((n_cond, n_genes), dtype=bool)

    df_delta = score_conditions(
        pred=pred_delta, true=true_delta, control=control, de_mask=de_mask, representation="delta"
    )
    df_absolute = score_conditions(
        pred=pred_delta, true=true_delta, control=control, de_mask=de_mask, representation="absolute"
    )

    mean_r_delta = df_delta["pearson_r"].mean()
    mean_r_absolute = df_absolute["pearson_r"].mean()

    # On delta: zero prediction vs small noisy true delta -> near-zero/undefined correlation,
    # nothing to be proud of.
    assert abs(mean_r_delta) < 0.3 or np.isnan(mean_r_delta)

    # On absolute: (control + 0) vs (control + true_delta) is dominated by the shared
    # baseline_shape term in both vectors, so correlation is driven far higher even
    # though the prediction carries zero information about the actual perturbation.
    assert mean_r_absolute > 0.9

    # Report the actual numbers for the write-up (also asserts the qualitative claim
    # "dramatically higher" quantitatively, not just by eyeballing the two bounds above).
    assert mean_r_absolute - (mean_r_delta if not np.isnan(mean_r_delta) else 0.0) > 0.6


def test_de_gene_skip_counting_no_de_genes():
    n_cond, n_genes = 3, 10
    rng = np.random.default_rng(4)
    pred = rng.normal(size=(n_cond, n_genes))
    true = rng.normal(size=(n_cond, n_genes))
    control = np.zeros((n_cond, n_genes))

    de_mask = np.zeros((n_cond, n_genes), dtype=bool)  # condition 0: no DE genes
    de_mask[1, :] = True  # condition 1: all genes DE (>=2)
    de_mask[2, 0] = True  # condition 2: exactly 1 DE gene -> still below threshold of 2

    df = score_conditions(pred=pred, true=true, control=control, de_mask=de_mask, representation="delta")

    assert np.isnan(df["pearson_r_de"].iloc[0])
    assert np.isnan(df["direction_acc_de"].iloc[0])
    assert np.isnan(df["pearson_r_de"].iloc[2])  # only 1 DE gene -> also skipped
    assert not np.isnan(df["pearson_r_de"].iloc[1])

    agg = aggregate(df)
    # 2 of 3 conditions skipped for pearson_r_de (0 DE genes, 1 DE gene)
    assert agg.loc["pearson_r_de", "n_skipped"] == 2
    assert agg.loc["pearson_r_de", "n"] == 1


def test_aggregate_returns_std_and_n():
    df = pd.DataFrame(
        {
            "pearson_r": [0.1, 0.5, 0.9, np.nan],
            "mae": [1.0, 2.0, 3.0, 4.0],
        }
    )
    agg = aggregate(df)
    assert set(agg.columns) == {"mean", "std", "n", "n_skipped"}
    assert agg.loc["pearson_r", "n"] == 3
    assert agg.loc["pearson_r", "n_skipped"] == 1
    assert agg.loc["mae", "n"] == 4
    assert agg.loc["mae", "n_skipped"] == 0
    assert agg.loc["pearson_r", "std"] > 0
    assert agg.loc["pearson_r", "mean"] == pytest.approx(np.mean([0.1, 0.5, 0.9]))


def test_direction_acc_de_zero_prediction_does_not_beat_chance():
    # A zero prediction should not be rewarded for "matching" nonzero true DE signs.
    n_cond, n_genes = 1, 6
    true = np.array([[1.0, -1.0, 2.0, -3.0, 0.5, -0.5]])
    pred = np.zeros_like(true)
    control = np.zeros_like(true)
    de_mask = np.ones((n_cond, n_genes), dtype=bool)

    df = score_conditions(pred=pred, true=true, control=control, de_mask=de_mask, representation="delta")
    # sign(0) matches none of the nonzero true signs -> direction_acc_de should be 0.0
    assert df["direction_acc_de"].iloc[0] == pytest.approx(0.0)


def test_metric_direction_mapping_complete():
    expected_metrics = {"pearson_r", "mae", "pearson_r_de", "direction_acc_de", "pert_discrimination"}
    assert set(METRIC_DIRECTION.keys()) == expected_metrics
    assert METRIC_DIRECTION["mae"] == "lower_better"
    assert METRIC_DIRECTION["pert_discrimination"] == "lower_better"
    assert METRIC_DIRECTION["pearson_r"] == "higher_better"


# ---------------------------------------------------------------------------
# fix #15 / REVIEW.md L-2: _pert_discrimination was rewritten as one cdist +
# rankdata(axis=1) call instead of an O(n^2*g) python loop. Re-run the
# existing small-n tie tests above (still must pass unchanged) AND add a
# larger-n tie test so a regression to a per-row tie-breaking bug (e.g. an
# argsort that resolves ties by index instead of averaging) cannot hide at
# small n.
# ---------------------------------------------------------------------------

def test_pert_discrimination_identical_predictions_is_chance_at_larger_n():
    # B1-style: every predicted profile is the SAME zero vector, at n=300
    # instead of n=10/8 (the pre-vectorization tests), so a regression that
    # only happens to behave at small n (e.g. an off-by-one in a chunked
    # cdist call) cannot hide. The per-row ranking pred[i] sees is identical
    # for every i here (pred[i] is the same vector regardless of i), so each
    # condition's own normalized rank tracks where ||true[i]|| falls among
    # all ||true[j]|| -- NOT literally 0.5 per row -- but it must average to
    # 0.5 over many random conditions (chance), not collapse toward 0 the
    # way a naive first-tie-wins argsort would.
    rng = np.random.default_rng(42)
    n_cond, n_genes = 300, 50
    true = rng.normal(size=(n_cond, n_genes)) * 2.0
    pred = np.zeros_like(true)
    control = np.zeros((n_cond, n_genes))
    de_mask = np.ones((n_cond, n_genes), dtype=bool)

    df = score_conditions(pred=pred, true=true, control=control, de_mask=de_mask, representation="delta")
    mean_rank = df["pert_discrimination"].mean()
    assert mean_rank == pytest.approx(0.5, abs=0.05)


def test_pert_discrimination_exact_ties_within_a_row_use_average_rank_at_larger_n():
    # Construct REAL within-row ties: many true profiles are exact duplicates
    # of each other, so for any pred[i], several true[j] are equidistant.
    # Average-rank tie handling must give every member of a tied group the
    # same (averaged) rank -- a naive argsort would hand one of them an
    # artificially low (better) rank by index-order luck.
    rng = np.random.default_rng(11)
    n_groups, group_size, n_genes = 30, 5, 20  # n_cond = 150
    unique_profiles = rng.normal(size=(n_groups, n_genes)) * 3.0
    true = np.repeat(unique_profiles, group_size, axis=0)
    pred = true + rng.normal(scale=0.01, size=true.shape)  # tiny distinguishing noise on pred only
    control = np.zeros_like(true)
    de_mask = np.ones_like(true, dtype=bool)

    df = score_conditions(pred=pred, true=true, control=control, de_mask=de_mask, representation="delta")
    ranks = df["pert_discrimination"].to_numpy()
    # Within each group of exact-duplicate true profiles, every member's
    # distance-from-pred[i]-to-every-true[j] row sees the SAME tied cluster
    # of distances for the other (group_size - 1) duplicates, so their
    # average rank contribution from the tie should be identical up to the
    # small pred-side noise -- in particular, none of them should be pinned
    # to rank 0 (perfect) just because of array-index ordering luck.
    for g in range(n_groups):
        group_ranks = ranks[g * group_size:(g + 1) * group_size]
        assert group_ranks.std() < 0.2  # noise-sized spread, not index-order-sized


def test_pert_discrimination_vectorized_matches_naive_loop_reference():
    """cdist+rankdata(axis=1) must match a naive per-row reference implementation
    bit-for-bit (modulo float tolerance) on non-tied, realistic data -- guards
    against the vectorized rewrite silently transposing an axis or mixing up
    which profile is "pred" vs "true" in the distance matrix."""
    rng = np.random.default_rng(5)
    n_cond, n_genes = 40, 12
    true = rng.normal(size=(n_cond, n_genes))
    pred = true + rng.normal(scale=0.5, size=(n_cond, n_genes))
    control = np.zeros((n_cond, n_genes))
    de_mask = np.ones((n_cond, n_genes), dtype=bool)

    from scipy.stats import rankdata as _rankdata

    def naive(pred, true):
        n = pred.shape[0]
        out = np.empty(n)
        for i in range(n):
            dists = np.linalg.norm(true - pred[i], axis=1)
            ranks = _rankdata(dists, method="average")
            out[i] = (ranks[i] - 1.0) / (n - 1.0)
        return out

    expected = naive(pred, true)
    df = score_conditions(pred=pred, true=true, control=control, de_mask=de_mask, representation="delta")
    np.testing.assert_allclose(df["pert_discrimination"].to_numpy(), expected, atol=1e-9)


# ---------------------------------------------------------------------------
# bootstrap_ci_mean (M-2): used by run.py for winner/"beats" claims. Must be
# deterministic given a seed, return a degenerate interval at n=1, NaN at
# n=0, and bracket the true mean with high probability at larger n.
# ---------------------------------------------------------------------------

def test_bootstrap_ci_mean_deterministic_given_seed():
    vals = np.array([0.1, 0.4, 0.9, 0.2, 0.5, np.nan, 0.3])
    lo1, hi1 = bootstrap_ci_mean(vals, seed=123, n_boot=500)
    lo2, hi2 = bootstrap_ci_mean(vals, seed=123, n_boot=500)
    assert lo1 == lo2 and hi1 == hi2
    lo3, hi3 = bootstrap_ci_mean(vals, seed=124, n_boot=500)
    assert (lo1, hi1) != (lo3, hi3)  # different seed -> (almost certainly) different resample


def test_bootstrap_ci_mean_edge_cases():
    assert all(np.isnan(x) for x in bootstrap_ci_mean(np.array([np.nan, np.nan]), seed=0))
    lo, hi = bootstrap_ci_mean(np.array([0.7]), seed=0)
    assert lo == pytest.approx(0.7) and hi == pytest.approx(0.7)


def test_bootstrap_ci_mean_brackets_true_mean_typically():
    rng = np.random.default_rng(9)
    vals = rng.normal(loc=2.0, scale=0.3, size=200)
    lo, hi = bootstrap_ci_mean(vals, seed=0, n_boot=2000)
    assert lo < 2.0 < hi

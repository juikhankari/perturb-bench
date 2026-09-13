import numpy as np
import pytest

from perturb_bench.ceiling import compute_ceiling


def test_compute_ceiling_perfect_halves_gives_max_score():
    # If the two halves are IDENTICAL (zero measurement noise), the ceiling should
    # show perfect agreement: pearson_r ~ 1, mae ~ 0, pert_discrimination ~ 0.
    rng = np.random.default_rng(0)
    n_rep, n_cond, n_genes = 3, 6, 25
    base = rng.normal(size=(n_cond, n_genes)) * 2.0

    halves = np.empty((n_rep, 2, n_cond, n_genes), dtype=np.float32)
    for r in range(n_rep):
        halves[r, 0] = base
        halves[r, 1] = base

    control = np.zeros((n_cond, n_genes))
    de_mask = np.ones((n_cond, n_genes), dtype=bool)
    genes = np.arange(n_genes)

    agg = compute_ceiling(halves, control, de_mask, genes, representation="delta")

    assert agg.loc["pearson_r", "mean"] == pytest.approx(1.0, abs=1e-6)
    assert agg.loc["mae", "mean"] == pytest.approx(0.0, abs=1e-6)
    assert agg.loc["pert_discrimination", "mean"] == pytest.approx(0.0, abs=1e-6)
    # aggregated over n_rep * n_cond scored rows
    assert agg.loc["pearson_r", "n"] == n_rep * n_cond


def test_compute_ceiling_noisy_halves_has_variance_and_is_below_perfect():
    rng = np.random.default_rng(1)
    n_rep, n_cond, n_genes = 4, 5, 30
    true_signal = rng.normal(size=(n_cond, n_genes)) * 2.0

    halves = np.empty((n_rep, 2, n_cond, n_genes), dtype=np.float32)
    for r in range(n_rep):
        noise0 = rng.normal(scale=1.5, size=(n_cond, n_genes))
        noise1 = rng.normal(scale=1.5, size=(n_cond, n_genes))
        halves[r, 0] = true_signal + noise0
        halves[r, 1] = true_signal + noise1

    control = np.zeros((n_cond, n_genes))
    de_mask = np.ones((n_cond, n_genes), dtype=bool)
    genes = np.arange(n_genes)

    agg = compute_ceiling(halves, control, de_mask, genes, representation="delta")

    assert agg.loc["pearson_r", "mean"] < 1.0
    assert agg.loc["pearson_r", "std"] >= 0.0
    assert agg.loc["mae", "mean"] > 0.0


def test_compute_ceiling_gene_subset_is_respected():
    rng = np.random.default_rng(2)
    n_rep, n_cond, n_genes = 2, 4, 40
    base = rng.normal(size=(n_cond, n_genes))
    halves = np.empty((n_rep, 2, n_cond, n_genes), dtype=np.float32)
    for r in range(n_rep):
        halves[r, 0] = base
        halves[r, 1] = base

    control = np.zeros((n_cond, n_genes))
    de_mask = np.ones((n_cond, n_genes), dtype=bool)
    genes = np.arange(10)  # subset of the full gene axis

    agg = compute_ceiling(halves, control, de_mask, genes, representation="delta")
    # should run without shape errors and still show perfect agreement on the subset
    assert agg.loc["pearson_r", "mean"] == pytest.approx(1.0, abs=1e-6)


def test_compute_ceiling_absolute_representation_runs():
    rng = np.random.default_rng(3)
    n_rep, n_cond, n_genes = 2, 3, 15
    base = rng.normal(size=(n_cond, n_genes))
    halves = np.empty((n_rep, 2, n_cond, n_genes), dtype=np.float32)
    for r in range(n_rep):
        halves[r, 0] = base + rng.normal(scale=0.1, size=(n_cond, n_genes))
        halves[r, 1] = base + rng.normal(scale=0.1, size=(n_cond, n_genes))

    control = rng.normal(loc=5.0, size=(n_cond, n_genes))
    de_mask = np.ones((n_cond, n_genes), dtype=bool)
    genes = np.arange(n_genes)

    agg = compute_ceiling(halves, control, de_mask, genes, representation="absolute")
    assert "pearson_r" in agg.index
    assert not agg["mean"].isna().all()


def test_compute_ceiling_rejects_bad_representation():
    halves = np.zeros((1, 2, 1, 1))
    control = np.zeros((1, 1))
    de_mask = np.ones((1, 1), dtype=bool)
    genes = np.arange(1)
    with pytest.raises(ValueError):
        compute_ceiling(halves, control, de_mask, genes, representation="bogus")

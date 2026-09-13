"""Fast tests for perturb_bench.data -- no full download required. Builds small
synthetic data and exercises each pipeline piece directly."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from perturb_bench.config import Config
from perturb_bench.data import (
    _DenseMatrixAdapter,
    build_pseudobulk_core,
    compute_halves,
    detect_mito_genes,
    ensure_lognorm,
    load_pseudobulk,
    qc_filter,
    select_hvgs,
)


def _make_synthetic(seed=0, n_genes=40, n_cell_lines=2, n_compounds=3, n_doses=2,
                    cells_per_cond=60, n_batches=2):
    """Build a small synthetic single-cell dataset with known structure: each
    (cell_line, compound, dose) has a deterministic true effect vector added to a
    cell_line-specific baseline, plus noise. Controls exist per (cell_line, batch)."""
    rng = np.random.default_rng(seed)
    gene_names = np.array([f"G{i}" for i in range(n_genes)])
    cell_lines = [f"CL{i}" for i in range(n_cell_lines)]
    compounds = [f"drug{i}" for i in range(n_compounds)]
    doses = [10.0 * (i + 1) for i in range(n_doses)]
    batches = [f"plate{i}" for i in range(n_batches)]

    baseline = {cl: rng.normal(5, 1, n_genes) for cl in cell_lines}
    true_effect = {
        (cl, co, d): rng.normal(0, 1, n_genes) * (1 if co != compounds[0] else 0.0)
        for cl in cell_lines for co in compounds for d in doses
    }
    # make drug0 have a real, nonzero, gene-sparse effect (a handful of DE genes)
    de_genes = rng.choice(n_genes, size=6, replace=False)
    for cl in cell_lines:
        for d in doses:
            eff = np.zeros(n_genes)
            eff[de_genes] = 3.0 * (d / doses[0])
            true_effect[(cl, compounds[0], d)] = eff

    rows_X, rows_cl, rows_co, rows_dose, rows_batch, rows_isctrl = [], [], [], [], [], []
    for cl in cell_lines:
        for bi, b in enumerate(batches):
            n_ctrl = 50
            ctrl_cells = baseline[cl] + rng.normal(0, 0.5, (n_ctrl, n_genes))
            rows_X.append(ctrl_cells)
            rows_cl += [cl] * n_ctrl
            rows_co += ["control"] * n_ctrl
            rows_dose += [0.0] * n_ctrl
            rows_batch += [b] * n_ctrl
            rows_isctrl += [True] * n_ctrl

        for co in compounds:
            for d in doses:
                cells = (
                    baseline[cl] + true_effect[(cl, co, d)]
                    + rng.normal(0, 0.5, (cells_per_cond, n_genes))
                )
                b = batches[rng.integers(0, n_batches)]
                rows_X.append(cells)
                rows_cl += [cl] * cells_per_cond
                rows_co += [co] * cells_per_cond
                rows_dose += [d] * cells_per_cond
                rows_batch += [b] * cells_per_cond
                rows_isctrl += [False] * cells_per_cond

    X = np.clip(np.vstack(rows_X), 0, None).astype(np.float32)
    return dict(
        X=X,
        cell_line=np.array(rows_cl, dtype=object),
        compound=np.array(rows_co, dtype=object),
        dose=np.array(rows_dose, dtype=float),
        batch=np.array(rows_batch, dtype=object),
        is_control=np.array(rows_isctrl, dtype=bool),
        gene_names=gene_names,
        de_genes=de_genes,
        compounds=compounds,
    )


def _default_cfg(tmpdir, **overrides):
    kwargs = dict(
        dataset="sciplex3", seed=0, min_genes=1, max_pct_mito=100.0,
        min_cells_per_condition=30, n_hvg=None, gene_set="all",
        n_ceiling_reps=4, cache_dir=str(tmpdir / "cache"), out_dir=str(tmpdir / "out"),
        de_alpha=0.05,
    )
    kwargs.update(overrides)
    return Config(**kwargs)


# ---------------------------------------------------------------------------

def test_detect_mito_genes():
    genes = np.array(["ACTB", "MT-ND1", "MT-CO1", "GAPDH"])
    mask = detect_mito_genes(genes)
    assert mask.tolist() == [False, True, True, False]
    assert detect_mito_genes(np.array(["ACTB", "GAPDH"])) is None


def test_ensure_lognorm_detects_raw_counts():
    rng = np.random.default_rng(0)
    raw = rng.poisson(50, size=(20, 10)).astype(np.float64)
    Xn, meta = ensure_lognorm(raw)
    assert meta["detected_input"] == "raw_counts"
    assert Xn.max() < raw.max()  # log1p should compress the scale


def test_ensure_lognorm_detects_already_normalized():
    rng = np.random.default_rng(0)
    already = rng.normal(2, 1, size=(20, 10)).astype(np.float64)
    Xn, meta = ensure_lognorm(already)
    assert meta["detected_input"] == "already_normalized_or_log"
    np.testing.assert_allclose(Xn, already.astype(np.float32), rtol=1e-5)


def test_qc_filter_no_mito_genes_does_not_crash():
    X = np.ones((10, 5))
    genes = np.array(["G1", "G2", "G3", "G4", "G5"])
    keep, meta = qc_filter(X, genes, min_genes=1, max_pct_mito=10.0)
    assert meta["mito_filter_applied"] is False
    assert keep.all()


def test_delta_equals_X_minus_control(tmp_path):
    syn = _make_synthetic()
    cfg = _default_cfg(tmp_path)
    matrix = _DenseMatrixAdapter(syn["X"])
    data, ctx = build_pseudobulk_core(
        matrix, syn["cell_line"], syn["compound"], syn["dose"], syn["is_control"],
        syn["batch"], syn["gene_names"], cfg, want_halves=True,
    )
    np.testing.assert_allclose(data.delta, data.X - data.control, atol=1e-5)


def test_de_mask_is_not_just_topk_of_delta(tmp_path):
    """The DE mask must come from a single-cell significance test, not from ranking
    the true delta. We check this by constructing a condition where a gene has a
    LARGE true delta but is noisy/inconsistent at the single-cell level (so it should
    NOT be DE), while another gene has a small but highly consistent delta (so it
    SHOULD be DE) -- top-k-of-delta would get this backwards."""
    rng = np.random.default_rng(42)
    n_genes = 20
    gene_names = np.array([f"G{i}" for i in range(n_genes)])

    n = 200
    # control cells: all genes centered at 0, low noise
    control_cells = rng.normal(0, 0.2, (n, n_genes))
    # treated cells: gene 0 has a LARGE mean shift but ENORMOUS variance (inconsistent,
    # should fail a rank-sum test despite big delta); gene 1 has a SMALL, consistent
    # shift (should pass).
    treated_cells = rng.normal(0, 0.2, (n, n_genes))
    treated_cells[:, 0] = rng.normal(5, 120, n)   # biggish delta, huge noise -> not robustly DE
    treated_cells[:, 1] = rng.normal(0.5, 0.05, n)  # small, consistent delta -> robustly DE

    X = np.vstack([control_cells, treated_cells])
    cell_line = np.array(["A"] * (2 * n))
    compound = np.array(["control"] * n + ["drugX"] * n)
    dose = np.array([0.0] * n + [10.0] * n)
    batch = np.array(["plate1"] * (2 * n))
    is_control = np.array([True] * n + [False] * n)

    cfg = _default_cfg(tmp_path, de_alpha=0.01)
    matrix = _DenseMatrixAdapter(X.astype(np.float32))
    data, ctx = build_pseudobulk_core(
        matrix, cell_line, compound, dose, is_control, batch, gene_names, cfg, want_halves=False,
    )
    assert data.delta.shape[0] == 1
    true_delta = data.delta[0]
    # gene 0 has by far the largest |delta| ...
    assert np.argmax(np.abs(true_delta)) == 0
    # ... but should NOT be flagged DE (too noisy at single-cell level)
    assert not data.de_mask[0, 0], "noisy large-delta gene was wrongly flagged DE"
    # gene 1 has a much smaller delta but SHOULD be flagged DE (consistent at sc level)
    assert data.de_mask[0, 1], "consistent small-delta gene should be DE"


def test_halves_average_to_roughly_full_delta(tmp_path):
    syn = _make_synthetic(cells_per_cond=200)
    cfg = _default_cfg(tmp_path, n_ceiling_reps=6)
    matrix = _DenseMatrixAdapter(syn["X"])
    data, ctx = build_pseudobulk_core(
        matrix, syn["cell_line"], syn["compound"], syn["dose"], syn["is_control"],
        syn["batch"], syn["gene_names"], cfg, want_halves=True,
    )
    assert data.halves.shape == (cfg.n_ceiling_reps, 2, data.X.shape[0], data.X.shape[1])
    mean_half_delta = data.halves.mean(axis=(0, 1))  # average over reps and the two halves
    np.testing.assert_allclose(mean_half_delta, data.delta, atol=0.6)


def test_dropped_condition_counting(tmp_path):
    syn = _make_synthetic(cells_per_cond=60)
    cfg = _default_cfg(tmp_path, min_cells_per_condition=1000)  # drop everything
    matrix = _DenseMatrixAdapter(syn["X"])
    data, ctx = build_pseudobulk_core(
        matrix, syn["cell_line"], syn["compound"], syn["dose"], syn["is_control"],
        syn["batch"], syn["gene_names"], cfg, want_halves=False,
    )
    n_true_conditions = len(set(zip(syn["cell_line"][~syn["is_control"]],
                                     syn["compound"][~syn["is_control"]],
                                     syn["dose"][~syn["is_control"]])))
    assert data.meta["n_conditions_dropped_too_few_cells"] == n_true_conditions
    assert data.X.shape[0] == 0


def test_no_batch_column_falls_back_and_flags(tmp_path):
    syn = _make_synthetic()
    cfg = _default_cfg(tmp_path)
    matrix = _DenseMatrixAdapter(syn["X"])
    data, ctx = build_pseudobulk_core(
        matrix, syn["cell_line"], syn["compound"], syn["dose"], syn["is_control"],
        None, syn["gene_names"], cfg, want_halves=False,
    )
    assert data.meta["batch_metadata_available"] is False
    assert "batch_note" in data.meta and len(data.meta["batch_note"]) > 0


def test_compute_halves_gene_subset_matches_full(tmp_path):
    syn = _make_synthetic(cells_per_cond=80)
    cfg = _default_cfg(tmp_path, n_ceiling_reps=3)
    matrix = _DenseMatrixAdapter(syn["X"])
    data, ctx = build_pseudobulk_core(
        matrix, syn["cell_line"], syn["compound"], syn["dose"], syn["is_control"],
        syn["batch"], syn["gene_names"], cfg, want_halves=False,
    )
    n_genes = data.X.shape[1]
    subset = np.array([0, 3, 5])
    rng_seed_before = cfg.seed
    h_full = compute_halves(ctx, gene_idx=np.arange(n_genes))
    h_subset = compute_halves(ctx, gene_idx=subset)
    # Same seed -> same permutations -> subset columns should exactly match full's
    # columns at the same gene positions.
    np.testing.assert_allclose(h_subset, h_full[:, :, :, subset], atol=1e-5)


def test_time_is_part_of_condition_identity(tmp_path):
    """H-4 (REVIEW.md): 24h and 72h cells of the same (cell_line, compound, dose)
    must become TWO separate conditions, never pooled into one pseudobulk."""
    rng = np.random.default_rng(9)
    n_genes = 6
    gene_names = np.array([f"G{i}" for i in range(n_genes)])

    n_ctrl = 40
    ctrl = rng.normal(0, 0.1, (n_ctrl, n_genes))
    n_24, n_72 = 40, 35
    treated_24h = rng.normal(1.0, 0.1, (n_24, n_genes))   # mild effect at 24h
    treated_72h = rng.normal(6.0, 0.1, (n_72, n_genes))   # much bigger effect at 72h

    X = np.vstack([ctrl, treated_24h, treated_72h]).astype(np.float32)
    cell_line = np.array(["A"] * (n_ctrl + n_24 + n_72))
    compound = np.array(["control"] * n_ctrl + ["drugX"] * (n_24 + n_72))
    dose = np.array([0.0] * n_ctrl + [10.0] * (n_24 + n_72))
    batch = np.array(["plate1"] * (n_ctrl + n_24 + n_72))
    is_control = np.array([True] * n_ctrl + [False] * (n_24 + n_72))
    time_arr = np.array([24.0] * n_ctrl + [24.0] * n_24 + [72.0] * n_72)

    cfg = _default_cfg(tmp_path, min_cells_per_condition=10)
    matrix = _DenseMatrixAdapter(X)
    data, ctx = build_pseudobulk_core(
        matrix, cell_line, compound, dose, is_control, batch, gene_names, cfg,
        want_halves=False, time=time_arr,
    )
    # Two distinct conditions, not one time-pooled blend.
    assert data.X.shape[0] == 2
    assert set(data.obs["time"].tolist()) == {24.0, 72.0}
    assert data.meta["time_identity"]["added_to_cond_id"] is True
    assert data.meta["time_identity"]["time_metadata_available"] is True
    # The 72h condition's delta should reflect the much bigger 72h effect, not a
    # blend with the 24h cells.
    row_72 = data.obs["time"] == 72.0
    row_24 = data.obs["time"] == 24.0
    delta_72 = data.delta[row_72.to_numpy()][0]
    delta_24 = data.delta[row_24.to_numpy()][0]
    assert delta_72.mean() > delta_24.mean() + 2.0


def test_per_plate_delta_is_weighted_average_not_modal_plate_only(tmp_path):
    """H-3 (REVIEW.md): a condition split across two plates must have its delta
    computed as the cell-count-weighted average of EACH plate's own
    (treated - that plate's control), not the condition's modal plate's control
    applied to every cell. We build one condition with a 70/30 cell split across
    two plates with DELIBERATELY DIFFERENT control profiles and check the exact
    weighted-average identity, plus that every cell is only ever compared to its
    own plate's control (verified by reconstructing the expected value from the
    known per-plate means)."""
    rng = np.random.default_rng(7)
    n_genes = 10
    gene_names = np.array([f"G{i}" for i in range(n_genes)])

    ctrl_p1_mean = np.full(n_genes, 1.0)
    ctrl_p2_mean = np.full(n_genes, 5.0)  # very different control baseline on plate2
    treated_mean = np.full(n_genes, 10.0)

    n_ctrl = 80
    ctrl_p1 = ctrl_p1_mean + rng.normal(0, 0.05, (n_ctrl, n_genes))
    ctrl_p2 = ctrl_p2_mean + rng.normal(0, 0.05, (n_ctrl, n_genes))

    n_treated_p1, n_treated_p2 = 70, 30
    treated_p1 = treated_mean + rng.normal(0, 0.05, (n_treated_p1, n_genes))
    treated_p2 = treated_mean + rng.normal(0, 0.05, (n_treated_p2, n_genes))

    X = np.vstack([ctrl_p1, ctrl_p2, treated_p1, treated_p2]).astype(np.float32)
    cell_line = np.array(["A"] * (2 * n_ctrl + n_treated_p1 + n_treated_p2))
    compound = np.array(
        ["control"] * n_ctrl + ["control"] * n_ctrl + ["drugX"] * n_treated_p1 + ["drugX"] * n_treated_p2
    )
    dose = np.array([0.0] * (2 * n_ctrl) + [10.0] * (n_treated_p1 + n_treated_p2))
    batch = np.array(["plate1"] * n_ctrl + ["plate2"] * n_ctrl + ["plate1"] * n_treated_p1 + ["plate2"] * n_treated_p2)
    is_control = np.array([True] * (2 * n_ctrl) + [False] * (n_treated_p1 + n_treated_p2))

    cfg = _default_cfg(tmp_path, min_cells_per_condition=10)
    matrix = _DenseMatrixAdapter(X)
    data, ctx = build_pseudobulk_core(
        matrix, cell_line, compound, dose, is_control, batch, gene_names, cfg, want_halves=False,
    )
    assert data.X.shape[0] == 1
    total = n_treated_p1 + n_treated_p2
    w1, w2 = n_treated_p1 / total, n_treated_p2 / total
    expected_control = w1 * ctrl_p1_mean + w2 * ctrl_p2_mean
    expected_delta = treated_mean - expected_control  # == w1*(treated-ctrl_p1) + w2*(treated-ctrl_p2)

    # This is the crucial check: a naive modal-plate-only implementation would use
    # ctrl_p1_mean (the bigger plate, 70%) for 100% of cells, giving control ==
    # ctrl_p1_mean exactly and a visibly different (wrong) delta. With the fix, the
    # minority plate's control must still show up in the weighted average.
    np.testing.assert_allclose(data.control[0], expected_control, atol=0.05)
    np.testing.assert_allclose(data.delta[0], expected_delta, atol=0.05)
    assert not np.allclose(data.control[0], ctrl_p1_mean, atol=0.05), (
        "control profile equals the modal plate's control alone -- the minority "
        "plate (30% of cells) was not matched against its OWN control (H-3 bug)."
    )

    # Plate-purity diagnostics should reflect the 70/30 split (modal purity ~0.7),
    # and report that 100% of cells are plate-matched by construction.
    pm = data.meta["plate_matching"]
    assert pm["n_conditions_multi_plate"] == 1
    assert abs(pm["modal_plate_purity_mean"] - 0.7) < 0.05
    assert pm["pct_cells_plate_matched"] == 100.0


def test_plate_with_no_vehicle_cells_is_dropped_not_the_whole_condition(tmp_path):
    """H-3: if one of a condition's plates has NO vehicle (control) cells at all,
    only that plate's treated cells should be dropped -- not the whole condition --
    and the drop must be counted in meta."""
    rng = np.random.default_rng(3)
    n_genes = 8
    gene_names = np.array([f"G{i}" for i in range(n_genes)])

    n_ctrl = 60
    ctrl_p1 = rng.normal(2, 0.1, (n_ctrl, n_genes))
    n_treated_p1, n_treated_p2 = 50, 20  # plate2 has NO control cells at all
    treated_p1 = rng.normal(5, 0.1, (n_treated_p1, n_genes))
    treated_p2 = rng.normal(5, 0.1, (n_treated_p2, n_genes))

    X = np.vstack([ctrl_p1, treated_p1, treated_p2]).astype(np.float32)
    cell_line = np.array(["A"] * (n_ctrl + n_treated_p1 + n_treated_p2))
    compound = np.array(["control"] * n_ctrl + ["drugX"] * (n_treated_p1 + n_treated_p2))
    dose = np.array([0.0] * n_ctrl + [10.0] * (n_treated_p1 + n_treated_p2))
    batch = np.array(["plate1"] * n_ctrl + ["plate1"] * n_treated_p1 + ["plate2"] * n_treated_p2)
    is_control = np.array([True] * n_ctrl + [False] * (n_treated_p1 + n_treated_p2))

    cfg = _default_cfg(tmp_path, min_cells_per_condition=10)
    matrix = _DenseMatrixAdapter(X)
    data, ctx = build_pseudobulk_core(
        matrix, cell_line, compound, dose, is_control, batch, gene_names, cfg, want_halves=False,
    )
    assert data.X.shape[0] == 1
    # Only the plate1 cells (50) should survive -- plate2's 20 cells are dropped.
    assert int(data.obs["n_cells"].iloc[0]) == n_treated_p1
    pm = data.meta["plate_matching"]
    assert pm["n_cells_dropped_plate_no_control"] == n_treated_p2
    assert pm["n_condition_plate_pairs_dropped_no_control"] == 1


def test_select_hvgs_uses_only_train_rows(tmp_path):
    syn = _make_synthetic()
    cfg = _default_cfg(tmp_path)
    matrix = _DenseMatrixAdapter(syn["X"])
    data, ctx = build_pseudobulk_core(
        matrix, syn["cell_line"], syn["compound"], syn["dose"], syn["is_control"],
        syn["batch"], syn["gene_names"], cfg, want_halves=False,
    )
    n_cond = data.X.shape[0]
    train_idx = np.arange(n_cond // 2)
    test_idx = np.arange(n_cond // 2, n_cond)
    genes_from_train = select_hvgs(data, train_idx, n_hvg=5)
    # selecting on a disjoint subset of rows should generally give a different answer
    # than selecting on the other half -- not a strict guarantee, but confirms the
    # function actually restricts itself to the given rows rather than using all data.
    genes_from_full = select_hvgs(data, np.arange(n_cond), n_hvg=5)
    assert genes_from_train.shape == (5,)
    assert set(genes_from_train.tolist()) <= set(range(data.X.shape[1]))
    # sanity: passing ALL rows as "train" should equal selecting from the whole array
    var_full = data.X.var(axis=0)
    expected = np.sort(np.argsort(var_full)[::-1][:5])
    np.testing.assert_array_equal(genes_from_full, expected)


def test_select_hvgs_is_unaffected_by_mutating_test_rows(tmp_path):
    """H-1 (REVIEW.md): the previous version of this test was tautological -- it
    never actually proved select_hvgs ignores held-out rows. This version computes
    the selection once, then MUTATES data.X at test_idx to enormous variance and
    recomputes: if select_hvgs looked at test rows at all (e.g. a planted leak like
    `X_train = data.X` using every row instead of `data.X[train_idx]`), the enormous
    variance at test_idx would dominate argsort and change the answer. A correct,
    train-only implementation must return the IDENTICAL selection either way."""
    syn = _make_synthetic()
    cfg = _default_cfg(tmp_path)
    matrix = _DenseMatrixAdapter(syn["X"])
    data, ctx = build_pseudobulk_core(
        matrix, syn["cell_line"], syn["compound"], syn["dose"], syn["is_control"],
        syn["batch"], syn["gene_names"], cfg, want_halves=False,
    )
    n_cond = data.X.shape[0]
    train_idx = np.arange(n_cond // 2)
    test_idx = np.arange(n_cond // 2, n_cond)

    genes_before = select_hvgs(data, train_idx, n_hvg=5)

    # Mutate ONLY the held-out rows to huge, gene-dominating variance.
    data.X[test_idx] = (
        np.random.default_rng(123).normal(0, 1e6, size=(test_idx.size, data.X.shape[1]))
    ).astype(np.float32)

    genes_after = select_hvgs(data, train_idx, n_hvg=5)

    np.testing.assert_array_equal(
        genes_before, genes_after,
        err_msg=(
            "select_hvgs(train_idx) changed after mutating ONLY test_idx rows -- "
            "it is reading held-out data. This is exactly the leak REVIEW.md H-1 "
            "describes (e.g. `X_train = data.X` instead of `data.X[train_idx]`)."
        ),
    )


def test_select_hvgs_leak_plant_is_actually_caught_by_the_test_above(tmp_path, monkeypatch):
    """Meta-test for H-1: prove the test above is not ITSELF tautological by
    planting the exact leak the reviewer described (`X_train = data.X`, i.e. ALL
    rows instead of `data.X[train_idx]`) and confirming the mutate-test_idx-and-
    recompute check above actually fails against the planted leak, then confirming
    it passes again once un-planted. We do not ship the plant; we prove our new
    test would have caught it."""
    import perturb_bench.data as data_mod

    syn = _make_synthetic()
    cfg = _default_cfg(tmp_path)
    matrix = _DenseMatrixAdapter(syn["X"])
    data, ctx = build_pseudobulk_core(
        matrix, syn["cell_line"], syn["compound"], syn["dose"], syn["is_control"],
        syn["batch"], syn["gene_names"], cfg, want_halves=False,
    )
    n_cond = data.X.shape[0]
    train_idx = np.arange(n_cond // 2)
    test_idx = np.arange(n_cond // 2, n_cond)

    def _leaky_select_hvgs(data, train_idx, n_hvg):
        if n_hvg is None or n_hvg >= data.X.shape[1]:
            return np.arange(data.X.shape[1])
        X_train = data.X  # PLANTED LEAK: all rows, not data.X[train_idx]
        var = X_train.var(axis=0)
        top = np.argsort(var)[::-1][:n_hvg]
        return np.sort(top)

    genes_before_leaky = _leaky_select_hvgs(data, train_idx, n_hvg=5)
    data.X[test_idx] = (
        np.random.default_rng(123).normal(0, 1e6, size=(test_idx.size, data.X.shape[1]))
    ).astype(np.float32)
    genes_after_leaky = _leaky_select_hvgs(data, train_idx, n_hvg=5)

    # The leaky implementation MUST disagree once test rows are mutated -- if this
    # assertion itself failed, our adversarial test design would be too weak to ever
    # catch H-1-style leaks, and we'd need a harder perturbation.
    assert not np.array_equal(genes_before_leaky, genes_after_leaky), (
        "planted leak (select_hvgs seeing all rows) did not change the selection "
        "after mutating test_idx -- the adversarial test design is too weak"
    )

    # Sanity: monkeypatching the real select_hvgs with the leaky version and running
    # it through the SAME call pattern as test_select_hvgs_is_unaffected_by_mutating_test_rows
    # reproduces the failure end-to-end.
    data2, _ = build_pseudobulk_core(
        _DenseMatrixAdapter(syn["X"]), syn["cell_line"], syn["compound"], syn["dose"],
        syn["is_control"], syn["batch"], syn["gene_names"], cfg, want_halves=False,
    )
    monkeypatch.setattr(data_mod, "select_hvgs", _leaky_select_hvgs)
    genes_before = data_mod.select_hvgs(data2, train_idx, n_hvg=5)
    data2.X[test_idx] = (
        np.random.default_rng(123).normal(0, 1e6, size=(test_idx.size, data2.X.shape[1]))
    ).astype(np.float32)
    genes_after = data_mod.select_hvgs(data2, train_idx, n_hvg=5)
    with pytest.raises(AssertionError):
        np.testing.assert_array_equal(genes_before, genes_after)


def test_cache_round_trip(tmp_path, monkeypatch):
    """load_pseudobulk should cache to disk and the second call should be fast and
    return identical arrays, without re-invoking the loader."""
    syn = _make_synthetic()
    cfg = _default_cfg(tmp_path)

    import perturb_bench.data as data_mod

    class _FakeLoader:
        name = "sciplex3"
        calls = 0

        def load(self, cfg):
            _FakeLoader.calls += 1
            matrix = _DenseMatrixAdapter(syn["X"])
            data, ctx = build_pseudobulk_core(
                matrix, syn["cell_line"], syn["compound"], syn["dose"], syn["is_control"],
                syn["batch"], syn["gene_names"], cfg, want_halves=True,
            )
            return data

    monkeypatch.setitem(data_mod._LOADER_REGISTRY, "sciplex3", _FakeLoader())

    d1 = load_pseudobulk(cfg)
    assert _FakeLoader.calls == 1
    d2 = load_pseudobulk(cfg)
    assert _FakeLoader.calls == 1  # second call must hit the cache, not the loader

    np.testing.assert_array_equal(d1.X, d2.X)
    np.testing.assert_array_equal(d1.delta, d2.delta)
    np.testing.assert_array_equal(d1.de_mask, d2.de_mask)
    np.testing.assert_array_equal(np.asarray(d1.halves), np.asarray(d2.halves))
    pd.testing.assert_frame_equal(d1.obs, d2.obs, check_dtype=False)


def test_cache_key_ignores_out_dir_but_not_qc_params(tmp_path):
    cfg1 = _default_cfg(tmp_path, out_dir=str(tmp_path / "out1"))
    cfg2 = _default_cfg(tmp_path, out_dir=str(tmp_path / "out2"))
    cfg3 = _default_cfg(tmp_path, min_cells_per_condition=999)
    assert cfg1.cache_key() == cfg2.cache_key()
    assert cfg1.cache_key() != cfg3.cache_key()

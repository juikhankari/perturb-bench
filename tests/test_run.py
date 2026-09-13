"""Tests for perturb_bench.run -- the end-to-end orchestrator. Uses a small
SYNTHETIC PseudobulkData, never the real 12GB sciplex3 cache."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from perturb_bench.config import Config
from perturb_bench.data import PseudobulkData, _DenseMatrixAdapter, build_pseudobulk_core
import perturb_bench.run as run_mod
import perturb_bench.splits as splits_mod
from perturb_bench.ceiling import compute_ceiling


def _make_synthetic(tmp_path, seed=0, n_cell_lines=3, n_compounds=5, n_doses=2,
                     n_genes=30, cells_per_cond=80, n_ceiling_reps=4):
    rng = np.random.default_rng(seed)
    gene_names = np.array([f"G{i}" for i in range(n_genes)])
    cell_lines = [f"CL{i}" for i in range(n_cell_lines)]
    compounds = [f"drug{i}" for i in range(n_compounds)]
    doses = [10.0 * (i + 1) for i in range(n_doses)]
    batches = ["plate0", "plate1"]

    baseline = {cl: rng.normal(5, 1, n_genes) for cl in cell_lines}
    de_genes = rng.choice(n_genes, size=5, replace=False)
    effect = {}
    for c in compounds:
        eff = np.zeros(n_genes)
        eff[de_genes] = rng.normal(2, 0.3, len(de_genes))
        effect[c] = eff

    rows_X, rows_cl, rows_co, rows_dose, rows_batch, rows_isctrl = [], [], [], [], [], []
    for cl in cell_lines:
        for b in batches:
            n_ctrl = 60
            ctrl_cells = baseline[cl] + rng.normal(0, 0.4, (n_ctrl, n_genes))
            rows_X.append(ctrl_cells)
            rows_cl += [cl] * n_ctrl
            rows_co += ["control"] * n_ctrl
            rows_dose += [0.0] * n_ctrl
            rows_batch += [b] * n_ctrl
            rows_isctrl += [True] * n_ctrl
        for co in compounds:
            for d in doses:
                cells = (
                    baseline[cl] + effect[co] * (d / doses[0])
                    + rng.normal(0, 0.4, (cells_per_cond, n_genes))
                )
                b = batches[rng.integers(0, len(batches))]
                rows_X.append(cells)
                rows_cl += [cl] * cells_per_cond
                rows_co += [co] * cells_per_cond
                rows_dose += [d] * cells_per_cond
                rows_batch += [b] * cells_per_cond
                rows_isctrl += [False] * cells_per_cond

    X = np.clip(np.vstack(rows_X), 0, None).astype(np.float32)
    matrix = _DenseMatrixAdapter(X)
    cfg = Config(
        dataset="sciplex3", seed=seed, min_genes=1, max_pct_mito=100.0,
        min_cells_per_condition=10, n_hvg=15, gene_set="hvg",
        n_ceiling_reps=n_ceiling_reps, cache_dir=str(tmp_path / "cache"),
        out_dir=str(tmp_path / "out"), de_alpha=0.05,
    )
    data, ctx = build_pseudobulk_core(
        matrix,
        np.array(rows_cl, dtype=object), np.array(rows_co, dtype=object),
        np.array(rows_dose, dtype=float), np.array(rows_isctrl, dtype=bool),
        np.array(rows_batch, dtype=object), gene_names, cfg, want_halves=True,
    )
    return data, cfg


@pytest.fixture
def synthetic(tmp_path):
    return _make_synthetic(tmp_path)


def _patch_loader(monkeypatch, data):
    monkeypatch.setattr(run_mod, "load_pseudobulk", lambda cfg: data)


# ---------------------------------------------------------------------------
# End-to-end
# ---------------------------------------------------------------------------

def test_end_to_end_run_produces_expected_columns(tmp_path, synthetic, monkeypatch):
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)

    results, meta = run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "out"))

    expected_cols = [
        "split", "fold", "baseline", "metric", "representation", "target",
        "value", "std", "n_conditions", "n_test", "n_skipped",
        "ci_low", "ci_high",
        "normalized", "normalized_std", "normalization_valid",
        "floor_source", "floor_value", "ceiling_value",
        "fallback_rate", "fallback_rate_dose_collapsed",
    ]
    assert list(results.columns) == expected_cols

    # ADDENDUM 3: both scoring targets must appear.
    assert set(results["target"].unique()) == {"replicate_half", "full_pseudobulk"}
    assert len(results) > 0

    # structural columns must never be NaN
    for col in ["split", "fold", "baseline", "metric", "representation", "floor_source"]:
        assert results[col].isna().sum() == 0, f"{col} has NaN values"

    assert (tmp_path / "out" / "results.csv").exists()
    assert (tmp_path / "out" / "report.md").exists()

    report_text = (tmp_path / "out" / "report.md").read_text()
    assert "per_drug_mean_delta" in report_text
    assert "SPEC §10" in report_text


def test_end_to_end_run_covers_all_regimes(tmp_path, synthetic, monkeypatch):
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)
    results, meta = run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "out"))
    assert set(results["split"].unique()) == set(splits_mod.REGIMES)
    # ceiling and floor baselines appear as rows
    assert "ceiling" in results["baseline"].unique()
    assert "no_change" in results["baseline"].unique()
    assert "global_mean_delta" in results["baseline"].unique()


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_determinism_same_seed_byte_identical_csv(tmp_path, synthetic, monkeypatch):
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)

    run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "out1"))
    run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "out2"))

    bytes1 = (tmp_path / "out1" / "results.csv").read_bytes()
    bytes2 = (tmp_path / "out2" / "results.csv").read_bytes()
    assert bytes1 == bytes2


# ---------------------------------------------------------------------------
# Leakage assertion actually invoked
# ---------------------------------------------------------------------------

def test_assert_no_leakage_called_once_per_fold(tmp_path, synthetic, monkeypatch):
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)

    call_count = {"n": 0}
    real_assert = splits_mod.assert_no_leakage

    def spy(obs, fold):
        call_count["n"] += 1
        return real_assert(obs, fold)

    monkeypatch.setattr(run_mod, "assert_no_leakage", spy)

    results, meta = run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "out"))

    assert call_count["n"] == meta["n_folds"]
    assert call_count["n"] > 0
    # every (split, fold) pair that appears in results.csv had a leakage check
    n_distinct_folds = results[["split", "fold"]].drop_duplicates().shape[0]
    assert n_distinct_folds == call_count["n"]


def test_leaky_fold_fails_the_run_rather_than_silently_scoring(tmp_path, synthetic, monkeypatch):
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)

    def leaky_make_folds(obs, regime, seed):
        folds = splits_mod.make_folds(obs, regime, seed)
        if regime == "held_out_context" and folds:
            # Sabotage: put a held-out-cell-line row back into train_idx.
            bad = folds[0]
            leaked_train = np.concatenate([bad.train_idx, bad.test_idx[:1]])
            bad.train_idx = leaked_train
        return folds

    monkeypatch.setattr(run_mod, "make_folds", leaky_make_folds)

    with pytest.raises(AssertionError):
        run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "out"))


# ---------------------------------------------------------------------------
# Normalized column sanity: ~0 at floor, ~1 at ceiling
# ---------------------------------------------------------------------------

def test_normalized_is_zero_at_floor_and_one_at_ceiling(tmp_path, synthetic, monkeypatch):
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)
    results, meta = run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "out"))

    # ADDENDUM 3 Decision 1: the PRIMARY target is replicate_half -- baseline
    # and ceiling face identical target noise there, so (unlike
    # full_pseudobulk, which can legitimately hit the direction guard on real
    # data per REVIEW.md C-1) floor->0 / ceiling->1 should hold cleanly.
    delta = results[(results["representation"] == "delta") & (results["target"] == "replicate_half")]

    # mae's floor is no_change -- no_change's own normalized mae score must be ~0.
    nc_mae = delta[(delta["baseline"] == "no_change") & (delta["metric"] == "mae")]
    assert len(nc_mae) > 0
    assert nc_mae["normalized"].apply(lambda v: np.isclose(v, 0.0, atol=1e-6) or pd.isna(v)).all()
    assert nc_mae["floor_source"].eq("no_change").all()
    assert nc_mae["normalization_valid"].all()

    # ceiling's own normalized score must be ~1 for every metric it's defined for.
    ceil_rows = delta[delta["baseline"] == "ceiling"]
    assert len(ceil_rows) > 0
    finite = ceil_rows["normalized"].dropna()
    assert len(finite) > 0
    assert finite.apply(lambda v: np.isclose(v, 1.0, atol=1e-6)).all()

    # pearson_r's floor is global_mean_delta per ADDENDUM 1 -- its floor_source
    # must say so, and no_change's pearson_r on delta must be NaN (never coerced to 0).
    pr = delta[delta["metric"] == "pearson_r"]
    assert pr["floor_source"].eq("global_mean_delta").all()
    nc_pr = pr[pr["baseline"] == "no_change"]
    assert nc_pr["value"].isna().all()
    assert nc_pr["normalized"].isna().all()

    # ADDENDUM 3 Decision 4: direction_acc_de's floor is now global_mean_delta,
    # not no_change (H-5: no_change's 0.0 there is a sign(0) artifact).
    dad = delta[delta["metric"] == "direction_acc_de"]
    assert dad["floor_source"].eq("global_mean_delta").all()


def test_normalization_valid_gates_normalized_everywhere(tmp_path, synthetic, monkeypatch):
    """Global invariant (ADDENDUM 3 Decision 3): wherever normalization_valid
    is False, normalized MUST be NaN, on every row of the results, for both
    targets -- a guard-suppressed row must never leak a number."""
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)
    results, meta = run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "out"))

    invalid = results[results["normalization_valid"] == False]  # noqa: E712
    if len(invalid):
        assert invalid["normalized"].isna().all()
    valid_nonnan = results[(results["normalization_valid"]) & results["value"].notna()]
    # where valid and the underlying floor/ceiling both exist, normalized is
    # either a finite number or NaN only for the genuine degenerate
    # (ceiling==floor) case -- never silently coerced.
    assert valid_nonnan["normalization_valid"].all()


# ---------------------------------------------------------------------------
# The ceiling gene-union axis bug (ADDENDUM 2): halves lives on a different,
# smaller gene axis than control/de_mask. A fold's gene selection (full-axis
# positions) must be translated into LOCAL positions within the halves axis
# before indexing `halves`, while control/de_mask must be sliced down to that
# same local axis before being handed to compute_ceiling (whose contract
# applies one `genes` array to all three arrays it's given). This test uses a
# gene union that is a strict, non-contiguous, "non-identity" subset of the
# full axis so an off-by-identity mapping bug cannot accidentally pass.
# ---------------------------------------------------------------------------

def test_ceiling_for_fold_maps_full_axis_genes_to_local_halves_positions():
    n_cond, n_genes_full = 4, 12
    halves_gene_idx = [1, 4, 7, 9, 11]  # sorted, sparse, non-identity subset
    n_genes_halves = len(halves_gene_idx)

    rng = np.random.default_rng(7)
    control = rng.normal(size=(n_cond, n_genes_full)).astype(np.float32)
    de_mask = np.zeros((n_cond, n_genes_full), dtype=bool)
    de_mask[:, halves_gene_idx] = True

    n_rep = 3
    half0 = rng.normal(size=(n_rep, n_cond, n_genes_halves)).astype(np.float32)
    half1 = half0 + 0.1 * rng.normal(size=half0.shape).astype(np.float32)
    halves = np.stack([half0, half1], axis=1)  # (n_rep, 2, n_cond, n_genes_halves)

    data = PseudobulkData(
        X=np.zeros((n_cond, n_genes_full), dtype=np.float32),
        control=control,
        delta=np.zeros((n_cond, n_genes_full), dtype=np.float32),
        de_mask=de_mask,
        halves=halves,
        obs=pd.DataFrame(
            {"cell_line": ["A"] * n_cond, "compound": ["d"] * n_cond, "dose": [1.0] * n_cond},
            index=[f"cond{i}" for i in range(n_cond)],
        ),
        var=np.array([f"G{i}" for i in range(n_genes_full)]),
        meta={"halves_gene_idx": halves_gene_idx},
    )

    # Fold selected a subset of the union at full-axis positions [4, 9, 11].
    # Their CORRECT local positions within halves_gene_idx=[1,4,7,9,11] are [1,3,4]
    # -- deliberately not equal to the full-axis values, so an identity-mapping
    # bug (using [4,9,11] directly as column indices into the 5-wide halves axis)
    # would either go out of bounds or pick the wrong columns.
    fold_genes = np.array([4, 9, 11])
    local_expected = np.array([1, 3, 4])

    all_rows = np.arange(n_cond)
    result = run_mod._ceiling_for_fold(data, all_rows, fold_genes, "delta")

    expected = compute_ceiling(
        data.halves,
        control[:, halves_gene_idx],
        de_mask[:, halves_gene_idx],
        local_expected,
        "delta",
    )

    pd.testing.assert_frame_equal(result, expected)

    # REVIEW.md C-2: the ceiling must be scored over the FOLD'S TEST ROWS only,
    # not every condition in the dataset. A strict subset must give the answer
    # you get by slicing the halves/control/de_mask rows yourself, and must
    # differ from the all-rows answer (so a regression to all-rows is caught).
    test_rows = np.array([0, 2])
    result_sub = run_mod._ceiling_for_fold(data, test_rows, fold_genes, "delta")
    expected_sub = compute_ceiling(
        data.halves[:, :, test_rows, :],
        control[np.ix_(test_rows, halves_gene_idx)],
        de_mask[np.ix_(test_rows, halves_gene_idx)],
        local_expected,
        "delta",
    )
    pd.testing.assert_frame_equal(result_sub, expected_sub)
    assert result_sub.loc["pearson_r", "n"] == len(test_rows) * n_rep
    assert result.loc["pearson_r", "n"] == n_cond * n_rep


def test_ceiling_for_fold_raises_loudly_on_genes_outside_the_union():
    n_cond, n_genes_full = 3, 8
    halves_gene_idx = [0, 2, 5]
    halves = np.zeros((2, 2, n_cond, len(halves_gene_idx)), dtype=np.float32)
    control = np.zeros((n_cond, n_genes_full), dtype=np.float32)
    de_mask = np.zeros((n_cond, n_genes_full), dtype=bool)

    data = PseudobulkData(
        X=np.zeros((n_cond, n_genes_full), dtype=np.float32),
        control=control, delta=np.zeros((n_cond, n_genes_full), dtype=np.float32),
        de_mask=de_mask, halves=halves,
        obs=pd.DataFrame({"cell_line": ["A"] * n_cond, "compound": ["d"] * n_cond, "dose": [1.0] * n_cond}),
        var=np.array([f"G{i}" for i in range(n_genes_full)]),
        meta={"halves_gene_idx": halves_gene_idx},
    )

    # gene 3 is NOT in the union -- must raise rather than silently drop/mis-map it.
    fold_genes = np.array([0, 3])
    with pytest.raises(AssertionError):
        run_mod._ceiling_for_fold(data, np.arange(n_cond), fold_genes, "delta")


# ---------------------------------------------------------------------------
# ADDENDUM 3 Decision 3 -- the direction guard (C-1 part 1). This is the
# permanent backstop: unit-test it directly with a DELIBERATELY INVERTED
# floor/ceiling pair, independent of any real data path.
# ---------------------------------------------------------------------------

def test_direction_guard_fires_on_inverted_pair_and_logs_loudly(capsys):
    # mae is lower_better: ceiling should be BELOW floor. Feed it the inverted
    # case (ceiling above floor) -- exactly the bug measured in REVIEW.md C-1
    # (split-half mae ceiling 0.0247-0.0394 vs no_change floor 0.0230-0.0325).
    normalized, normalized_std, valid = run_mod._normalize_with_guard(
        value=0.05, std=0.01, floor_mean=0.03, ceiling_mean=0.04,
        metric="mae", fold_name="fake_fold", representation="delta", target="full_pseudobulk",
    )
    assert valid is False
    assert np.isnan(normalized)
    assert np.isnan(normalized_std)
    err = capsys.readouterr().err
    assert "DIRECTION GUARD FIRED" in err
    assert "fake_fold" in err
    assert "mae" in err
    assert "0.03" in err or "0.03000" in err  # floor_mean named
    assert "0.04" in err or "0.04000" in err  # ceiling_mean named


def test_direction_guard_passes_on_correctly_ordered_pair():
    # Same metric, correctly ordered (ceiling below floor) -- must NOT fire,
    # and must return the ordinary normalize_score() value.
    normalized, normalized_std, valid = run_mod._normalize_with_guard(
        value=0.035, std=0.01, floor_mean=0.05, ceiling_mean=0.01,
        metric="mae", fold_name="fake_fold", representation="delta", target="full_pseudobulk",
    )
    assert valid is True
    assert not np.isnan(normalized)
    assert 0.0 <= normalized <= 1.0


def test_direction_guard_higher_better_metric_inverted():
    # pearson_r is higher_better: ceiling should be ABOVE floor. Feed the
    # inverted case.
    normalized, normalized_std, valid = run_mod._normalize_with_guard(
        value=0.2, std=0.05, floor_mean=0.5, ceiling_mean=0.3,
        metric="pearson_r", fold_name="fake_fold", representation="delta", target="replicate_half",
    )
    assert valid is False
    assert np.isnan(normalized)


def test_direction_guard_degenerate_equal_floor_ceiling_is_not_a_violation():
    # floor == ceiling is degenerate (no dynamic range), not a direction
    # contradiction -- must return NaN/valid=True, not NaN/valid=False.
    normalized, normalized_std, valid = run_mod._normalize_with_guard(
        value=0.5, std=0.0, floor_mean=0.3, ceiling_mean=0.3,
        metric="pearson_r", fold_name="fake_fold", representation="delta", target="replicate_half",
    )
    assert valid is True
    assert np.isnan(normalized)


def test_direction_guard_fires_end_to_end_via_run_when_ceiling_is_sabotaged(tmp_path, synthetic, monkeypatch):
    """L-2 replacement (REVIEW.md): the old test asserted the ceiling's own
    normalized score is tautologically 1.0, which cannot fail even if the
    ceiling computation is broken. This plants a deliberately WRONG ceiling
    (for mae, lower_better) that is numerically worse than the no_change
    floor -- the exact shape of bug C-1 -- and asserts (a) normalized is NaN
    everywhere for that (fold, metric) combination, (b) normalization_valid
    is False there, and (c) a normal (non-sabotaged) run on the same data
    does NOT have normalization_valid=False for that combination, so this
    test would actually fail if the guard were removed or broken.
    """
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)

    baseline_results, _ = run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "baseline_out"))
    baseline_delta = baseline_results[
        (baseline_results["representation"] == "delta")
        & (baseline_results["metric"] == "mae")
        & (baseline_results["target"] == "replicate_half")
    ]
    assert baseline_delta["normalization_valid"].all(), "sanity: unsabotaged run should not trip the guard on mae"

    real_ceiling_raw_for_fold = run_mod._ceiling_raw_for_fold

    def sabotaged(half0, half1, control_sub, de_mask_sub, representation, n_rep):
        raw = real_ceiling_raw_for_fold(half0, half1, control_sub, de_mask_sub, representation, n_rep)
        # Make the ceiling's mae WORSE than the floor would be (floor mae is
        # always >= 0; pushing the ceiling's mae up past any plausible floor
        # reproduces the inverted-sign scenario from REVIEW.md C-1) -- done
        # at the RAW per-row level so both the aggregate mean AND the
        # bootstrap CI the guard/report see are consistently sabotaged.
        raw = raw.copy()
        raw["mae"] = 10_000.0
        return raw

    monkeypatch.setattr(run_mod, "_ceiling_raw_for_fold", sabotaged)
    sabotaged_results, _ = run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "sabotaged_out"))

    sab_delta = sabotaged_results[
        (sabotaged_results["representation"] == "delta")
        & (sabotaged_results["metric"] == "mae")
    ]
    assert len(sab_delta) > 0
    assert not sab_delta["normalization_valid"].any(), "guard should have fired everywhere mae is scored"
    assert sab_delta["normalized"].isna().all(), "normalized must be NaN once the guard fires -- this is the L-2 replacement assertion"

    # And the baseline's normalized value genuinely MOVED (from a real number
    # to NaN) as a direct result of the planted wrong ceiling.
    base_ceiling_mae = baseline_results[
        (baseline_results["baseline"] == "ceiling") & (baseline_results["metric"] == "mae")
        & (baseline_results["target"] == "replicate_half") & (baseline_results["representation"] == "delta")
    ]["normalized"]
    assert base_ceiling_mae.notna().any()


# ---------------------------------------------------------------------------
# M-1: n_test / n_skipped are written per row, independent of n_conditions
# (which, for a ceiling/replicate_half row, is n_rep * n_test rather than a
# condition count -- REVIEW.md M-1).
# ---------------------------------------------------------------------------

def test_n_test_and_n_skipped_written_and_distinguishable_from_n_conditions(tmp_path, synthetic, monkeypatch):
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)
    results, meta = run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "out"))

    assert "n_test" in results.columns
    assert "n_skipped" in results.columns
    assert (results["n_test"] > 0).all()

    # ceiling rows on replicate_half: n_conditions should be (approximately,
    # modulo DE-skips) n_rep * n_test, strictly more than n_test itself --
    # i.e. n_conditions does NOT read as a plain condition count here, which
    # is exactly the M-1 confusion being fixed. pearson_r has no DE-skips.
    n_rep = meta["n_ceiling_reps"]
    ceil_pr = results[
        (results["baseline"] == "ceiling") & (results["metric"] == "pearson_r")
        & (results["target"] == "replicate_half") & (results["representation"] == "delta")
    ]
    assert len(ceil_pr) > 0
    for _, row in ceil_pr.iterrows():
        assert row["n_conditions"] == pytest.approx(row["n_test"] * n_rep)

    # full_pseudobulk rows: n_conditions == n_test for a metric with no skips.
    # no_change is excluded: its pearson_r is STRUCTURALLY NaN (zero-variance
    # prediction, ADDENDUM 1), which is a different kind of "undefined" than
    # a DE-skip and legitimately makes n_conditions (the non-NaN count) 0.
    fp_pr = results[
        (~results["baseline"].isin(["ceiling", "no_change"])) & (results["metric"] == "pearson_r")
        & (results["target"] == "full_pseudobulk") & (results["representation"] == "delta")
    ]
    assert len(fp_pr) > 0
    for _, row in fp_pr.iterrows():
        assert row["n_conditions"] == row["n_test"]


# ---------------------------------------------------------------------------
# M-3: per_drug_mean_delta's tier-1 (dose-collapsed) fallback is counted
# separately from the tier-2 (fallback_rate) column.
# ---------------------------------------------------------------------------

def test_dose_collapse_fallback_rate_is_recorded_for_per_drug_mean_delta(tmp_path, synthetic, monkeypatch):
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)
    results, meta = run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "out"))

    pdmd = results[results["baseline"] == "per_drug_mean_delta"]
    assert "fallback_rate_dose_collapsed" in pdmd.columns
    # random/held_out_context folds see every compound in training (same
    # dose grid), so dose-collapse should be near 0 there; the column must
    # at least be populated (not universally NaN) somewhere in the run.
    assert pdmd["fallback_rate_dose_collapsed"].notna().any()


# ---------------------------------------------------------------------------
# H-6: incremental write + --resume.
# ---------------------------------------------------------------------------

def test_resume_skips_already_completed_folds(tmp_path, synthetic, monkeypatch):
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)
    out_dir = tmp_path / "out"

    full_results, full_meta = run_mod.run(cfg, quick=True, out_dir=str(out_dir))

    # Simulate a crash: truncate results.csv to only the first fold's rows
    # (the header plus exactly the rows for one (split, fold) pair).
    first_key = full_results[["split", "fold"]].drop_duplicates().iloc[0]
    partial = full_results[
        (full_results["split"] == first_key["split"]) & (full_results["fold"] == first_key["fold"])
    ]
    partial.to_csv(out_dir / "results.csv", index=False)

    call_log: list[tuple[str, str]] = []
    real_assert = splits_mod.assert_no_leakage

    def spy(obs, fold):
        call_log.append((fold.regime, fold.name))
        return real_assert(obs, fold)

    monkeypatch.setattr(run_mod, "assert_no_leakage", spy)

    resumed_results, resumed_meta = run_mod.run(cfg, quick=True, out_dir=str(out_dir), resume=True)

    # The already-completed fold must NOT have been recomputed.
    assert (first_key["split"], first_key["fold"]) not in call_log

    # The final, resumed results must match a from-scratch run exactly
    # (same rows, just produced via a different code path).
    pd.testing.assert_frame_equal(
        resumed_results.reset_index(drop=True), full_results.reset_index(drop=True),
    )


def test_resume_with_no_existing_file_behaves_like_a_fresh_run(tmp_path, synthetic, monkeypatch):
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)
    out_dir = tmp_path / "out"
    results, meta = run_mod.run(cfg, quick=True, out_dir=str(out_dir), resume=True)
    assert len(results) > 0
    assert (out_dir / "results.csv").exists()


# ---------------------------------------------------------------------------
# Determinism must survive the bootstrap CI columns and the incremental
# write path (both are new sources of potential nondeterminism).
# ---------------------------------------------------------------------------

def test_determinism_same_seed_byte_identical_csv_with_new_columns(tmp_path, synthetic, monkeypatch):
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)

    run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "out1"))
    run_mod.run(cfg, quick=True, out_dir=str(tmp_path / "out2"))

    bytes1 = (tmp_path / "out1" / "results.csv").read_bytes()
    bytes2 = (tmp_path / "out2" / "results.csv").read_bytes()
    assert bytes1 == bytes2


# ---------------------------------------------------------------------------
# run_meta.json is written BEFORE the sweep starts (H-6).
# ---------------------------------------------------------------------------

def test_run_meta_written_before_sweep_and_contains_provenance(tmp_path, synthetic, monkeypatch):
    data, cfg = synthetic
    _patch_loader(monkeypatch, data)
    out_dir = tmp_path / "out"

    seen_meta_early = {}

    real_make_folds = splits_mod.make_folds

    def spy_make_folds(obs, regime, seed):
        if not seen_meta_early and (out_dir / "run_meta.json").exists():
            with open(out_dir / "run_meta.json") as f:
                seen_meta_early.update(json.load(f))
        return real_make_folds(obs, regime, seed)

    monkeypatch.setattr(run_mod, "make_folds", spy_make_folds)
    run_mod.run(cfg, quick=True, out_dir=str(out_dir))

    assert seen_meta_early, "run_meta.json must exist before the first fold is processed"
    assert "versions" in seen_meta_early
    assert "git" in seen_meta_early
    assert seen_meta_early.get("seed") == cfg.seed

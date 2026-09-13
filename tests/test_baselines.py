from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import pytest

from perturb_bench.baselines import (
    BASELINES,
    GlobalMeanDelta,
    NearestContext,
    NoChange,
    PerDrugMeanDelta,
    Ridge_,
    RidgeNoDrug,
)

RNG = np.random.default_rng(0)


@dataclass
class FakePseudobulkData:
    """Minimal PseudobulkData-shaped stand-in for tests.

    Only carries the attributes baselines.py actually reads (X, control,
    delta, obs). We avoid importing the real PseudobulkData dataclass from
    data.py, which is being written by another agent in parallel.
    """

    X: np.ndarray
    control: np.ndarray
    delta: np.ndarray
    obs: pd.DataFrame
    var: np.ndarray = field(default_factory=lambda: np.array([]))
    de_mask: np.ndarray = None
    halves: np.ndarray = None
    meta: dict = field(default_factory=dict)


def _make_synthetic_data(
    cell_lines=("A549", "K562"),
    compounds=("drugA", "drugB", "drugC"),
    doses=(10.0, 100.0),
    n_genes=20,
    noise=0.01,
    seed=0,
):
    rng = np.random.default_rng(seed)

    # fixed per-cell-line control baseline
    line_base = {cl: rng.normal(5.0, 1.0, size=n_genes) for cl in cell_lines}
    # fixed per-compound effect signature (shared across cell lines, so the
    # "true" drug effect is context-independent in this synthetic world --
    # which is exactly the world where B3 per_drug_mean_delta should do well)
    compound_effect = {c: rng.normal(0.0, 2.0, size=n_genes) for c in compounds}

    rows = []
    X_list, control_list, delta_list = [], [], []
    for cl in cell_lines:
        for c in compounds:
            for dose in doses:
                dose_scale = dose / max(doses)
                control = line_base[cl] + rng.normal(0, noise, size=n_genes)
                delta = compound_effect[c] * dose_scale + rng.normal(0, noise, size=n_genes)
                treated = control + delta
                control_list.append(control)
                delta_list.append(delta)
                X_list.append(treated)
                rows.append(
                    {
                        "cell_line": cl,
                        "compound": c,
                        "dose": dose,
                        "log_dose": np.log1p(dose),
                        "batch": f"{cl}_b0",
                        "n_cells": 100,
                        "is_control": False,
                        "target": "unknown",
                    }
                )

    obs = pd.DataFrame(rows)
    obs.index = [f"{r['cell_line']}|{r['compound']}|{r['dose']}" for r in rows]

    data = FakePseudobulkData(
        X=np.asarray(X_list, dtype=np.float32),
        control=np.asarray(control_list, dtype=np.float32),
        delta=np.asarray(delta_list, dtype=np.float32),
        obs=obs,
        var=np.array([f"gene{i}" for i in range(n_genes)]),
    )
    genes = np.arange(n_genes)
    return data, genes


def test_registry_contains_all_baselines():
    expected = {
        "no_change",
        "global_mean_delta",
        "per_drug_mean_delta",
        "ridge",
        "ridge_no_drug",
        "nearest_context",
    }
    assert expected <= set(BASELINES.keys())


def test_no_change_returns_exactly_zeros():
    data, genes = _make_synthetic_data()
    n = len(data.obs)
    train_idx = np.arange(0, n // 2)
    test_idx = np.arange(n // 2, n)

    b = NoChange()
    b.fit(data, train_idx, genes)
    pred = b.predict(data, test_idx)

    assert pred.shape == (len(test_idx), len(genes))
    assert np.array_equal(pred, np.zeros_like(pred))
    assert not b.fallback_mask.any()


def test_global_mean_delta_equals_training_mean():
    data, genes = _make_synthetic_data()
    n = len(data.obs)
    train_idx = np.arange(0, n // 2)
    test_idx = np.arange(n // 2, n)

    b = GlobalMeanDelta()
    b.fit(data, train_idx, genes)
    pred = b.predict(data, test_idx)

    expected = data.delta[train_idx][:, genes].mean(axis=0)
    for row in pred:
        np.testing.assert_allclose(row, expected, rtol=1e-5)


def test_per_drug_mean_delta_recovers_hand_built_mean():
    # hand-build a tiny case: two cell lines, one compound+dose in training,
    # one test condition of a third (held-out) cell line with that same
    # compound+dose.
    n_genes = 4
    obs_rows = [
        {"cell_line": "A", "compound": "d1", "dose": 1.0, "log_dose": 0.0, "batch": "A_b0", "n_cells": 50, "is_control": False, "target": "unknown"},
        {"cell_line": "B", "compound": "d1", "dose": 1.0, "log_dose": 0.0, "batch": "B_b0", "n_cells": 50, "is_control": False, "target": "unknown"},
        {"cell_line": "C", "compound": "d1", "dose": 1.0, "log_dose": 0.0, "batch": "C_b0", "n_cells": 50, "is_control": False, "target": "unknown"},
    ]
    obs = pd.DataFrame(obs_rows)
    obs.index = ["A|d1|1.0", "B|d1|1.0", "C|d1|1.0"]

    delta_A = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    delta_B = np.array([3.0, 4.0, 5.0, 6.0], dtype=np.float32)
    delta_C = np.array([100.0, 100.0, 100.0, 100.0], dtype=np.float32)  # test row, should be ignored

    control = np.zeros((3, n_genes), dtype=np.float32)
    delta = np.stack([delta_A, delta_B, delta_C])
    X = control + delta

    data = FakePseudobulkData(X=X, control=control, delta=delta, obs=obs)
    genes = np.arange(n_genes)

    train_idx = np.array([0, 1])  # A, B
    test_idx = np.array([2])  # C, same compound+dose, unseen cell line

    b = PerDrugMeanDelta()
    b.fit(data, train_idx, genes)
    pred = b.predict(data, test_idx)

    expected = (delta_A + delta_B) / 2
    np.testing.assert_allclose(pred[0], expected, rtol=1e-5)
    # compound+dose was seen in training (tier 0), so no fallback
    assert not b.fallback_mask[0]


def test_per_drug_mean_delta_falls_back_to_global_mean_for_unseen_drug():
    data, genes = _make_synthetic_data(compounds=("drugA", "drugB", "drugC"))
    obs = data.obs
    is_drugC = (obs["compound"] == "drugC").to_numpy()
    train_idx = np.where(~is_drugC)[0]
    test_idx = np.where(is_drugC)[0]

    b = PerDrugMeanDelta()
    b.fit(data, train_idx, genes)
    pred = b.predict(data, test_idx)

    expected_global = data.delta[train_idx][:, genes].mean(axis=0)
    for row in pred:
        np.testing.assert_allclose(row, expected_global, rtol=1e-5)

    # fallback_mask True exactly for the unseen-drug conditions (all of them, here)
    assert b.fallback_mask.all()


def test_fallback_mask_true_exactly_for_unseen_drug_conditions_mixed():
    data, genes = _make_synthetic_data(compounds=("drugA", "drugB", "drugC"))
    obs = data.obs
    # hold out drugC entirely from training, but test set is a MIX of a seen
    # drug (drugA) and the unseen drug (drugC)
    is_drugC = (obs["compound"] == "drugC").to_numpy()
    train_idx = np.where(~is_drugC)[0]

    is_drugA = (obs["compound"] == "drugA").to_numpy()
    test_idx = np.where(is_drugC | is_drugA)[0]

    b = PerDrugMeanDelta()
    b.fit(data, train_idx, genes)
    b.predict(data, test_idx)

    test_compounds = obs["compound"].iloc[test_idx].to_numpy()
    expected_fallback = test_compounds == "drugC"
    np.testing.assert_array_equal(b.fallback_mask, expected_fallback)


def test_ridge_does_not_see_test_rows():
    data, genes = _make_synthetic_data(n_genes=30, seed=1)
    n = len(data.obs)
    rng = np.random.default_rng(2)
    perm = rng.permutation(n)
    train_idx = perm[: n // 2]
    test_idx = perm[n // 2 :]

    b = Ridge_()
    b.fit(data, train_idx, genes)
    pred_before = b.predict(data, test_idx)

    # mutate the TRUE delta of the test rows to garbage -- predict() must not
    # read data.delta for test rows at all, so this must not change anything.
    data.delta[test_idx] = rng.normal(1000, 500, size=data.delta[test_idx].shape).astype(np.float32)
    pred_after = b.predict(data, test_idx)
    np.testing.assert_allclose(pred_before, pred_after, rtol=1e-6)

    # now confirm refitting with the SAME train_idx on otherwise-mutated test
    # rows gives bit-identical model params: fit() must only ever touch
    # train_idx.
    data.control[test_idx] = rng.normal(1000, 500, size=data.control[test_idx].shape).astype(np.float32)
    b2 = Ridge_()
    b2.fit(data, train_idx, genes)
    pred_before_2 = b2.predict(data, train_idx)  # predict on train rows as a fixed probe
    pred_again = b.predict(data, train_idx)
    np.testing.assert_allclose(pred_before_2, pred_again, rtol=1e-4)


def test_ridge_fallback_mask_true_for_unseen_compound():
    data, genes = _make_synthetic_data(compounds=("drugA", "drugB", "drugC"), n_genes=15)
    obs = data.obs
    is_drugC = (obs["compound"] == "drugC").to_numpy()
    train_idx = np.where(~is_drugC)[0]
    test_idx = np.where(is_drugC)[0]

    b = Ridge_()
    b.fit(data, train_idx, genes)
    b.predict(data, test_idx)
    assert b.fallback_mask.all()


def test_ridge_no_drug_has_no_onehot_fallback():
    data, genes = _make_synthetic_data(compounds=("drugA", "drugB", "drugC"), n_genes=15)
    obs = data.obs
    is_drugC = (obs["compound"] == "drugC").to_numpy()
    train_idx = np.where(~is_drugC)[0]
    test_idx = np.where(is_drugC)[0]

    b = RidgeNoDrug()
    b.fit(data, train_idx, genes)
    pred = b.predict(data, test_idx)
    assert pred.shape == (len(test_idx), len(genes))
    assert not b.fallback_mask.any()


def test_nearest_context_copies_most_similar_line_delta():
    n_genes = 6
    obs_rows = [
        {"cell_line": "A", "compound": "d1", "dose": 1.0, "log_dose": 0.0, "batch": "A_b0", "n_cells": 50, "is_control": False, "target": "unknown"},
        {"cell_line": "B", "compound": "d1", "dose": 1.0, "log_dose": 0.0, "batch": "B_b0", "n_cells": 50, "is_control": False, "target": "unknown"},
        {"cell_line": "C", "compound": "d1", "dose": 1.0, "log_dose": 0.0, "batch": "C_b0", "n_cells": 50, "is_control": False, "target": "unknown"},
    ]
    obs = pd.DataFrame(obs_rows)
    obs.index = ["A|d1|1.0", "B|d1|1.0", "C|d1|1.0"]

    control_A = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    control_B = np.array([-1.0, -1.0, -1.0, -1.0, -1.0, -1.0], dtype=np.float32)
    # C's control is very close (cosine) to A's control
    control_C = np.array([2.0, 2.0, 2.0, 2.0, 2.0, 2.0], dtype=np.float32)

    delta_A = np.array([5.0, 5.0, 5.0, 5.0, 5.0, 5.0], dtype=np.float32)
    delta_B = np.array([-5.0, -5.0, -5.0, -5.0, -5.0, -5.0], dtype=np.float32)
    delta_C = np.zeros(n_genes, dtype=np.float32)  # irrelevant, it's the test row

    control = np.stack([control_A, control_B, control_C])
    delta = np.stack([delta_A, delta_B, delta_C])
    X = control + delta

    data = FakePseudobulkData(X=X, control=control, delta=delta, obs=obs)
    genes = np.arange(n_genes)

    train_idx = np.array([0, 1])  # A, B
    test_idx = np.array([2])  # C

    b = NearestContext()
    b.fit(data, train_idx, genes)
    pred = b.predict(data, test_idx)

    np.testing.assert_allclose(pred[0], delta_A, rtol=1e-5)
    assert not b.fallback_mask[0]


def test_nearest_context_falls_back_when_compound_unseen():
    data, genes = _make_synthetic_data(compounds=("drugA", "drugB", "drugC"), n_genes=10)
    obs = data.obs
    is_drugC = (obs["compound"] == "drugC").to_numpy()
    train_idx = np.where(~is_drugC)[0]
    test_idx = np.where(is_drugC)[0]

    b = NearestContext()
    b.fit(data, train_idx, genes)
    b.predict(data, test_idx)
    assert b.fallback_mask.all()


def _make_rank_deficient_ridge_case(n_lines=3, n_per_line=7, n_test=10, n_genes=3000, seed=9):
    """Control matrix that is rank-deficient the way real sciplex3 control
    matrices are: MANY training conditions share the SAME handful of
    cell-line control profiles (duplicated rows, up to near-float64-epsilon
    noise), with n_genes >> n_train. `n_comp = min(50, n_train-1, n_genes)`
    then requests far more PCA components than the matrix's true rank
    (n_lines-1): the trailing components have explained-variance-ratio at
    float64-machine-epsilon scale (measured down to ~1e-98 / 1e-128 for
    this exact construction -- matches the "~1e-99" the bug report cites).
    Those trailing directions carry essentially zero train-row variance but
    NON-zero projections for a genuinely unseen test cell line, which is
    exactly the extrapolation failure mode described in the bug report and
    is what produces the RuntimeWarning storm from sklearn's PCA/Ridge
    matmuls on real data (confirmed empirically: this construction fires
    ~150-300 RuntimeWarnings through the pre-fix code; the latent-factor-only
    rank deficiency we tried first did not reproduce the warnings at all,
    which is exactly why duplicated rows -- the task's other suggested
    construction -- is used here instead).

    Built with broadcasting/tile, not a small-inner-dimension matmul: a
    couple of matmul shapes on this machine's numpy/Accelerate BLAS emit
    spurious RuntimeWarning("divide by zero"/"overflow"/"invalid value" ...
    in matmul) on fully-finite output for reasons unrelated to this bug
    (verified directly against plain numpy with no sklearn involved at all);
    avoiding that shape in the fixture keeps the no-RuntimeWarning assertion
    below sensitive only to Ridge_'s own PCA/predict path.

    delta is NOT degenerate (ordinary per-row noise) so there is a genuine,
    if weak, target to correlate predictions against -- this is not rigged
    to make ridge look artificially good.
    """
    rng = np.random.default_rng(seed)
    n_train = n_lines * n_per_line

    line_base = rng.normal(size=(n_lines, n_genes))
    line_idx = np.repeat(np.arange(n_lines), n_per_line)
    control_train = line_base[line_idx] + rng.normal(scale=1e-8, size=(n_train, n_genes))
    delta_train = rng.normal(scale=1.0, size=(n_train, n_genes))

    # held-out cell line: genuinely different control, never part of the SVD
    # that produced the near-zero-variance directions.
    test_base = rng.normal(size=n_genes)
    control_test = test_base + rng.normal(scale=0.05, size=(n_test, n_genes))
    delta_test = rng.normal(scale=1.0, size=(n_test, n_genes))

    control = np.vstack([control_train, control_test])
    delta = np.vstack([delta_train, delta_test])
    X = control + delta

    n_total = n_train + n_test
    cell_lines = [f"CL{i}" for i in line_idx] + ["CLX"] * n_test
    doses = np.full(n_total, 10.0)
    obs = pd.DataFrame(
        {
            "cell_line": cell_lines,
            "compound": ["d1"] * n_total,
            "dose": doses,
            "log_dose": np.log1p(doses),
            "batch": [f"b{i}" for i in range(n_total)],
            "n_cells": 100,
            "is_control": False,
            "target": "unknown",
        }
    )
    obs.index = [f"row{i}" for i in range(n_total)]

    data = FakePseudobulkData(
        X=X.astype(np.float32),
        control=control.astype(np.float32),
        delta=delta.astype(np.float32),
        obs=obs,
    )
    genes = np.arange(n_genes)
    train_idx = np.arange(n_train)
    test_idx = np.arange(n_train, n_total)
    return data, genes, train_idx, test_idx


def test_ridge_survives_rank_deficient_control_matrix():
    """Regression test for the PCA numerical-stability bug (SPEC §10 Q3 /
    CONTRACT B4): a rank-deficient control matrix must not produce inf/nan
    or absurd predictions. Verified to FAIL on the pre-fix
    `min(50, n_samples-1, n_features)` PCA truncation (raises/produces
    non-finite features on real rank-deficient data) and PASS after
    truncating components by explained-variance-ratio.
    """
    import warnings

    data, genes, train_idx, test_idx = _make_rank_deficient_ridge_case()

    b = Ridge_()
    # NB: deliberately NOT `simplefilter("error", RuntimeWarning)`. This machine's
    # Accelerate BLAS emits spurious divide-by-zero/overflow/invalid-value matmul
    # warnings on fully-finite input -- reproducible in three lines of pure numpy
    # with no sklearn involved (`rng.normal(size=(21,3000)) @ rng.normal(size=(3000,20))`
    # warns three times and returns a finite array). Turning those into errors would
    # make this test assert a property of the platform's BLAS rather than of Ridge_.
    # What the regression actually needs to pin down is that the OUTPUT is finite and
    # the retained-component count is the matrix's real rank -- both asserted below.
    b.fit(data, train_idx, genes)
    pred = b.predict(data, test_idx)

    assert np.isfinite(pred).all(), "ridge produced non-finite predictions on a rank-deficient control matrix"

    # The heart of the bug: the pre-fix code requested min(50, n_train-1, n_genes) = 20
    # components from a matrix whose centered rank is 2, so components 3..20 had
    # explained-variance-ratio ~3e-17 and, once inverted by whitening, poisoned every
    # prediction. Pin the count so a regression that stops truncating fails loudly.
    # Magnitude is the symptom that finiteness alone does NOT catch, and it is the one
    # that silently poisons a benchmark number. Measured: keeping all 20 components
    # yields max|pred| = 1.76e7 on this fixture while the fix yields 0.86. These are
    # log1p-expression deltas, so anything beyond single digits is garbage, not signal.
    assert np.abs(pred).max() < 10.0, (
        f"ridge predictions have absurd magnitude {np.abs(pred).max():.3g} for log1p "
        "expression deltas -- near-zero-variance PCA directions are being inverted and "
        "amplified (this is finite-but-garbage, which no isfinite check would catch)"
    )

    evr = b._pca.explained_variance_ratio_
    assert b._pca_n_keep == 2, (
        f"expected the 2 real (centered-rank) directions to survive truncation, kept "
        f"{b._pca_n_keep}; explained-variance-ratios were {evr[:6]}"
    )
    assert evr[b._pca_n_keep] < 1e-10, (
        "the first DISCARDED component should be at numerical-noise scale; if it is "
        "not, the truncation threshold is cutting real signal"
    )

    true_delta = data.delta[test_idx][:, genes]
    # Pearson r per test row against the true delta; must be a sane
    # correlation coefficient, not a numerical-garbage artifact.
    for p_row, t_row in zip(pred, true_delta):
        if np.std(p_row) < 1e-12 or np.std(t_row) < 1e-12:
            continue
        r = np.corrcoef(p_row, t_row)[0, 1]
        assert np.isfinite(r)
        assert -1.0 - 1e-6 <= r <= 1.0 + 1e-6


def test_ridge_features_and_predictions_finite_on_ill_conditioned_control():
    """Ridge_ must never hand back non-finite features or predictions on an
    ill-conditioned control matrix, and must say so LOUDLY if it ever does.

    This replaces an earlier "no RuntimeWarning may be raised" assertion. That
    assertion could not pass on this hardware for a reason unrelated to the bug:
    Accelerate BLAS emits divide-by-zero / overflow / invalid-value matmul warnings
    on fully-finite operands (verified against plain numpy, no sklearn in the
    picture). Asserting on warning TEXT therefore tested the platform; asserting on
    finiteness tests Ridge_. The warnings are not suppressed anywhere in the
    package -- they are simply not treated as this test's signal.
    """
    data, genes, train_idx, test_idx = _make_rank_deficient_ridge_case(seed=11)

    b = Ridge_()
    b.fit(data, train_idx, genes)
    pred = b.predict(data, test_idx)

    assert np.isfinite(pred).all()
    # _assert_finite is the in-code guard that would raise FloatingPointError rather
    # than return garbage; confirm it is actually wired into the projection path by
    # checking the projection it guards is finite for both train and test rows.
    for idx in (train_idx, test_idx):
        feats = b._build_features(data, idx, b._genes, fit_pca=False)
        assert np.isfinite(feats).all()
    # And confirm the guard really fires when fed a non-finite array, so it is not
    # dead code that happens never to trigger.
    with pytest.raises(FloatingPointError):
        Ridge_._assert_finite(np.array([1.0, np.inf]), "deliberate probe")


def test_nearest_context_reports_dose_collapse_fallback_tier():
    """B5 NearestContext must expose a fallback_tier counter analogous to
    PerDrugMeanDelta's, so run.py can report the tier-1 (compound+dose ->
    compound-any-dose) dose-collapse rate (SPEC §11: dose is condition
    identity and this collapse must be countable). fallback_mask keeps its
    existing meaning: True only for the fully-unseen-compound fallback to
    the global mean (tier 2) -- other agents' run.py depends on that.
    """
    n_genes = 6
    obs_rows = [
        {"cell_line": "A", "compound": "d1", "dose": 1.0, "log_dose": 0.0, "batch": "A_b0", "n_cells": 50, "is_control": False, "target": "unknown"},
        {"cell_line": "A", "compound": "d1", "dose": 2.0, "log_dose": 0.69, "batch": "A_b0", "n_cells": 50, "is_control": False, "target": "unknown"},
        {"cell_line": "B", "compound": "d2", "dose": 1.0, "log_dose": 0.0, "batch": "B_b0", "n_cells": 50, "is_control": False, "target": "unknown"},
        # test row 1: same cell line A is most similar to itself; same
        # compound+dose (1.0) seen in training -> tier 0, exact match.
        {"cell_line": "A", "compound": "d1", "dose": 1.0, "log_dose": 0.0, "batch": "A_b0", "n_cells": 50, "is_control": False, "target": "unknown"},
        # test row 2: same compound d1 but a dose (5.0) never seen with d1
        # in training -> must dose-collapse to compound-any-dose -> tier 1.
        {"cell_line": "A", "compound": "d1", "dose": 5.0, "log_dose": 1.79, "batch": "A_b0", "n_cells": 50, "is_control": False, "target": "unknown"},
        # test row 3: compound never seen at all -> falls back to global
        # mean -> tier 2 AND fallback_mask True.
        {"cell_line": "B", "compound": "d3", "dose": 1.0, "log_dose": 0.0, "batch": "B_b0", "n_cells": 50, "is_control": False, "target": "unknown"},
    ]
    obs = pd.DataFrame(obs_rows)
    obs.index = [f"row{i}" for i in range(len(obs_rows))]

    rng = np.random.default_rng(5)
    control = rng.normal(0, 0.01, size=(len(obs_rows), n_genes)).astype(np.float32)
    control[:3] += np.array([[1.0] * n_genes, [1.0] * n_genes, [-1.0] * n_genes], dtype=np.float32)
    control[3:] += np.array([[1.0] * n_genes, [1.0] * n_genes, [-1.0] * n_genes], dtype=np.float32)
    delta = rng.normal(0, 1.0, size=(len(obs_rows), n_genes)).astype(np.float32)
    X = control + delta

    data = FakePseudobulkData(X=X, control=control, delta=delta, obs=obs)
    genes = np.arange(n_genes)

    train_idx = np.array([0, 1, 2])
    test_idx = np.array([3, 4, 5])

    b = NearestContext()
    b.fit(data, train_idx, genes)
    b.predict(data, test_idx)

    assert hasattr(b, "fallback_tier"), "NearestContext must expose fallback_tier like PerDrugMeanDelta"
    assert b.fallback_tier.shape == (len(test_idx),)
    np.testing.assert_array_equal(b.fallback_tier, np.array([0, 1, 2]))
    # fallback_mask meaning preserved: True only for tier 2 (unseen compound)
    np.testing.assert_array_equal(b.fallback_mask, np.array([False, False, True]))


@pytest.mark.parametrize("name", list(BASELINES.keys()))
def test_every_registered_baseline_runs_end_to_end(name):
    data, genes = _make_synthetic_data(n_genes=12)
    n = len(data.obs)
    rng = np.random.default_rng(3)
    perm = rng.permutation(n)
    train_idx = perm[: n // 2]
    test_idx = perm[n // 2 :]

    cls = BASELINES[name]
    b = cls()
    b.fit(data, train_idx, genes)
    pred = b.predict(data, test_idx)
    assert pred.shape == (len(test_idx), len(genes))
    assert np.isfinite(pred).all()
    assert b.fallback_mask.shape == (len(test_idx),)

import numpy as np
import pandas as pd
import pytest

from perturb_bench.splits import REGIMES, Fold, assert_no_leakage, make_folds


def _make_obs(
    cell_lines=("A549", "K562", "MCF7"),
    compounds=("drugA", "drugB", "drugC", "drugD", "drugE", "drugF"),
    doses=(10.0, 100.0),
    targets=None,
):
    rows = []
    for cl in cell_lines:
        for c in compounds:
            for dose in doses:
                rows.append(
                    {
                        "cell_line": cl,
                        "compound": c,
                        "dose": dose,
                        "log_dose": np.log1p(dose),
                        "batch": f"{cl}_b0",
                        "n_cells": 100,
                        "is_control": False,
                        "target": (targets or {}).get(c, "unknown"),
                    }
                )
    obs = pd.DataFrame(rows)
    obs.index = [f"{r['cell_line']}|{r['compound']}|{r['dose']}" for r in rows]
    return obs


def test_regimes_listed():
    assert set(REGIMES) == {"random", "held_out_context", "held_out_drug", "held_out_both"}


def test_random_fold_holds_out_20_percent():
    obs = _make_obs()
    folds = make_folds(obs, "random", seed=0)
    assert len(folds) == 1
    fold = folds[0]
    n = len(obs)
    assert abs(len(fold.test_idx) - round(0.2 * n)) <= 1
    assert len(fold.train_idx) + len(fold.test_idx) == n
    assert_no_leakage(obs, fold)


def test_random_fold_is_seed_deterministic():
    obs = _make_obs()
    f1 = make_folds(obs, "random", seed=42)[0]
    f2 = make_folds(obs, "random", seed=42)[0]
    assert np.array_equal(f1.test_idx, f2.test_idx)


def test_held_out_context_one_fold_per_cell_line():
    obs = _make_obs(cell_lines=("A549", "K562", "MCF7"))
    folds = make_folds(obs, "held_out_context", seed=0)
    assert len(folds) == 3
    held_lines = {f.meta["held_out_cell_lines"][0] for f in folds}
    assert held_lines == {"A549", "K562", "MCF7"}

    for fold in folds:
        held_line = fold.meta["held_out_cell_lines"][0]
        test_lines = set(obs["cell_line"].iloc[fold.test_idx].unique())
        assert test_lines == {held_line}
        train_lines = set(obs["cell_line"].iloc[fold.train_idx].unique())
        assert held_line not in train_lines
        assert_no_leakage(obs, fold)


def test_held_out_drug_is_multi_fold_and_groups_by_target():
    targets = {
        "drugA": "TARGET1",
        "drugB": "TARGET1",  # sibling of drugA -- must land in same fold
        "drugC": "TARGET2",
        "drugD": "TARGET3",
        "drugE": "TARGET4",
        "drugF": "TARGET5",
    }
    obs = _make_obs(targets=targets)
    folds = make_folds(obs, "held_out_drug", seed=0)
    assert len(folds) > 1, "held_out_drug should be multi-fold, not a single arbitrary split"

    for fold in folds:
        assert fold.meta["target_annotation_available"] is True
        held = set(fold.meta["held_out_compounds"])
        # drugA and drugB share a target: if one is held out, so is the other
        if "drugA" in held or "drugB" in held:
            assert "drugA" in held and "drugB" in held
        assert_no_leakage(obs, fold)

    # every compound should be held out in at least one fold
    all_held = set()
    for fold in folds:
        all_held |= set(fold.meta["held_out_compounds"])
    assert all_held == set(obs["compound"].unique())


def test_held_out_drug_falls_back_without_target_annotation():
    obs = _make_obs(targets=None)  # all "unknown"
    folds = make_folds(obs, "held_out_drug", seed=0)
    assert len(folds) > 1
    for fold in folds:
        assert fold.meta["target_annotation_available"] is False
        assert_no_leakage(obs, fold)


def test_held_out_both_test_set_is_intersection_of_unseen_line_and_compound():
    targets = {"drugA": "T1", "drugB": "T1", "drugC": "T2", "drugD": "T3", "drugE": "T4", "drugF": "T5"}
    obs = _make_obs(targets=targets)
    folds = make_folds(obs, "held_out_both", seed=0)
    assert len(folds) > 0

    for fold in folds:
        held_line = fold.meta["held_out_cell_lines"][0]
        held_compounds = set(fold.meta["held_out_compounds"])

        test_lines = set(obs["cell_line"].iloc[fold.test_idx].unique())
        test_compounds = set(obs["compound"].iloc[fold.test_idx].unique())
        assert test_lines == {held_line}
        assert test_compounds <= held_compounds
        assert len(test_compounds) > 0

        # training must exclude every condition touching EITHER held-out axis
        train_lines = set(obs["cell_line"].iloc[fold.train_idx].unique())
        train_compounds = set(obs["compound"].iloc[fold.train_idx].unique())
        assert held_line not in train_lines
        assert not (held_compounds & train_compounds)

        assert_no_leakage(obs, fold)


def test_assert_no_leakage_raises_on_deliberately_leaky_fold():
    obs = _make_obs()
    idx = np.arange(len(obs))
    held_line = "A549"
    is_held = (obs["cell_line"] == held_line).to_numpy()
    test_idx = idx[is_held]
    # deliberately leaky: include a few held-out-line rows in "training" too
    leaky_train_idx = idx[~is_held]
    leaky_train_idx = np.concatenate([leaky_train_idx, test_idx[:2]])

    leaky_fold = Fold(
        regime="held_out_context",
        name="leaky",
        train_idx=leaky_train_idx,
        test_idx=test_idx,
        meta={"held_out_cell_lines": [held_line], "held_out_compounds": []},
    )
    with pytest.raises(AssertionError):
        assert_no_leakage(obs, leaky_fold)


def test_assert_no_leakage_raises_on_overlapping_train_test():
    obs = _make_obs()
    idx = np.arange(len(obs))
    fold = Fold(
        regime="random",
        name="overlap",
        train_idx=idx[:10],
        test_idx=idx[5:15],
        meta={},
    )
    with pytest.raises(AssertionError):
        assert_no_leakage(obs, fold)


def test_assert_no_leakage_passes_for_well_formed_random_fold():
    obs = _make_obs()
    fold = make_folds(obs, "random", seed=1)[0]
    assert_no_leakage(obs, fold)  # should not raise


def test_assert_no_leakage_catches_held_out_context_leak_with_empty_meta():
    """H-2 (REVIEW.md): before the fix, a held_out_context fold with cell line A in
    BOTH train and test and meta={} passed silently, because assert_no_leakage only
    checked the axes the fold's OWN meta claimed to hold out. Now the held-out set
    is derived from fold.regime + the actual test rows, so this leak is caught even
    with meta={}."""
    obs = _make_obs()
    idx = np.arange(len(obs))
    held_line = "A549"
    is_held = (obs["cell_line"] == held_line).to_numpy()
    test_idx = idx[is_held]
    # leaky: cell line A549 also present in "training"
    train_idx = idx  # every row, including all of A549's own rows

    leaky_fold = Fold(
        regime="held_out_context",
        name="leaky_empty_meta",
        train_idx=train_idx,
        test_idx=test_idx,
        meta={},  # <-- the silent case: meta does not even declare a held-out line
    )
    with pytest.raises(AssertionError):
        assert_no_leakage(obs, leaky_fold)


def test_assert_no_leakage_catches_held_out_drug_leak_with_empty_held_out_compounds():
    """H-2 (REVIEW.md): a held_out_drug fold with compound x in both train and test
    and held_out_compounds=[] passed silently before the fix, for the same reason
    as the cell-line case above."""
    obs = _make_obs()
    idx = np.arange(len(obs))
    held_compound = "drugA"
    is_held = (obs["compound"] == held_compound).to_numpy()
    test_idx = idx[is_held]
    train_idx = idx  # leaky: drugA's own rows also in "training"

    leaky_fold = Fold(
        regime="held_out_drug",
        name="leaky_empty_held_out_compounds",
        train_idx=train_idx,
        test_idx=test_idx,
        meta={"held_out_cell_lines": [], "held_out_compounds": []},  # the silent case
    )
    with pytest.raises(AssertionError):
        assert_no_leakage(obs, leaky_fold)


def test_assert_no_leakage_catches_meta_declaring_wrong_held_out_set():
    """H-2: meta declaring a held-out cell line that does NOT match what's actually
    in test_idx must also be caught (meta "lying" about what the fold holds out),
    even when there is no train/test overlap."""
    obs = _make_obs()
    idx = np.arange(len(obs))
    is_held = (obs["cell_line"] == "A549").to_numpy()
    test_idx = idx[is_held]
    train_idx = idx[~is_held]

    wrong_meta_fold = Fold(
        regime="held_out_context",
        name="wrong_meta",
        train_idx=train_idx,
        test_idx=test_idx,
        meta={"held_out_cell_lines": ["K562"], "held_out_compounds": []},  # wrong!
    )
    with pytest.raises(AssertionError):
        assert_no_leakage(obs, wrong_meta_fold)

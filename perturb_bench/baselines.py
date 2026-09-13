"""Baselines for the perturbation-response benchmark harness. SPEC §5.

Common interface (see CONTRACT.md):
    fit(data, train_idx, genes) -> None
    predict(data, test_idx) -> (n_test, n_genes) predicted DELTA
    fallback_mask -> bool (n_test,) property, valid for the most recent predict() call

`data` is expected to be an instance of `perturb_bench.data.PseudobulkData`, but
we never import that module at the top level: data.py is being written in
parallel and may be missing or broken. We only need duck-typed attribute
access (X, control, delta, obs, var), so nothing here imports data.py except
behind `TYPE_CHECKING` for annotations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids a hard runtime dependency
    from .data import PseudobulkData


def _sel(arr: np.ndarray, idx: np.ndarray, genes: np.ndarray) -> np.ndarray:
    """arr[idx][:, genes] without materializing the full row slice twice."""
    idx = np.asarray(idx)
    genes = np.asarray(genes)
    return arr[np.ix_(idx, genes)]


class BaseBaseline:
    """Shared bookkeeping: name, fitted genes, fallback mask plumbing."""

    name: str = "base"

    def __init__(self) -> None:
        self._genes: np.ndarray | None = None
        self._fallback_mask: np.ndarray = np.zeros(0, dtype=bool)

    @property
    def fallback_mask(self) -> np.ndarray:
        return self._fallback_mask

    def fit(self, data: "PseudobulkData", train_idx: np.ndarray, genes: np.ndarray) -> None:
        raise NotImplementedError

    def predict(self, data: "PseudobulkData", test_idx: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NoChange(BaseBaseline):
    """B1: predict zero delta. Zero parameters. The floor / inflation detector."""

    name = "no_change"

    def fit(self, data: "PseudobulkData", train_idx: np.ndarray, genes: np.ndarray) -> None:
        self._genes = np.asarray(genes)

    def predict(self, data: "PseudobulkData", test_idx: np.ndarray) -> np.ndarray:
        n_test = len(test_idx)
        n_genes = len(self._genes)
        self._fallback_mask = np.zeros(n_test, dtype=bool)
        return np.zeros((n_test, n_genes), dtype=np.float32)


class GlobalMeanDelta(BaseBaseline):
    """B2: mean delta over ALL training conditions. One vector for everything."""

    name = "global_mean_delta"

    def __init__(self) -> None:
        super().__init__()
        self._mean_delta: np.ndarray | None = None

    def fit(self, data: "PseudobulkData", train_idx: np.ndarray, genes: np.ndarray) -> None:
        self._genes = np.asarray(genes)
        train_delta = _sel(data.delta, train_idx, self._genes)
        self._mean_delta = train_delta.mean(axis=0)

    def predict(self, data: "PseudobulkData", test_idx: np.ndarray) -> np.ndarray:
        n_test = len(test_idx)
        self._fallback_mask = np.zeros(n_test, dtype=bool)
        return np.tile(self._mean_delta, (n_test, 1)).astype(np.float32)


class PerDrugMeanDelta(BaseBaseline):
    """B3: the key adversary.

    pred[c*, compound, dose] = mean over training conditions with the SAME
    compound AND SAME dose. Falls back to same-compound-any-dose, then to the
    global mean (B2). fallback_mask is True only for the last tier: the
    compound was entirely unseen in training and we fell all the way to B2.
    """

    name = "per_drug_mean_delta"

    def __init__(self) -> None:
        super().__init__()
        self._global_mean: np.ndarray | None = None
        self._by_compound_dose: dict[tuple[str, float], np.ndarray] = {}
        self._by_compound: dict[str, np.ndarray] = {}
        self.fallback_tier: np.ndarray = np.zeros(0, dtype=int)  # 0=exact,1=compound-only,2=global

    def fit(self, data: "PseudobulkData", train_idx: np.ndarray, genes: np.ndarray) -> None:
        self._genes = np.asarray(genes)
        train_idx = np.asarray(train_idx)
        train_delta = _sel(data.delta, train_idx, self._genes)
        obs = data.obs.iloc[train_idx]

        self._global_mean = train_delta.mean(axis=0)

        by_cd: dict[tuple[str, float], list[np.ndarray]] = {}
        by_c: dict[str, list[np.ndarray]] = {}
        compounds = obs["compound"].to_numpy()
        doses = obs["dose"].to_numpy()
        for i in range(len(train_idx)):
            c = compounds[i]
            d = float(doses[i])
            by_cd.setdefault((c, d), []).append(train_delta[i])
            by_c.setdefault(c, []).append(train_delta[i])

        self._by_compound_dose = {k: np.mean(v, axis=0) for k, v in by_cd.items()}
        self._by_compound = {k: np.mean(v, axis=0) for k, v in by_c.items()}

    def predict(self, data: "PseudobulkData", test_idx: np.ndarray) -> np.ndarray:
        test_idx = np.asarray(test_idx)
        obs = data.obs.iloc[test_idx]
        n_test = len(test_idx)
        n_genes = len(self._genes)
        out = np.zeros((n_test, n_genes), dtype=np.float32)
        fallback = np.zeros(n_test, dtype=bool)
        tiers = np.zeros(n_test, dtype=int)

        compounds = obs["compound"].to_numpy()
        doses = obs["dose"].to_numpy()
        for i in range(n_test):
            c = compounds[i]
            d = float(doses[i])
            key_cd = (c, d)
            if key_cd in self._by_compound_dose:
                out[i] = self._by_compound_dose[key_cd]
                tiers[i] = 0
            elif c in self._by_compound:
                out[i] = self._by_compound[c]
                tiers[i] = 1
            else:
                out[i] = self._global_mean
                tiers[i] = 2
                fallback[i] = True

        self._fallback_mask = fallback
        self.fallback_tier = tiers
        return out


class Ridge_(BaseBaseline):
    """B4: ridge on [PCA(control_expression[c*]); onehot(compound); log_dose] -> delta.

    Control expression is high-dimensional; we reduce it with a PCA fit on
    TRAINING rows only (leakage rules apply to PCA too — documented here).
    alpha is tuned by KFold CV over training conditions (never touching test
    rows). fallback_mask is True for test conditions whose compound was never
    seen in training (one-hot is all zeros -> silent degradation to an
    effectively drug-agnostic, undertrained prediction).
    """

    name = "ridge"
    use_drug_onehot = True
    _n_pca_components = 50
    # Components explaining less than this fraction of the TRAINING
    # control matrix's variance are dropped before they ever reach Ridge.
    #
    # Root cause this guards against: `min(50, n_samples-1, n_features)`
    # requests up to n_samples-1 components regardless of the matrix's TRUE
    # rank. Real control matrices are rank-deficient in exactly this way --
    # many training conditions share the same handful of cell-line control
    # profiles (near-duplicate rows) -- so PCA's "full" SVD dutifully
    # returns trailing components with explained-variance-ratio down at
    # float64-machine-epsilon scale (measured as low as ~1e-98/1e-128 on a
    # synthetic repro of this exact structure). Those directions carry
    # ~zero TRAIN-row variance but project to O(1), non-zero values on a
    # genuinely unseen test row (PCA components are unit vectors; a tiny
    # eigenvalue says nothing about how large a held-out point's projection
    # onto that eigenvector will be). Keeping them is precisely what
    # produced the divide-by-zero/overflow/invalid-value RuntimeWarnings
    # from sklearn's PCA/Ridge matmuls on real sciplex3 folds and the
    # catastrophic held_out_context ridge score.
    #
    # 1e-6 is the threshold: comfortably above float64 eps (~2e-16) and the
    # ~1e-30-to-1e-290 noise-floor ratios rank deficiency produces, and
    # comfortably below the variance explained by any PCA direction that
    # could plausibly generalize on this kind of expression data (real
    # components of interest here explain >=1e-3 to 1e-2 of variance, per
    # REVIEW.md's measurements on the actual cache). It is a class
    # attribute so a caller can tighten/loosen it without subclassing.
    _min_variance_ratio = 1e-6

    _alphas = (0.1, 1.0, 10.0, 100.0, 1000.0)
    _cv_folds = 5

    def __init__(self) -> None:
        super().__init__()
        self._pca: PCA | None = None
        self._pca_n_keep: int | None = None
        self._compound_to_col: dict[str, int] = {}
        self._model: Ridge | None = None
        self._best_alpha: float | None = None
        self._log_dose_mean: float = 0.0
        self._log_dose_std: float = 1.0

    @staticmethod
    def _assert_finite(arr: np.ndarray, where: str) -> None:
        if not np.isfinite(arr).all():
            n_bad = int((~np.isfinite(arr)).sum())
            raise FloatingPointError(
                f"Ridge_: non-finite values ({n_bad}/{arr.size}) produced at {where}. "
                "A silently-corrupted feature/prediction is exactly what this harness "
                "exists to prevent -- refusing to return garbage. See "
                "Ridge_._min_variance_ratio / _build_features for the PCA truncation "
                "this guards."
            )

    def _project_control(self, control: np.ndarray) -> np.ndarray:
        """Project control expression onto the KEPT PCA components only, then whiten.

        We do this by hand rather than calling ``PCA.transform`` because sklearn's
        transform projects onto (and, with ``whiten=True``, divides by the variance
        of) EVERY fitted component before a caller gets a chance to slice. On a
        rank-deficient control matrix the trailing components have
        explained variance at float64-epsilon scale, so that full projection
        overflows or yields inf/nan -- the truncation has to happen BEFORE the
        arithmetic, not after it.

        Whitening (dividing by the component standard deviation) keeps every
        retained PCA direction on unit scale so Ridge's single alpha penalizes the
        control block, the compound one-hot block and log_dose comparably.
        """
        assert self._pca is not None and self._pca_n_keep is not None
        keep = self._pca_n_keep
        comps = self._pca.components_[:keep]  # (keep, n_genes)
        var = self._pca.explained_variance_[:keep]  # strictly > 0 by construction
        centered = control - self._pca.mean_
        projected = centered @ comps.T
        whitened = projected / np.sqrt(var)
        self._assert_finite(whitened, "whitened PCA projection of control expression")
        return whitened

    def _build_features(
        self, data: "PseudobulkData", idx: np.ndarray, control_genes: np.ndarray, fit_pca: bool
    ) -> np.ndarray:
        # float64 throughout: keeps the PCA/whitening/Ridge arithmetic in
        # double precision end to end (see CONTRACT; data.control is
        # float32 on disk, so we upcast here rather than losing precision
        # mid-pipeline). svd_solver is pinned to "full" (exact, and cheap at
        # our problem sizes: at most a few hundred conditions x a few
        # thousand genes).
        control = _sel(data.control, idx, control_genes).astype(np.float64)
        if fit_pca:
            n_comp = min(self._n_pca_components, control.shape[0] - 1, control.shape[1])
            n_comp = max(1, n_comp)
            # whiten=True rescales each retained component to unit variance
            # (dividing by sqrt(explained_variance_)) so Ridge's single
            # alpha penalizes every PCA direction, the onehot block, and
            # log_dose on a comparable scale, instead of implicitly
            # under-regularizing whichever block happens to have the
            # largest raw numeric range. This is safe ONLY because we
            # truncate near-zero-variance components BEFORE whitening can
            # divide by them -- whitening first (or not truncating) is
            # exactly the inversion-of-near-zero-variance failure mode this
            # fix removes.
            # NOTE: whiten is deliberately FALSE here. sklearn's
            # PCA.transform(whiten=True) whitens ALL n_comp components and only
            # then would we slice -- so the near-zero-variance directions are
            # divided by sqrt(~0) *during* transform, producing inf/nan in the
            # matmul before truncation can ever take effect. (That was the
            # actual bug: the truncation was real but happened one step too
            # late.) We fit unwhitened, decide `keep` from the
            # explained-variance ratios, and then project manually in
            # `_project_control` using ONLY the kept components, whitening by
            # hand. The discarded directions are never multiplied through.
            pca = PCA(n_components=n_comp, random_state=0, svd_solver="full", whiten=False)
            pca.fit(control)
            evr = pca.explained_variance_ratio_  # sorted descending by construction
            keep = int(np.sum(evr > self._min_variance_ratio))
            if keep == 0:
                # Degenerate case: even the dominant direction is below the
                # threshold (e.g. an almost-perfectly-constant control
                # matrix). Keep it anyway rather than emitting a zero-width
                # feature block -- one weak feature beats none.
                keep = 1
            # A retained component must have strictly positive variance or the
            # manual whitening below divides by zero. Tighten `keep` to the
            # prefix that actually does.
            pos = np.flatnonzero(pca.explained_variance_[:keep] > 0.0)
            keep = int(pos[-1]) + 1 if pos.size else 0
            if keep == 0:
                raise RuntimeError(
                    "Ridge_: PCA found no control-expression direction with "
                    "strictly positive variance. The control matrix is "
                    "degenerate (constant or empty); refusing to emit a "
                    "silently meaningless feature block."
                )
            self._pca = pca
            self._pca_n_keep = keep
            control_reduced = self._project_control(control)
        else:
            assert self._pca is not None and self._pca_n_keep is not None, (
                "PCA must be fit before transform"
            )
            control_reduced = self._project_control(control)

        self._assert_finite(control_reduced, "PCA-reduced control features")
        del control  # large; drop the reference promptly

        obs = data.obs.iloc[idx]
        log_dose = obs["log_dose"].to_numpy(dtype=np.float64).reshape(-1, 1)
        if fit_pca:
            self._log_dose_mean = float(log_dose.mean())
            std = float(log_dose.std())
            self._log_dose_std = std if std > 1e-12 else 1.0
        log_dose_scaled = (log_dose - self._log_dose_mean) / self._log_dose_std

        if self.use_drug_onehot:
            onehot = np.zeros((len(idx), len(self._compound_to_col)), dtype=np.float64)
            compounds = obs["compound"].to_numpy()
            for i, c in enumerate(compounds):
                col = self._compound_to_col.get(c)
                if col is not None:
                    onehot[i, col] = 1.0
            feats = np.concatenate([control_reduced, onehot, log_dose_scaled], axis=1)
        else:
            feats = np.concatenate([control_reduced, log_dose_scaled], axis=1)

        self._assert_finite(feats, "assembled Ridge feature matrix")
        return feats

    def fit(self, data: "PseudobulkData", train_idx: np.ndarray, genes: np.ndarray) -> None:
        self._genes = np.asarray(genes)
        train_idx = np.asarray(train_idx)

        if self.use_drug_onehot:
            train_compounds = sorted(data.obs.iloc[train_idx]["compound"].unique().tolist())
            self._compound_to_col = {c: i for i, c in enumerate(train_compounds)}

        X_train = self._build_features(data, train_idx, self._genes, fit_pca=True)
        y_train = _sel(data.delta, train_idx, self._genes).astype(np.float64)

        n = X_train.shape[0]
        cv_folds = min(self._cv_folds, n)
        if cv_folds < 2:
            # not enough training conditions to cross-validate; use a fixed
            # middle-of-the-road alpha
            self._best_alpha = 1.0
        else:
            kf = KFold(n_splits=cv_folds, shuffle=True, random_state=0)
            best_alpha, best_score = None, -np.inf
            for alpha in self._alphas:
                scores = []
                for tr, va in kf.split(X_train):
                    m = Ridge(alpha=alpha)
                    m.fit(X_train[tr], y_train[tr])
                    pred = m.predict(X_train[va])
                    # negative MSE as the CV score (higher is better)
                    scores.append(-float(np.mean((pred - y_train[va]) ** 2)))
                mean_score = float(np.mean(scores))
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha
            self._best_alpha = best_alpha

        self._model = Ridge(alpha=self._best_alpha)
        self._model.fit(X_train, y_train)

    def predict(self, data: "PseudobulkData", test_idx: np.ndarray) -> np.ndarray:
        test_idx = np.asarray(test_idx)
        assert self._model is not None, "fit() must be called before predict()"

        X_test = self._build_features(data, test_idx, self._genes, fit_pca=False)
        preds = self._model.predict(X_test)
        self._assert_finite(preds, "Ridge_.predict output")
        preds = preds.astype(np.float32)

        fallback = np.zeros(len(test_idx), dtype=bool)
        if self.use_drug_onehot:
            compounds = data.obs.iloc[test_idx]["compound"].to_numpy()
            for i, c in enumerate(compounds):
                if c not in self._compound_to_col:
                    fallback[i] = True
        self._fallback_mask = fallback
        return preds


class RidgeNoDrug(Ridge_):
    """B4b: drug-agnostic ridge variant, features [control_expression; log_dose] only.

    SPEC §5 forbids silently changing B4's encoding for held_out_drug, so this
    is registered separately and used explicitly in that regime.
    """

    name = "ridge_no_drug"
    use_drug_onehot = False

    def predict(self, data: "PseudobulkData", test_idx: np.ndarray) -> np.ndarray:
        test_idx = np.asarray(test_idx)
        assert self._model is not None, "fit() must be called before predict()"
        X_test = self._build_features(data, test_idx, self._genes, fit_pca=False)
        preds = self._model.predict(X_test)
        self._assert_finite(preds, "RidgeNoDrug.predict output")
        preds = preds.astype(np.float32)
        # no drug feature is used at all, so there is no "unseen onehot" failure mode
        self._fallback_mask = np.zeros(len(test_idx), dtype=bool)
        return preds


class NearestContext(BaseBaseline):
    """B5: copy the delta of the training cell line whose mean control profile
    is most similar (cosine similarity) to the held-out condition's control,
    for the same compound+dose. Falls back to the global mean (B2) when no
    training cell line has that compound at all, or when the most similar
    cell lines never saw that exact compound+dose.
    """

    name = "nearest_context"

    def __init__(self) -> None:
        super().__init__()
        self._line_mean_control: dict[str, np.ndarray] = {}
        self._line_cd_delta: dict[str, dict[tuple[str, float], np.ndarray]] = {}
        self._line_c_delta: dict[str, dict[str, np.ndarray]] = {}
        self._global_mean: np.ndarray | None = None
        # 0 = exact compound+dose match on the most-similar line,
        # 1 = dose-collapsed to compound-any-dose on some line (SPEC §11:
        #     dose is part of condition identity, so this must be countable),
        # 2 = compound entirely unseen in training -> fell back to the
        #     global mean (B2). fallback_mask stays True ONLY for tier 2 --
        #     its existing meaning (run.py depends on it) is unchanged.
        self.fallback_tier: np.ndarray = np.zeros(0, dtype=int)

    def fit(self, data: "PseudobulkData", train_idx: np.ndarray, genes: np.ndarray) -> None:
        self._genes = np.asarray(genes)
        train_idx = np.asarray(train_idx)
        obs = data.obs.iloc[train_idx]
        control = _sel(data.control, train_idx, self._genes)
        delta = _sel(data.delta, train_idx, self._genes)

        self._global_mean = delta.mean(axis=0)

        lines = obs["cell_line"].to_numpy()
        compounds = obs["compound"].to_numpy()
        doses = obs["dose"].to_numpy()

        control_by_line: dict[str, list[np.ndarray]] = {}
        cd_delta: dict[str, dict[tuple[str, float], list[np.ndarray]]] = {}
        c_delta: dict[str, dict[str, list[np.ndarray]]] = {}
        for i in range(len(train_idx)):
            line = lines[i]
            control_by_line.setdefault(line, []).append(control[i])
            cd_delta.setdefault(line, {}).setdefault((compounds[i], float(doses[i])), []).append(delta[i])
            c_delta.setdefault(line, {}).setdefault(compounds[i], []).append(delta[i])

        self._line_mean_control = {k: np.mean(v, axis=0) for k, v in control_by_line.items()}
        self._line_cd_delta = {
            line: {k: np.mean(v, axis=0) for k, v in d.items()} for line, d in cd_delta.items()
        }
        self._line_c_delta = {
            line: {k: np.mean(v, axis=0) for k, v in d.items()} for line, d in c_delta.items()
        }

    @staticmethod
    def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return -np.inf
        return float(np.dot(a, b) / (na * nb))

    def predict(self, data: "PseudobulkData", test_idx: np.ndarray) -> np.ndarray:
        test_idx = np.asarray(test_idx)
        obs = data.obs.iloc[test_idx]
        control = _sel(data.control, test_idx, self._genes)
        n_test = len(test_idx)
        n_genes = len(self._genes)
        out = np.zeros((n_test, n_genes), dtype=np.float32)
        fallback = np.zeros(n_test, dtype=bool)
        tiers = np.zeros(n_test, dtype=int)

        lines_by_sim = {}  # cache ranking per test row not needed, compute fresh
        train_lines = list(self._line_mean_control.keys())

        compounds = obs["compound"].to_numpy()
        doses = obs["dose"].to_numpy()
        for i in range(n_test):
            c = compounds[i]
            d = float(doses[i])
            sims = sorted(
                train_lines,
                key=lambda ln: self._cosine_sim(control[i], self._line_mean_control[ln]),
                reverse=True,
            )
            matched = False
            for ln in sims:
                cd_map = self._line_cd_delta.get(ln, {})
                if (c, d) in cd_map:
                    out[i] = cd_map[(c, d)]
                    matched = True
                    tiers[i] = 0
                    break
            if not matched:
                for ln in sims:
                    c_map = self._line_c_delta.get(ln, {})
                    if c in c_map:
                        out[i] = c_map[c]
                        matched = True
                        tiers[i] = 1
                        break
            if not matched:
                out[i] = self._global_mean
                fallback[i] = True
                tiers[i] = 2

        self._fallback_mask = fallback
        self.fallback_tier = tiers
        return out


BASELINES: dict[str, type] = {
    "no_change": NoChange,
    "global_mean_delta": GlobalMeanDelta,
    "per_drug_mean_delta": PerDrugMeanDelta,
    "ridge": Ridge_,
    "ridge_no_drug": RidgeNoDrug,
    "nearest_context": NearestContext,
}

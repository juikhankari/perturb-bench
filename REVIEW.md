# REVIEW.md — perturbation-response benchmark harness

Reviewer: Fable (Claude), 2026-09-12. Scope: SPEC.md, CONTRACT.md (+ADDENDUM 1/2),
DATA_RECON.md, `perturb_bench/*.py`, `tests/*.py`, the real sciplex3 cache at
`cache/pseudobulk_sciplex3_4ddea36a267a8dbc`, and the raw h5ad.

Everything below was verified by reading the code AND running probes against the real
cache / raw h5ad (probe scripts in the session scratchpad). Numbers are measured, not
assumed. Where I could not verify something, it is listed in §5.

## 0. Bottom line

The harness is structurally sound in the places where the literature usually fails:
splits are genuinely condition-level, `de_mask` comes only from the single-cell test,
HVG/PCA/ridge-alpha are train-only, `absolute` is labeled as a diagnostic, NaN is never
coerced to 0. The four Sonnet agents did honest work.

But **the normalized-score column — the harness's headline output — is currently wrong
for `mae` on every real fold, and mis-denominated for every metric on every fold**, for
two independent reasons neither the tests nor the report would reveal:

1. **The split-half `mae` ceiling is WORSE than the `no_change` floor** (measured on all
   four folds probed: ceiling 0.025–0.039 vs floor 0.023–0.033; split-half MAE exceeds
   no_change MAE for 86–94% of test conditions). `normalize_score` then silently inverts:
   a baseline with WORSE MAE than no_change is reported as normalized = +1.07 (A549),
   +1.82 (K562), **+7.31 (MCF7)**, +2.06 (random). Nothing checks that floor/ceiling
   bracket in the direction `METRIC_DIRECTION` says they should.
2. **The ceiling is computed over ALL 2,233 conditions, not the fold's test set**
   (`run.py:_ceiling_for_fold` never receives `test_idx`). The denominator is therefore
   not fold-matched (MCF7 mae ceiling: 0.0247 on its own test rows vs 0.0369 over all
   rows — 50% off), the `pert_discrimination` ceiling ranks against 2,233 distractors
   while baselines rank against 110–749, and this single choice is also what blows the
   runtime from minutes to hours (§1.1).

Both are silent. Both must be fixed before any number from `results.csv` is quoted.

**I applied ONE fix** (issue 2 above; small, unambiguous, CRITICAL) — see §4, flagged
separately. Everything else is planned, not implemented.

---

## 1. Verdicts on the four known issues

### 1.1 RUNTIME — CONFIRMED, and the cause is precisely the fold-mismatched ceiling

Profiled on the real cache (`n_genes=2000`):

| call | n rows | wall |
|---|---|---|
| `_pert_discrimination` | 447 | 0.75 s |
| `_pert_discrimination` | 745 | 0.81 s |
| `_pert_discrimination` | **2233** | **20.2 s** (super-linear: the `true - pred[i]` broadcast temp is 36 MB/iter and falls out of cache) |
| `score_conditions` for ONE ceiling rep, all 2233 conds, delta | 21.1 s |

The ceiling is scored `n_ceiling_reps(10) × 2 representations = 20` times per fold at
n=2233 → **~7 min per fold just for the ceiling**, × 24 folds ≈ **2.8 h**. Baselines are
negligible (~5 s/fold). The `--quick` run confirms: 45–50 s per fold even at 200 genes,
all of it ceiling. Your hypothesis was right on both counts: it is O(n²·g) AND it is over
all 2,233 conditions instead of the fold's test set. The "over all conditions" part is
not merely slow, it is wrong (§2, C-2).

After restricting the ceiling to test rows (the fix in §4), MEASURED: the same 7-fold
`--quick` sweep dropped from **5m37s to ~25s** (per-fold 45–50 s → 0.6–6.6 s). At 2000
genes the full 24-fold sweep should now land well under the 10-min SPEC target;
vectorizing `_pert_discrimination` (one `cdist` + `rankdata(axis=1)`) is a further
easy win.

**Incremental output: CONFIRMED absent.** `run.py:222-231` builds the whole `rows` list
and writes `results.csv` once at the end. A crash at fold 23 loses everything. The
currently running process (PID 2256, 39+ min CPU) has produced nothing and, on the
numbers above, would not finish for ~2 more hours — and its output would carry bugs C-1
and C-2 anyway. Recommend killing it.

### 1.2 MITO QC THRESHOLD — WORSE THAN DESCRIBED

Measured directly from the raw h5ad (all 799,317 cells, 11 s streaming pass):

| | pct_mito median | p75 | p90 | p99 |
|---|---|---|---|---|
| all cells | 16.95 | 23.51 | 30.27 | 46.8 |
| A549 | 19.27 | 25.68 | 32.50 | |
| K562 | 11.46 | 16.82 | 23.48 | |
| MCF7 | 17.73 | 23.68 | 29.52 | |
| vehicle (dose 0) | 19.22 | 27.49 | 36.98 | |

Fraction of cells dropped by threshold:

| threshold | overall | vehicle | 10 nM | 100 nM | 1 µM | 10 µM | A549 | K562 | MCF7 |
|---|---|---|---|---|---|---|---|---|---|
| **20% (current)** | **37.3%** | **47.2%** | 35.6 | 36.6 | 36.3 | 38.1 | **46.6** | **16.2** | **39.6** |
| 30% | 10.4% | 19.8% | 8.9 | 9.6 | 9.7 | 10.9 | 14.2 | 4.3 | 9.2 |
| 40% | 2.45% | 7.2% | 1.9 | 2.0 | 2.1 | 2.6 | 3.5 | 1.5 | 1.4 |
| 50% | 0.68% | 2.7% | 0.5 | 0.5 | 0.5 | 0.7 | 0.9 | 0.7 | 0.2 |

Three biases, in decreasing order of severity — the dose dependence you flagged is the
*smallest* of them:

1. **Cell-line dependent**: A549 loses 46.6% of cells, K562 16.2%. This is the exact
   axis `held_out_context` is measured on. A549 post-QC median cells/condition is 120
   vs MCF7 283.
2. **Control-vs-treated dependent**: vehicle cells are dropped at 47.2% vs 36–38% for
   treated. The control profile and the treated profile are filtered differently, so
   `delta = X - control` picks up a filter artifact in every condition.
3. Dose dependent: 35.6% → 38.1%, monotone but modest.

High-mito cells here are NOT the low-quality tail: they have HIGHER library size
(median 1,802 UMI vs 1,366 for ≤20%; Spearman(pct_mito, total)=+0.20). A 20% cutoff is
the 10x whole-cell heuristic and simply does not describe sci-RNA-seq3 chemistry.

**Does it bias the benchmark target? Yes, substantially.** I recomputed the delta for 24
random cached conditions from raw cells at 20% vs 50%, same matched-plate control:
median Pearson r between delta@20 and delta@50 = **0.886**; median
MAE(delta@20 − delta@50) / MAE(delta@50 − 0) = **0.55**, i.e. the choice of QC threshold
moves the prediction target by roughly half of its own magnitude. Several conditions
lose >60% of their cells at 20% (e.g. `A549|TAK-901|1000` 45 vs 146 cells).

**Defensible threshold**: 50% (drops 0.68%, flat across dose 0.5–0.7%, near-flat across
lines 0.2–0.9%), or a per-cell-line MAD-based outlier rule (median + 3·MAD, the
scanpy/scater convention), recorded in meta. Either way: **cache rebuild required
(~15 min cold)** — and warranted; the current cache encodes a filter that reshapes the
target and confounds the main regime.

### 1.3 ZERO-DE RATE — NOT AN ISSUE in the computation; CONFIRMED BUG in the report

DE computation (`data.py:253-270`, `459`) is correct:
- two-sided Mann–Whitney U, vectorized over genes, on log-normalized values (rank test,
  so normalization is irrelevant) — direction is right;
- BH-FDR across the 12,871 genes per condition via `statsmodels.multipletests(...,
  method="fdr_bh")`; NaN p (all-tie genes) → 1.0 before correction, correct;
- control cells = `control_idx_by_key[used_key]`, i.e. the matched (cell_line, modal
  plate) vehicle cells, same group used for the delta;
- the synthetic test `test_de_mask_is_not_just_topk_of_delta` is a genuine adversarial
  test (large-noisy delta must NOT be DE, small-consistent must) and passes.

The 45% is biology, not a bug. From the cache: zero-DE rate is **56.7% at 10 nM →
54.5% → 41.7% → 26.7% at 10 µM**, and **K562 62.1% / A549 54.2% / MCF7 18.5%**. The
top-DE conditions at 10 µM are AT9283, CUDC-907, Panobinostat, Givinostat, Dacinostat,
TSA, Abexinostat, Belinostat — HDAC inhibitors, sci-Plex's headline responders. That is
exactly the pattern a correct test should produce. (Caveat: those rates are computed on
the mito-filtered cells; with 37% of cells restored, power goes up and the rate will
drop — another reason to rebuild.)

**Report surfacing — BROKEN.** `report.py:127-155 _de_skip_note` estimates the skip rate
from `no_change`'s `n_conditions` on `pearson_r_de`. But `no_change` is an all-zero
vector so `pearson_r_de` is NaN for EVERY condition → `n_conditions = 0` → the note
prints **"roughly 100.0% ... are skipped"** regardless of the data. Verified on the
synthetic end-to-end run (0 zero-DE conditions in that data; note still says 100%). The
dataset-level number in the Fallback section (`n_conditions_zero_de_genes`) is correct.
Root cause: `aggregate()` computes `n_skipped` but `run.py:198-203` never writes it to
`results.csv`, so the report has no honest per-fold skip count to use.

### 1.4 HALVES GENE-AXIS REMAP — NOT AN ISSUE (remap exists and is correct)

`run.py:110-131`: `union = sorted(halves_gene_idx)`, `pos = searchsorted(union, genes)`
with an explicit equality check that raises on any uncovered gene, then
`control[:, union]`, `de_mask[:, union]` and `pos` are handed to `compute_ceiling` so all
three arrays are indexed on the same local axis. I ran an adversarial probe with a
**shuffled** fold subset (`[17,2,11,7]` against union `[2,5,7,11,13,17,19]`): the code
produces local positions `[5,0,3,2]` and results identical to indexing the halves
directly with those positions. Correct. The shipped test uses a sorted subset, but since
`select_hvgs` always returns sorted indices and every metric is permutation-invariant
across genes, an order bug cannot bite in practice.

One latent hole (LOW): the code re-sorts `meta["halves_gene_idx"]` but the `halves`
columns follow whatever order the list was written in. The real loader writes it sorted
(`_compute_fold_hvg_union` → `np.sort`), so this is fine today; a future loader writing
an unsorted list would silently mis-map. Fix: assert sortedness in `_ceiling_for_fold`
(one line).

---

## 2. New issues

Severity key: CRITICAL = silently wrong numbers or leakage; HIGH = materially misleading
output or unguarded leakage path; MEDIUM = correctness/robustness gap; LOW = hygiene.

### C-1 · CRITICAL · `mae` normalization is sign-inverted on every real fold; no direction guard anywhere
**Where**: `metrics.py:200-216 normalize_score`; `run.py:187-213` (never consults
`METRIC_DIRECTION`); `ceiling.py` (module docstring claims there is "no known analog" of
a split-half correction for mae — there is).
**Failure**: split-half MAE compares two half-pseudobulks, each with 2× the variance of
the full pseudobulk; their difference has 4× → the ceiling MAE ≈ 2× the noise floor of
the actual target. For weak-response conditions (≥45% of this dataset) that is larger
than `no_change`'s MAE. Measured: floor/ceiling = 0.0325/0.0394 (A549), 0.0269/0.0334
(K562), 0.0230/0.0247 (MCF7), 0.0313/0.0365 (random); split-half mae > no_change mae in
86.9 / 93.9 / 86.2 / 86.1 % of test conditions. `normalize_score` divides by a
negative-signed denominator and per_drug_mean_delta (mae 0.0399, worse than no_change's
0.0325) becomes normalized **+1.07 … +7.31**. Anyone reading `results.csv` would
conclude B3 beats the noise ceiling on MAE. **Confirmed in a real `--quick` run's
`results.csv`/`report.md`** (7 folds, 200 genes): ceiling mae 0.038 > no_change mae on
all 7 folds; ridge on held_out_context/A549 has mae 0.0561 vs no_change 0.0358 (much
worse) and is written as normalized **+8.847**; ridge on `random` has mae 0.0267 (better
than no_change 0.0312) and is written as **−0.657**; the report's main table prints
ridge_no_drug mae normalized = **21.159** on held_out_both. `pearson_r` is not inverted on the folds
probed (floor 0.03–0.15 vs ceiling 0.44–0.52) but nothing guarantees it.
**Fix** (two parts, both required):
1. **Guard, non-negotiable**: in `run.py`, after computing `floor_mean`/`ceiling_mean`,
   check `sign(ceiling - floor)` against `METRIC_DIRECTION[metric]`. If inverted, write
   `normalized = NaN`, add a `normalization_valid` bool column, and log loudly. Unit-test
   with an inverted pair. Never emit a normalized number whose scale is flipped.
2. **Correct the ceiling so it is a ceiling.** Options, in order of preference:
   (a) score baselines against a held-out half (`halves[r,1]`) and the ceiling as
   `halves[r,0]` vs `halves[r,1]` — same target noise for model and ceiling, works for
   every metric with no analytic correction, and is exactly the "real replicate" logic
   Arc uses; costs nothing extra since halves are already cached over every fold's genes;
   (b) analytic corrections per metric: Spearman–Brown for `pearson_r`/`pearson_r_de`,
   `mae_ceiling = split_half_mae / 2` under additive Gaussian noise (the docstring in
   `ceiling.py:19-26` is wrong that no analog exists), but nothing clean for
   `direction_acc_de`/`pert_discrimination` — so (b) alone still needs the guard.
   Orchestrator decision; (a) is the one I would take. Either way this is a
   CONTRACT-level change (the "sign-agnostic formula still works because floor/ceiling
   bracket it" sentence in CONTRACT.md is false for mae and must be amended).

### C-2 · CRITICAL · Ceiling computed over all conditions, not the fold's test set — FIXED, see §4
**Where**: `run.py:93-131 _ceiling_for_fold(data, genes, representation)` — no
`test_idx`; `run.py:185`.
**Failure**: denominator not fold-matched (numbers in §0); `n_conditions` for ceiling
rows = 10 × 2233 = 22,330 for every fold; `pert_discrimination` ceiling ranks against
2,233 distractors vs 110–749 for baselines (harder → pessimistic ceiling for small
folds); 2.8 h runtime. The existing test `test_normalized_is_zero_at_floor_and_one_at_ceiling`
cannot detect this because the ceiling row is normalized against itself (tautologically
1.0). The synthetic e2e confirms: `held_out_both/CL0__fold0` has n_test=2 but ceiling
`n_conditions=120` (=4 reps × all 30 conditions).

### H-1 · HIGH · The test suite does not guard HVG selection against leakage
**Where**: `tests/test_data.py:246-267 test_select_hvgs_uses_only_train_rows`.
**Evidence**: I planted `X_train = data.X` (all rows) in `select_hvgs` and ran the full
suite: **64 passed**. The test's only assertions are shape and that passing ALL rows
equals selecting on all rows — it is tautological with respect to the property in its
name. (Planting PCA-on-all-rows in ridge WAS caught by `test_ridge_does_not_see_test_rows`;
that test is real.)
**Fix**: mutate `data.X[test_idx]` to enormous variance after computing
`select_hvgs(train_idx)`, recompute, assert identical; and a run-level test that
`select_hvgs` is only ever called with `fold.train_idx` (spy on it).

### H-2 · HIGH · `assert_no_leakage` trusts the fold's self-declared `meta`
**Where**: `splits.py:270-285`.
**Evidence**: a `held_out_context` fold with cell line A in BOTH train and test and
`meta={}` passes silently; a `held_out_drug` fold with compound x in both and
`held_out_compounds=[]` passes silently. The function only checks the axes the fold
*claims* to hold out. Since `make_folds` writes the meta, the check is effectively
"did make_folds contradict itself". A regression in `make_folds` that forgets to fill
meta would disable the checklist without a single failure.
**Fix**: derive the held-out sets from `fold.regime` + `obs.iloc[test_idx]` (e.g. for
`held_out_context`: `set(obs.cell_line[test]) ∩ set(obs.cell_line[train]) == ∅`), and
cross-check against meta. Add the two silent cases above as failing tests.

### H-3 · HIGH · Batch matching is only half true: every condition spans 2 plates, controls are matched to the modal one only
**Where**: `data.py:400-411`; report text at `report.py:393-398` ("never against a
control from a different plate").
**Evidence**: from the raw obs, `plate` ⟂ `replicate` (plates are rep1-only or
rep2-only), each condition sits on median 2 plates / 2 wells, and **all 2,256 conditions
have <100% of cells on their modal plate; mean 44.3% of a condition's cells are on a
different plate from the control they are subtracted against** (p90: 49.8%). So the
"within-plate" guarantee holds for ~56% of the cells that make up each delta. The DE
test has the same mismatch (treated cells from both plates vs modal-plate controls).
**Fix**: compute per-(condition, plate) pseudobulk and per-plate delta against that
plate's vehicle cells, then average plates (weighted by cell count); do the same for
halves (split within plate) and for the DE test (stratified, or per-plate then combine).
Report the per-condition plate-purity distribution. **Cache rebuild required.**

### H-4 · HIGH · 82,110 A549 cells (39% of A549) are 72 h treatments pooled into 24 h conditions
**Where**: `data.py` cond_id = `cell_line|compound|dose`; DATA_RECON calls this "a small
72h subset".
**Evidence**: `time==72` for 82,110 cells, all A549, 48 compounds, plates 49–52, with
2,084 72 h vehicle cells. **188 conditions — every one of them A549 — mix 24 h and 72 h
cells**, i.e. a quarter of the held-out-context/A549 test set is a time-pooled blend.
The matched control for those conditions is the modal plate's, which is a 24 h or 72 h
plate depending on which replicate had more cells.
**Fix**: either add `time` to the condition key (cleanest; SPEC says dose is identity —
time is the same kind of thing) or drop `time != 24` with a config flag and record the
count in meta + report. **Cache rebuild required.**

### H-5 · HIGH · `direction_acc_de` floor (no_change = 0.000) is below chance and inflates every normalized score
**Where**: `run.py:48-54 FLOOR_BASELINE`, CONTRACT ADDENDUM 1.
**Evidence**: `no_change` scores exactly 0.0 on `direction_acc_de` purely because
`sign(0) ≠ sign(x)` — it is an artifact, not a floor. Real folds: B2 (global mean)
scores 0.53–0.77, B3 0.64–0.69, ceiling ≈0.996. Normalized against 0 the B3 numbers read
0.64–0.69 "of the way to the ceiling"; against the honest no-drug-identity floor (B2) they
read ≈0.24 (A549), ≈0.28 (K562), ≈0.02 (MCF7), and on `random` B3 is *below* B2. The
ADDENDUM 1 rationale for using B2 on `pearson_r` (B1 is degenerate there) applies with
equal force here.
**Fix**: amend ADDENDUM 1: `direction_acc_de` floor = `global_mean_delta` (or 0.5 chance,
but B2 is the consistent choice). `pert_discrimination`'s B1 floor of exactly 0.5 is
genuinely chance and is fine.

### H-6 · HIGH · Nothing written incrementally; no resume
**Where**: `run.py:147-231`.
**Fix**: append each fold's rows to `results.csv` as it completes (open in append mode
with header-once), write `run_meta.json` first, and add `--resume` that skips folds
already present. Cheap; do it alongside C-2.

### M-1 · MEDIUM · `n_skipped` is dropped on the floor; report cannot state the skip rate honestly
**Where**: `run.py:198-213` (writes `n_conditions` only), `report.py:127-155`.
**Fix**: write `n_skipped` (and `n_test`) per row; have `_de_skip_note` read them from a
baseline that is defined (or from any row, since skip counts are baseline-independent).
Also record `n_test_conditions` in results so ceiling rows' `n_conditions` (reps ×
conds) stop looking like condition counts.

### M-2 · MEDIUM · Means reported without variance in the main table and per-cell-line table
**Where**: `report.py:85-98 _regime_table`, `:158-195`. Cells are `normalized (n=…)` with
no std/CI; `_pool_normalized` explicitly has no std. SPEC §11 forbids this ("a few
high-response drugs can carry the average").
**Fix**: normalize per condition (score_i − floor_mean)/(ceiling_mean − floor_mean) and
carry std across conditions, or print raw mean ± std next to each normalized cell; at
minimum report a bootstrap CI over conditions. Also the B3-vs-B4 "winner" column and
Q1 "beats" are declared from point estimates with no uncertainty.

### M-3 · MEDIUM · Silent dose-collapse fallback in B3 and B5
**Where**: `baselines.py:147-149` (tier 1 = same compound, any dose, averaged over
doses), `:378-384`. `fallback_mask` reports only tier 2.
**Evidence**: real data: 6.9% of `random` test conditions and 0.3–0.4% of
`held_out_context` take the dose-collapsed path, unreported. SPEC §11: dose is identity.
**Fix**: write `fallback_tier` counts to results (`fallback_rate_dose_collapsed`), or
make tier 1 count as fallback. Same for `NearestContext`'s compound-only tier.

### M-4 · MEDIUM · Cache key excludes `n_hvg`/`gene_set` although `halves` content depends on them; `gene_set: all` is unusable on real data
**Where**: `config.py:22-34` (comment even contains "wait…"), `data.py:844-866`.
**Evidence**: with the real cache, `gene_set: all` makes `_select_genes` return 12,871
genes and `_ceiling_for_fold` raises "not covered by the cached halves gene union".
Loud, not silent — but the SPEC-required "all genes" option does not work, and the
comment in config.py is now false after ADDENDUM 2.
**Fix**: add `n_hvg`, `gene_set` to `_CACHE_KEY_FIELDS`; for `gene_set: all`, cache
halves over all genes (10×2×2233×12871×4 B = 2.3 GB, affordable) or cap reps and say so.

### M-5 · MEDIUM · Ridge alpha CV folds are random over conditions, not grouped by the regime's held-out axis
**Where**: `baselines.py:238`. Not leakage (train rows only), but in `held_out_context`
the alpha is tuned for within-cell-line generalization and then evaluated on cross-line
transfer. Fix: `GroupKFold` by cell line when ≥3 train lines… with 2 train lines that is
a 2-fold CV; document the limitation instead if not changed.

### L-1 · LOW · `_ceiling_for_fold` re-sorts `halves_gene_idx` without asserting it matches column order (§1.4).
### L-2 · LOW · Ceiling row `normalized` is tautologically 1.0; the test asserting it cannot fail. Replace with a test that plants a wrong ceiling and checks a baseline's normalized value moves.
### L-3 · LOW · `_pool` fills missing std with 0 (`report.py:64`), silently under-dispersing pooled std for n=1 folds.
### L-4 · LOW · 1,029 sklearn/scipy RuntimeWarnings per test run (`invalid value in matmul`, `overflow in vecdot`) — the PCA path on float32 the docstring worries about is still emitting them; investigate rather than suppress.
### L-5 · LOW · No git history (`main` has no commits); nothing is pinned or diffable.

### Verified OK (no issue found)
- **No deep-learning deps**: grep for torch/jax/tensorflow/keras/flax over package, tests,
  lock file: none.
- **`de_mask` provenance**: only `_mannwhitney_de_for_condition`; `metrics.py` never
  ranks deltas. SPEC §11 trap avoided.
- **HVG/PCA/alpha are train-only in the code as written** (`run.py:80-90`,
  `baselines.py:195-203, 238-252`); the halves gene-union superset is not leakage for
  the reason ADDENDUM 2 gives (each fold scores on its own train selection).
- **`assert_no_leakage` runs for every fold** (spied test `test_assert_no_leakage_called_once_per_fold`
  is real; `test_leaky_fold_fails_the_run…` plants an overlap and the run raises).
- **`pert_discrimination` ties**: average-rank; B1's identical zero vectors score exactly
  0.5000 on all real folds probed; cannot be gamed to ~0 on delta.
- **`absolute` labeling**: main table is delta-only; absolute appears only in the
  explicitly-labeled inflation section; `metrics.py`/`ceiling.py` docstrings say
  diagnostic.
- **ADDENDUM 1 NaN policy**: `no_change` pearson_r/pearson_r_de are NaN in csv and
  report (checked synthetic csv); `floor_source` recorded per row and correct.
- **Dose in identity**: cond_id and B3/B5 keys include dose; only the tier-1 fallback
  (M-3) collapses it.
- **Batch metadata present and used** (with the H-3 caveat).
- **`held_out_drug` grouping by target** works on real data (`target_annotation_available=True`,
  0% unknown targets, 34 compounds held out in fold 0).
- **Determinism**: VERIFIED on the real cache — two independent `--quick` runs
  (5m38s and 5m33s, pre-fix code) produced byte-identical `results.csv`
  (md5 `07b58b2ae131c21a5a35533713f8b1a8`).

---

## 3. Prioritized fix plan (correctness risk × effort)

| # | fix | files | effort | rebuild? |
|---|---|---|---|---|
| 1 | **Direction guard**: normalized=NaN + `normalization_valid` column + loud log when `sign(ceiling−floor)` contradicts `METRIC_DIRECTION`; unit test with an inverted pair (C-1 part 1) | run.py, metrics.py, tests | 1 h | no |
| 2 | **Ceiling on fold test rows only** (C-2) — DONE in §4; keep | run.py, tests/test_run.py | done | no |
| 3 | **Kill PID 2256; add incremental per-fold write + `--resume`** (H-6) | run.py | 1 h | no |
| 4 | **Fix the ceiling so it is a ceiling** (C-1 part 2): decide (a) score-vs-half or (b) analytic; amend CONTRACT | ceiling.py, run.py, CONTRACT.md | 3–4 h | no (halves already cached) |
| 5 | **`direction_acc_de` floor → global_mean_delta**; amend ADDENDUM 1 (H-5) | run.py, CONTRACT.md, report.py | 30 min | no |
| 6 | **Write `n_skipped`/`n_test` to csv; fix `_de_skip_note`** (M-1, §1.3) | run.py, report.py | 1 h | no |
| 7 | **Real HVG-leak test + spy that `select_hvgs` only sees `train_idx`** (H-1) | tests | 1 h | no |
| 8 | **Regime-derived leakage check** (H-2) + the two silent cases as tests | splits.py, tests | 1–2 h | no |
| 9 | **QC: `max_pct_mito` 20 → 50 (or per-line MAD)**, record distribution quantiles + per-line/per-dose drop rates in meta and report (§1.2) | config, data.py, report.py | 1 h + **rebuild ~15 min** | **YES** |
| 10 | **Time in condition identity or drop 72 h** (H-4) | data.py, report | 1–2 h | **YES** (same rebuild as 9) |
| 11 | **Per-plate delta / halves / DE** (H-3) | data.py | 4–6 h | **YES** (same rebuild) |
| 12 | Variance on every reported mean; CI on winner claims (M-2) | report.py | 2 h | no |
| 13 | Report tier-1 dose-collapse fallback (M-3) | baselines.py, run.py, report | 1 h | no |
| 14 | Cache key incl. `n_hvg`/`gene_set`; make `gene_set: all` work (M-4) | config.py, data.py | 1 h | rebuild only if used |
| 15 | Vectorize `_pert_discrimination` (cdist + rankdata axis=1) | metrics.py | 30 min | no |
| 16 | L-1…L-5 | various | 1 h | no |

Do 1–3 before anything is re-run. Do 9–11 together as ONE rebuild (they all touch
`data.py`'s condition/control construction) — three separate rebuilds would waste 45 min.
Items 4 and 5 change what `normalized` means and must be reflected in CONTRACT.md before
the next sweep so nobody compares numbers across the change.

---

## 4. The one fix I applied — READ THIS

**File**: `perturb_bench/run.py`, function `_ceiling_for_fold` (signature changed to
`_ceiling_for_fold(data, test_idx, genes, representation)`) and its call site in `run()`.
**Tests updated**: `tests/test_run.py::test_ceiling_for_fold_maps_full_axis_genes_to_local_halves_positions`
and `::test_ceiling_for_fold_raises_loudly_on_genes_outside_the_union` (pass the new
argument; the first now also asserts that a test-subset gives a different answer from
all-rows so the regression is detectable).
**What it does**: slices `halves`, `control`, `de_mask` to `fold.test_idx` rows before
scoring the ceiling, so the ceiling is over exactly the conditions each baseline is
scored on (and `pert_discrimination` ranks against the same distractor pool).
**Why I did it rather than just planning it**: it is a CRITICAL correctness bug (wrong
denominator for every normalized number, by up to 50% on MCF7 mae), the correct behavior
is unambiguous from SPEC §6/§7 and CONTRACT ("same output shape as a baseline's
scores"), the change is ~10 lines, and it is also the entire runtime problem. It does NOT
fix C-1 — normalized mae is still inverted until fix #1/#4 land.
**Status**: full test suite passes after the change (64 passed). The updated remap
test now also asserts that a strict test-row subset gives the sliced-halves answer and
that ceiling `n` equals `n_test × n_rep`, so a regression to all-rows fails.
Behavioral change: any `results.csv` produced before this edit has ceiling rows over all
conditions; do not compare across it.

---

## 5. Could not verify / caveats

- **Effect of fixes 9–11 on the actual benchmark numbers** (would need the 15-min
  rebuild; I did not rebuild, per instructions). The 24-condition delta@20-vs-@50 probe
  is a sample, not the full effect.
- **What `plate` physically is** in the scPerturb harmonization (PCR plate vs culture
  plate). The structure I measured (plates are replicate-pure, each condition on exactly
  one well per replicate-plate) is consistent with plate = culture/hash plate per
  replicate, which is the right batch unit — but I could not confirm from metadata alone.
- **Whether B2-as-floor for `pearson_r` is ever inverted on `held_out_drug`/`held_out_both`
  folds**: I probed `random` and the three `held_out_context` folds only (floor 0.03–0.15
  vs ceiling 0.44–0.52, not inverted). Fix #1 makes this moot.
- **The PCA `RuntimeWarning`s** (L-4): I did not trace whether components are ever
  actually corrupted on the real data.
- **pertpy/scPerturb version pin**: DATA_RECON is honest that no version string exists
  in the h5ad; I did not find one either.

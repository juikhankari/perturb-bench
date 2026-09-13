# sciplex3 data recon

## Source and how it was obtained

`pertpy.data.srivatsan_2020_sciplex3()` was run and downloaded successfully (no
fallback needed). Download took ~124s, landed at
`cache/sciplex3_raw.h5ad` (12.2 GB on disk). `pertpy` also keeps its own copy under
`./data/srivatsan_2020_sciplex3.h5ad` (2.5 GB, presumably a different encoding/
compression than our re-saved copy) -- that copy was deleted to reclaim disk space
since our own cache (`cache/sciplex3_raw.h5ad`) is self-sufficient.

Inspecting the HDF5 structure directly shows this is actually the **scPerturb
harmonized release** of sciplex3 (it carries scPerturb's standardized obs schema:
`cell_line`, `perturbation`, `dose_value`/`dose_unit`, `pathway`, `pathway_level_1`,
`pathway_level_2`, `target`, `plate`, `replicate`, `well`, `time`, `organism`,
`tissue_type`, `cancer`, `disease`) rather than the raw GEO deposit's native schema --
pertpy's loader fetches this harmonized h5ad under the hood. This is consistent with
SPEC §3's "Either is fine -- DOCUMENT WHICH AND PIN THE VERSION": we are effectively
on the scPerturb-harmonized sciplex3, fetched via pertpy's convenience wrapper.
`pertpy` version installed: whatever ships with the pinned `pertpy==1.3.0` in this
venv; no separate version pin exists for the harmonized h5ad itself (scPerturb doesn't
expose one in the file) -- the provenance string recorded in `meta["source"]` is
`pertpy.data.srivatsan_2020_sciplex3() [freshly downloaded]` (or `cached_h5ad:<path>`
on subsequent loads), which is the best available pin.

## Shape

`adata.shape = (799317, 110983)` -- 799,317 cells x 110,983 genes/features.

**Important**: 110,983 is NOT a usable "all genes" feature count for this harness on
this machine (24 GB free disk, ~26 GB RAM at the time of this work). See "Gene
filtering" below.

## obs columns (full list)

```
cancer, cell_line, celltype, disease, dose_unit, dose_value, ncounts, organism,
pathway, pathway_level_1, pathway_level_2, perturbation, perturbation_type, plate,
replicate, target, time, tissue_type, well
```

### Column resolution (what the loader auto-detects and uses)

| Role | Column used | Why |
|---|---|---|
| cell line | `cell_line` | 3 categories: `A549`, `K562`, `MCF7` -- exactly matches SPEC |
| compound | `perturbation` | 189 categories: 188 named compounds + `control` |
| control label | `perturbation == "control"` | literal string `"control"`, lowercase |
| dose | `dose_value` | float, unit in `dose_unit` (always `"nM"`, 1 category) |
| **batch/plate** | `plate` | **52 categories** (`plate1`...`plate52`-ish), nested under cell_line: A549 has 20 plates, K562 16, MCF7 17. Control (vehicle) cells exist in every (cell_line, plate) combination with 146-36,522 cells each (median ~280) -- plenty for matched-control aggregation. |
| target/pathway | `target` (87 categories) and `pathway`/`pathway_level_1`/`pathway_level_2` (21/17/55 categories) | all present; `target` used for `obs.target` and for `held_out_drug` compound grouping in splits.py |

Columns NOT used: `celltype` (redundant/coarser version of disease context, has a
`None` category), `cancer`/`disease`/`tissue_type`/`organism` (constant or near-constant,
not useful for splitting), `ncounts` (redundant with a freshly computed per-cell total),
`time` (mostly 24h, a few 72h -- not part of the condition key per CONTRACT's
`cond_id = cell_line|compound|dose`; a possible follow-up but out of scope here),
`well`/`replicate` (finer/coarser than `plate`; `plate` was chosen as the batch
variable per the table above).

## THE BATCH METADATA ANSWER (the single most important recon finding)

**YES -- real, usable batch/plate metadata exists.** The `plate` column gives 52
distinct physical plates, with control (vehicle) cells present on every single
(cell_line, plate) combination the loader encounters (53 combinations observed; every
treated condition's modal plate has a matching vehicle-cell group). This means the
matched-control-per-batch requirement in SPEC §3/§11 ("MATCH CONTROLS WITHIN
BATCH/PLATE... If no batch column exists, SAY SO EXPLICITLY") is fully satisfiable
for sciplex3: `meta["batch_metadata_available"] = True`, and the fallback path (no
batch -> per-cell_line-only controls) is NOT exercised on the real run. The fallback
path is implemented and unit-tested (`tests/test_data.py::test_no_batch_column_falls_back_and_flags`)
for datasets that lack batch info (e.g. a future "tahoe" loader), but sciplex3 itself
does not need it.

## X: raw counts or normalized?

**Raw integer counts.** `f['X']['data'].dtype == int64`, sampled values are small
positive integers (e.g. `[3 2 4 1 1 1 7 1 3 1 2 1 1 1 1 2 1 1 1 1]`), consistent with
UMI counts. `.raw` / `.layers` are empty (`layers: []`, no separate raw slot -- `X`
itself is the raw counts). `normalize_total` (to the dataset's per-cell median total
count, computed over the FULL un-filtered gene set for correctness) then `log1p` are
applied by the loader; this is detected programmatically (`ensure_lognorm`/the backed
streaming equivalent) rather than assumed, and documented in
`meta["normalization"]`.

`X` is stored CSR (`X/data`, `X/indices`, `X/indptr` groups), 1,007,419,688 nonzero
entries total.

## Mitochondrial genes

`var` index is gene **symbol** (not Ensembl ID -- there's a separate `var/ensembl_id`
column, but `var_names` is already `gene_symbol`, e.g. `TSPAN6, TNMD, DPM1, ...`).
74 genes match the `MT-` prefix (`MT-ND6, MT-CO2, MT-CYB, ...`, including tRNA genes
like `MT-TL1`), so the mito-% QC filter IS applicable here (unlike some datasets where
it must be skipped) -- `meta["qc"]["mito_filter_applied"] = True`.

## Gene filtering (disk/memory-driven, NOT HVG selection, NOT leakage)

110,983 "genes" includes ~33,182 features with **zero** nonzero entries across all
799,317 cells and tens of thousands more detected in only a handful of cells --
consistent with this being a comprehensive Ensembl/GENCODE-based feature list rather
than a curated expressed-gene panel. Caching `(n_cond, n_genes)` arrays and especially
the `(n_rep, 2, n_cond, n_genes)` split-half `halves` array at the full 110,983-gene
width would require many GB to low-TB of disk -- this machine had ~24-26 GB free disk
and ~26 GB RAM total at the time of this run, both blown by the literal full-gene
width.

**Fix applied**: a dataset-global detection filter keeps only genes detected
(count > 0) in at least 1% of all cells (`GENE_DETECTION_MIN_FRAC = 0.01` in
`perturb_bench/data.py`), equivalently >= ~7,993 of 799,317 cells (floor at 3 cells
for tiny datasets). This reduces 110,983 -> **12,871 genes** while retaining **97.3%**
of all nonzero count entries in the dataset. This is a detection-presence filter
computed over the WHOLE dataset (not per-fold, not using condition labels or effect
sizes) -- the same class of operation as the mitochondrial-content QC filter, not a
form of HVG selection and not leakage. It is recorded in
`meta["gene_detection_filter"]` and is applied BEFORE train/test splits exist.

**`halves` gets a second, much smaller cut on top of this** (see the module docstring
and inline comments in `perturb_bench/data.py`): rather than caching split-half deltas
over all 12,871 detection-filtered genes, `halves` is cached only over the UNION of
the train-only HVG gene indices that `select_hvgs` would pick across every fold of
every split regime (random / held_out_context / held_out_drug / held_out_both). This
was a correction to the original CONTRACT (which specified all-genes `halves` and
would have required ~14+ GB even at 30k genes, let alone 12,871 x full rep count).
Each fold still only ever computes/scores on genes selected from its OWN training
rows -- the cache merely happens to be a superset covering every fold's selection, so
no fold's HVG choice is informed by its own held-out data. `meta["halves_gene_idx"]`
records the exact positional indices (into the 12,871-gene axis) covered, and
`load_pseudobulk` asserts every fold's selection is a subset of it at load time,
raising if the splits logic ever changes in a way the cache doesn't cover.

## Conditions (treated, before dropping for min-cell-count)

3 cell lines x 188 compounds x 4 doses (0, 10, 100, 1000, 10000 nM observed as
distinct `dose_value`s excluding the 0/NaN used by controls -- 4 nonzero dose levels)
= up to 2,256 (cell_line, compound, dose) treated conditions. Per-condition cell
counts (pre-QC) range from 14 to 860 (median 264); only **8 of 2,256** fall below the
`min_cells_per_condition=30` threshold on raw cell counts.

Total control (vehicle) cells: 54,100, spread across 53 (cell_line, plate)
combinations, 146-36,522 cells each (the outlier-large group is one specific plate;
median is ~280).

## ACTUAL NUMBERS FROM THE REAL END-TO-END RUN

(`cache/run_real_load.py`, cold run against `configs/default.yaml`, seed 0)

- **QC**: 799,317 cells before QC -> 779 dropped for `min_genes<200` -> **297,404
  additionally dropped for `pct_mito>20%`** -> 501,134 cells survive QC (62.7% of
  input). The mito filter is by far the dominant QC cut here -- worth knowing before
  trusting downstream cell counts. `max_pct_mito=20.0` is the SPEC-suggested default;
  it was not tuned for this dataset.
- **Conditions after QC + pseudobulking**: 2,233 retained (23 of 2,256 dropped for
  `n_cells < 30` post-QC, vs. the 8 estimated pre-QC above -- mito filtering pushes a
  few more conditions below threshold). Zero conditions needed the per-cell-line
  control fallback (`n_conditions_fallback_control = 0`) -- every retained condition's
  modal plate had its own matched vehicle-cell group.
- **Genes after detection filter**: 12,871 (from 110,983), as estimated in recon.
- **Cells per cell line (post-QC, summed over retained conditions)**: A549 126,811 /
  K562 142,184 / MCF7 203,257 cells; 740 / 749 / 744 conditions respectively.
- **DE genes**: **44.96% of conditions (1,004 of 2,233) have ZERO genes passing
  BH-FDR<0.05** single-cell Wilcoxon vs matched control. This is a real and notable
  finding, not a bug -- it says a large fraction of (cell_line, compound, dose)
  combinations in sciplex3 produce a transcriptional response too weak/noisy to
  detect at the single-cell level with this test and correction, even though a
  pseudobulk delta is always nonzero. This number should be surfaced prominently in
  the final report: the `pearson_r_de` / `direction_acc_de` metrics are undefined
  (skipped) for nearly half of all conditions.
- **halves gene union**: 2,984 genes (union of per-fold, train-only top-2000-HVG
  selections across every fold of all 4 regimes) -- comfortably smaller than the
  12,871-gene "all genes" axis, confirming the gene-union fix was worth it.
  `halves` array: shape (10, 2, 2233, 2984) float32 = 533 MB.
- **Cache size on disk**: 866 MB total (`cache/pseudobulk_sciplex3_<key>/`), well
  within the ~26 GB disk budget.
- **Wall-clock**: cold load (full pipeline: streaming QC pass, 2,233 conditions'
  pseudobulk+DE test, fold-HVG-union computation, halves computation) = **884s
  (~14.7 min)**. Warm (cached) load = **0.54s**. Both measured on this machine
  (25.7 GB RAM, single run, no parallelism) -- the 884s cold cost is a ONE-TIME
  pipeline cost, not part of the "<10 min full sweep" target in SPEC §9, which refers
  to the baseline/metric sweep running against a warm cache.

## Known limitations / anything not fully honoured

- The gene-detection filter (1% of cells) and the fold-HVG-union restriction on
  `halves` are both pragmatic, disk/memory-driven additions beyond SPEC's literal
  text ("cache stores ALL genes"). They are documented here and in code comments
  precisely because they were NOT part of the original plan -- flagging per the
  task's "be blunt about anything you faked or skipped" instruction. Nothing about
  condition-level pseudobulk, delta, or DE-mask logic is affected: those three use the
  full 12,871-gene, detection-filtered axis, matching CONTRACT's `(n_cond, n_genes)`
  shape for `X`/`control`/`delta`/`de_mask`. Only `halves` (used solely for the noise
  ceiling) is gene-subsetted.
- `time` (24h vs 72h) is present in the data but not folded into `cond_id` per
  CONTRACT's fixed `cell_line|compound|dose` key -- almost all cells are 24h; a
  small 72h subset exists and is currently pooled in with 24h cells of the same
  (cell_line, compound, dose) if both exist, which would slightly blur the pseudobulk
  for any such condition. Not checked/quantified further; flagged as a known gap.

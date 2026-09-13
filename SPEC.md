# Perturbation Response Benchmark Harness — Specification

## 0. What this is
An evaluation harness + deliberately simple baselines for predicting transcriptional
responses to drug perturbation in cellular contexts the model has not seen.
This is NOT a request to build a novel model. DO NOT build a neural network.
The deliverable is measurement infrastructure plus simple baselines.
If you find yourself wanting to add a deep learning component, STOP. Out of scope.

## 1. Why (determines design decisions)
Published work here has a reproducibility problem. Ahlmann-Eltze/Huber/Anders
(Nat Methods 2025) found no foundation model beat simple baselines; on one task none
beat "predict no change". Arc's 2025 Virtual Cell Challenge: on MAE nearly every
submission did worse than predicting the average cell.
The common failure is not modeling, it is EVALUATION. Metrics on absolute expression
are dominated by baseline expression. Splits leak. Nobody reports a noise ceiling.
THEREFORE: the harness is the product. Its job is to make it impossible to fool
ourselves later.

## 2. Scope
IN: data loading + pseudobulk; 4 split regimes; 5 baselines; noise ceiling;
metric suite on deltas with an absolute-expression comparison demonstrating inflation;
normalized floor-to-ceiling scoring; results table + written report.
OUT (do not build): any trained neural model; single-cell-resolution prediction;
drug structure encoders / target-pharmacology features; hyperparameter search beyond
ridge alpha.

## 3. Data
Primary: sciplex3 (Srivatsan 2020). ~188 compounds x 3 cell lines (A549,K562,MCF7) x 4 doses.
Load via pertpy `pertpy.data.srivatsan_2020_sciplex3()` or the scPerturb harmonized
release. Either is fine — DOCUMENT WHICH AND PIN THE VERSION.
Design the loader behind an interface so Tahoe can be added later without touching
the rest of the harness.

Preprocessing:
1. Filter low-quality cells: standard QC (min genes, max mitochondrial %).
2. Normalize total counts per cell, then log1p.
3. Pseudobulk: mean profile per (cell_line, compound, dose). Record cell count per
   condition, drop conditions below a threshold (suggest 30 cells).
4. Vehicle controls aggregated per (cell_line, batch), used as context baseline X_c.

Delta: delta[c,d,dose] = pseudobulk[c,d,dose] - control[c]
MATCH CONTROLS WITHIN BATCH/PLATE. If treated cells and their control come from
different plates the delta is contaminated with batch effect. Check what batch metadata
sciplex3 exposes and use it. If no batch column exists, SAY SO EXPLICITLY IN THE REPORT
rather than silently ignoring it.

Gene selection: HVGs if desired, but FIT THE SELECTION ON TRAINING CONDITIONS ONLY.
Any gene filtering/scaling/normalization parameter estimated using held-out data is
leakage. Suggest 2000 HVGs; also support "all genes" as a config option.

## 4. Splits
Defined at CONDITION level = (cell_line, compound, dose).
- random: random 20% of conditions. Sanity check, easiest, numbers mean little.
- held_out_context: all conditions for one cell line. Transfer to unseen context. THE MAIN ONE.
- held_out_drug: all conditions for a set of compounds. Transfer to unseen intervention.
- held_out_both: unseen cell line x unseen compound. The real target. Hardest.

held_out_context: iterate over all 3 cell lines as held-out, report each separately AND
the mean. Three contexts is few — do not over-interpret; note this limitation in report.
held_out_drug: group compounds by target/pathway annotation if available so a held-out
compound has no near-identical sibling in training. If unavailable, hold out random
compounds and FLAG IT.

Leakage checklist (ASSERT PROGRAMMATICALLY):
- No held-out cell line appears in any training condition
- No held-out compound appears in any training condition
- HVG selection, scaling params, ridge fitting use training conditions only

## 5. Baselines
Each predicts a delta vector for a held-out condition. All run under all 4 regimes.
- B1 no_change: pred_delta = 0. Zero params. Detects metric inflation. If a metric ranks
  this highly, that metric measures baseline expression, not drug effect.
- B2 global_mean_delta: mean over all training conditions of delta. One vector for every
  drug. Measures how much of response is generic stress/toxicity signature.
- B3 per_drug_mean_delta: pred[c*,d] = mean over training cell lines c of delta[c,d].
  Knows drug, ignores context. THE KEY ADVERSARY. If a context-aware model cannot beat it
  on held_out_context then context does not meaningfully modulate response in this data
  and the core research hypothesis is in trouble. REPORT THIS COMPARISON PROMINENTLY.
  Undefined when drug unseen (held_out_drug, held_out_both): fall back to B2 and REPORT
  THE FALLBACK RATE.
- B4 ridge: pred_delta = W @ [control_expression[c*] ; onehot(drug) ; log_dose].
  First baseline that can express context-dependence (control expression is an input).
  Tests whether that dependence is linear. Tune alpha by CV WITHIN THE TRAINING SPLIT ONLY.
  Undefined for unseen drugs with one-hot. For held_out_drug either drop B4 or substitute
  a drug-agnostic variant and LABEL IT CLEARLY. Do not silently change the encoding.
- B5 nearest_context (optional, cheap, informative): find training cell line whose control
  profile is most similar to the held-out one; copy its delta for that drug. Tests whether
  naive context similarity suffices.

## 6. Noise ceiling — MANDATORY, NOT OPTIONAL
Without it a correlation of 0.4 is uninterpretable.
Split-half: for each condition with enough cells, randomly partition cells into two halves,
compute a pseudobulk delta for each, score one against the other using the FULL metric
suite. Average over conditions and over several random partitions (suggest 10).
This estimates the best achievable score given measurement noise.

## 7. Metrics
Compute EVERY metric TWICE: once on deltas, once on absolute post-treatment expression.
Report both. The absolute column exists to DEMONSTRATE THE INFLATION PROBLEM and must be
labeled as such in the report — a diagnostic, not a performance measure.
Per held-out condition, then averaged:
1. Pearson r between predicted and true delta across genes.
2. MAE on delta.
3. Pearson r on DE genes only. Define DE genes from the TRUE data using a significance
   test on treated-vs-control at the SINGLE-CELL level (Wilcoxon or t-test with multiple
   testing correction), NOT by taking top-k largest true deltas — the latter is a softer
   form of leakage. If no genes pass, skip the condition and record how often that happens.
4. Direction accuracy on DE genes: fraction where sign(pred)==sign(true).
5. Perturbation discrimination: given a predicted profile, rank all true held-out profiles
   by distance to it; report normalized rank of the correct one. Most directly asks "did
   you predict THIS drug's effect rather than a generic effect". Arc uses a version of this
   and it is where methods actually separate.

Normalized scoring:
normalized = (model_score - floor_score) / (ceiling_score - floor_score)
floor = no_change baseline (or context-mean baseline, whichever more appropriate per
metric); ceiling = split-half noise ceiling from §6. This matches the convention Arc
adopted for the 2026 Virtual Cell Challenge (each metric scaled between the cell context
mean and a real replicate experiment), keeping our numbers comparable to a live external
leaderboard.

## 8. Outputs
1. results.csv — long format, one row per (split, fold, baseline, metric, representation)
   where representation is delta or absolute. Include ceiling and floor as pseudo-baselines.
2. report.md — auto-generated. MUST contain:
   - main table: baselines x splits on normalized delta metrics
   - the absolute-vs-delta comparison, with a sentence NAMING the inflation
   - the B3-vs-B4 gap on held_out_context, called out explicitly
   - fallback rates, dropped conditions, any missing batch metadata
3. Deterministic: seed everything, pin versions, single command reproduces the full table.

## 9. Implementation notes
Python. scanpy/anndata for data, scikit-learn for ridge, pertpy for loading.
NO DEEP LEARNING DEPENDENCIES.
Structure: data.py, splits.py, baselines.py, metrics.py, run.py. Baselines behind a common
interface (fit(train)->None, predict(conditions)->array) so a real model can be dropped in
later without changing anything else.
Cache pseudobulk matrices to disk; recomputation from raw single-cell data every run makes
iteration painful.
Runtime target: full sweep under ten minutes on a laptop. If slower, the dataset or gene
set is too big for this stage.

## 10. Definition of done
Single command produces results.csv and report.md, and the report answers:
1. Does any baseline beat no_change on delta metrics, and by how much relative to the
   noise ceiling?
2. How much worse is held_out_context than random? (size of the generalization problem)
3. Does ridge beat per_drug_mean_delta on held_out_context? (is context-dependence
   learnable at all, linearly?)
4. How much does scoring on absolute expression inflate the numbers?
Question 3 determines whether the broader research programme is worth pursuing. IF THE
ANSWER IS NO, THAT IS A REAL FINDING and should be reported plainly rather than worked around.

## 11. Known traps (DO NOT FALL INTO THESE)
- Scoring on absolute expression and reporting it as performance
- Selecting HVGs or fitting normalization on the full dataset before splitting
- Comparing treated cells to controls from a different plate
- Defining DE genes by largest true delta, then scoring on those genes
- Reporting a mean across conditions without reporting variance — a few high-response
  drugs can carry the average
- Treating dose as a nuisance variable; it is part of the condition identity
- Declaring success on random splits

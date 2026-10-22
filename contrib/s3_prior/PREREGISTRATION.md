# S3 preregistration (dev study): hu.MAP 3.0 high-confidence prior

Committed **before** `contrib/s3_prior/explore_s3.py` is first run on real data. The run records the code commit, so
anyone can check that this file's commit is older. This is a **dev** study on the three used datasets (Petrosius 2023,
Leduc 2022, SCoPE2), which `docs/SLOTS.md` makes dev-only. A positive result is a proposal for a confirmation on an
untouched dataset (Amit's decision), never a confirmed claim.

## Prior exposure (disclosed)
- Amendment 4 (`results/phase4/dev/dev_ext.json`, same dev seeds) already tested hu.MAP at **all** confidence levels
  with E1/E6: it was ≈ Complex Portal, with no gain over kNN or low-rank. Those numbers were seen before this file
  was written.
- Our own descriptive checks (session 1, all data, no masks): partner-correlation signal of hu.MAP ≤ 3 is above
  its shuffled nulls. No RMSE of `humap3_high` under any mask has been computed before this commit.
- The cut-off ≤ 3 is our choice (the hu.MAP paper does not recommend one; levels 1–3 beat BioPlex3 on precision and
  recall). It was chosen in session 1, before the Amendment-4 numbers were known to us.

## Question
Does a high-precision **machine-learned** complex graph (`humap3_high`) give better side information for matrix
completion than strong data-only baselines **and** than the expert-curated Complex Portal graph?

## Design (mirrors `benchmarks/explore_ext.py`)
- **Seeds:** dev seeds 100, 101, 102. The subsample seed = the mask seed.
- **Targets and regimes:** Petrosius (all 215 cells; and 28 = 7 per cell type), Leduc 2022 and SCoPE2 (200 and 30
  cells, equal numbers per cell type), drawn with `np.random.default_rng(seed)` exactly as in `explore_ext.py`.
- **Masks:** `run.make_mask(X, eligible, kind, seed)`, with kind MCAR or MNAR; eligible = detected in ≥ 50% of the
  subsample's cells; 10% of observed eligible entries.
- **Graphs (built once per target on its features):** `humap3_high`, `humap3_union`, core `humap3`, `complexportal`,
  `corum`; 5 degree-preserving nulls (seeds 0–4) of each of the three hu.MAP graphs; per subsample, a data graph
  `phase2.data_graph(Xm, names, n_edges = that prior's edge count)` for each hu.MAP graph.
- **Methods:** base K = `phase2.loo_knn(Xm)`; base LR = `phase2.grmf(Xm, None, names, lam_g=0.0, full=True)`.
  **E1** = `phase2.residual_on_knn(Xm, G, names, K=K)` (primary); **E6** = the same with `K=LR` (secondary).
  kNN = `_fill(Xm, K)`, low-rank = `_fill(Xm, LR)`.

## Statistic and pass rule (identical to `benchmarks/confirm_v5.py`)
ΔRMSE = RMSE(prior arm) − RMSE(comparator) on the stratum's masked entries; 95% CI from `run.boot_rmse_diff`
(1000 paired resamples, seed 17), pooled over the 3 seeds. For the null comparator, the reference is the mean over
the 5 nulls. **Pass** = CI upper < −`SIG_TOL` (1e-9) **and** ΔRMSE ≤ −1% of the comparator's RMSE (SESOI).

## Primary verdict (one, fixed now)
Arm: `humap3_high` + **E1**. Stratum: **all** masked entries (the in_complex stratum is defined by CORUM / Complex
Portal membership only, so it is not neutral between priors). Regime: **low n** (Petrosius 28, Leduc 30, SCoPE2 30).
Masks: MCAR and MNAR. Comparators (5): kNN; low-rank; E1 + matched data graph; E1 + own nulls (mean of 5);
E1 + Complex Portal. 3 targets × 2 masks × 5 = **30 checks**.
**"Qualifies (dev)"** only if all 30 pass; otherwise **"does not qualify"**.

## Expected outcome (written before running)
At 30 cells on Leduc and SCoPE2, Amendment 4 shows E1 with any edge prior within ±0.2% of every comparator. So
**"does not qualify" is the most likely result.** Petrosius is the only dataset with a visible prior effect, so its
per-dataset verdict is the main secondary. Low-rank is 18–40% worse than kNN at low n, so its 6 checks are
near-automatic; the informative comparators are kNN, the data graph, the nulls and Complex Portal.

## Secondary (reported, never used to claim success)
- E6, with the same comparators (its data graph, nulls and CP all under E6).
- The 200/215-cell regime (where low-rank can beat kNN, so E1 may fail because of its base).
- Strata in_complex, not_in_complex, and a neutral one (entries whose protein has ≥ 1 neighbour in `humap3_high`
  **or** Complex Portal).
- Per-dataset verdicts (10 checks each).
- A per-seed sign check: ΔRMSE has the same sign in each of the 3 seeds.
- `humap3_union` and core `humap3` with the same comparators; CORUM rows.

## Matched diagnostic (descriptive, phase-1 imputer)
Petrosius 215 cells, seeds 100–102, MCAR and MNAR, `methods.impute_with_prior`. `humap3_high` vs Complex Portal and
vs CORUM on entries that **both** priors impute from partners (`layers["imputed_by_prior"]`, same seed and mask).
Verdict per pair: *curated better* / *hu.MAP better* (CI upper < −1e-9 on both masks, for the respective difference)
/ *inconclusive* (anything else, including masks that disagree or no matched entries). Also reported: the share of
masked entries imputed from partners vs kNN fallback, per prior and stratum.

## Caveats
CIs are optimistic (the bootstrap ignores seed / cell / protein clustering). No multiplicity correction across the
secondaries. hu.MAP was trained with Complex Portal labels, so it is not independent of curation. The datasets are
dev-only.

## Output
`contrib/s3_prior/output/s3_dev.json`, written by `python contrib/s3_prior/explore_s3.py --out contrib/s3_prior/output`.
The script refuses to run if `contrib/s3_prior/` has uncommitted changes, and records the commit, the settings and
the runtime. `--quick` (seed 100, low n only) is a smoke test and never gives a verdict.

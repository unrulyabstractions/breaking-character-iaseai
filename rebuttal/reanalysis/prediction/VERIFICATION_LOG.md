# Verification log: reanalysis/prediction

## 2026-11-12, prediction of M, U and G from the readings (reviewer AE2Q, Q3)

1. **Inputs (readings).** predict.py recomputes the six readings and the drift size of all 56 organisms
   (34 pool, 22 outside) through ../core/core.py and stops if any reading is more than 1e-6 from
   data/readings.csv. predict.json `inputs.max_diff_vs_readings_csv` shows at most 4.98e-07, which is
   the CSV's rounding. verify.py separately compares inputs.csv with readings.csv and measures.csv, and
   the pool split and tunings with the paper's definition. VERIFIED.
2. **Inputs (targets).** check_measures.py rebuilds M, B and C for all 56 organisms from
   data/raw/runs/<run>/em_organism.csv. I opened check_measures.json and confirmed:
   - max |dM| = 0.00125, |dB| = 0, |dC| = 4.8e-05;
   - no organism is off by more than 0.005 and no run file is missing;
   - no organism changes side of M = 0.05.
   45 of 56 have identical counts; the other 11 differ by a few answers. VERIFIED.
3. **Held-out predictions.** verify.py shares no code with predict.py or core.py. It recomputes every
   row of predictions.csv (7,296 rows: every scheme, rule, score and target) by an explicit refit in
   each fold, using pandas z-scores and numpy lstsq. The largest |diff| is 2.1e-14. VERIFIED.
4. **Observed metrics.** verify.py recomputes all 4,392 observed metric values in predict.json from
   those predictions with scipy.stats.spearmanr and a pairwise AUROC. The metrics are Spearman,
   within-family Spearman, RMSE, null RMSE, R^2 and AUROC. The largest |diff| is 8.6e-12. VERIFIED.
5. **Permutation p-values.** verify.py reran two cells by explicit refits under 2,000 fresh shuffles:
   - LOFO, rule W, composite, M: 0.0005 by refit (the floor for 2,000 shuffles) vs 0.0001 in the JSON;
   - OUT, rule W, composite, M, LoRA subset: 0.0495 vs 0.0550.
   Both agree within Monte Carlo error. VERIFIED for these two cells only. The other permutation
   p-values rest on the same code path (yhat = P y), and step 3 checked P's predictions on the observed
   data. They were not individually recomputed by refits: UNVERIFIED individually.
6. **Bootstrap intervals** (`spearman_ci95`, `auroc_ci95`). Not recomputed independently. UNVERIFIED.
7. **Held-out predictions looked at by eye.** I printed the composite predictions for LOFO W, LOFO R,
   OUT W, OUT R and OUT Wstrict.
   - The LOFO W AUROC of 0.996 comes from one inverted pair: Qwen3-32B reward-hacks (control) at
     0.098 above Qwen2.5-32B bad-user-medical at 0.085.
   - The OUT Wstrict RMSE above 0.4 comes from leave-self-out z-scores in tight families: Llama
     insecure-code at +25.6 and Gemma bad-medical (LoRA) at -35.5 on the shift deviation. VERIFIED.
8. **No writes outside the allowed folders.** I ran `find -newer predict.py` over the replication repo.
   My files are only in this folder; the other new files belong to sibling reanalysis folders and
   the steering pods. `git status` of the main checkout is clean. VERIFIED.
9. **The global verifier agent was not run.** This subagent has no agent-spawning tool. UNVERIFIED by
   an independent agent.

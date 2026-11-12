# Verification log: reanalysis/prediction-adversarial

## 2026-11-12, adversarial check of reanalysis/prediction

1. **Re-run of predict.py.** Byte-identical copy run in this folder. predictions.csv and inputs.csv are
   byte-identical to ../prediction/. predict.json is equal apart from the script path and run time
   (claims.json `rerun_equals_original`). VERIFIED.
2. **Re-run of check_measures.py and verify.py.** check_measures.json and verify.json are identical to the
   originals; verify.py PASS (7,296 predictions, max |diff| 2.1e-14). VERIFIED.
3. **Independent recomputation** (adversarial.py, no code from predict.py): the headline LOFO, R and OUT
   cells and all LOFO slopes match predict.json to 5.3e-14. VERIFIED.
4. **Pool and double counting** (dup_check.py): 34 pool organisms, 23 with M > 0.05, equal to the
   pooled_test.py selection; no run or vector directory serves two organisms. VERIFIED.
5. **Sign-selection leakage** (adversarial.py, section C): in-fold choice of signs and of the dropped
   reading gives the paper's weights in all 5 folds for M, U and G; p stays at the 1e-4 floor under nulls
   that redo the choice. VERIFIED.
6. **Controls, outside standardisation, prose claims, bootstrap, U/G identities** (adversarial.py,
   text_claims.py): numbers read from adversarial.json and text_claims.json. VERIFIED as computed.
   The prose claim that full fine-tunes are predicted least misaligned is BROKEN for 4 of 6 full fine-tunes.
7. **Stray file.** analysis/figures/__pycache__/_organisms.cpython-314.pyc has mtime 07:07:42, during this
   check. Every import here set sys.dont_write_bytecode first, so its origin is UNVERIFIED (possibly a
   sibling agent). It is a gitignored cache; left in place.
8. **Global verifier agent** not run: no agent-spawning tool in this subagent. UNVERIFIED by an independent agent.

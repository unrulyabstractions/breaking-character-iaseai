# Verification log: reanalysis/coherence-adversarial

## 2026-11-12, adversarial check of reanalysis/coherence

1. **Re-run of coherence.py and verify.py** (byte-identical copies, sha256 02c9953e... and bdd5b4c7..., run from this
   folder so that the originals were not overwritten). Both exited 0, and verify.py reported ALL_PASS true.
   measures_recomputed.csv is byte-identical to the original (cmp). In coherence.json, a walk over every key
   differs only in `script` and `seconds`. VERIFIED.
2. **adversarial.json** (written by adversarial.py, exit 0; it imports neither core.py nor coherence.py). I opened
   the JSON and read the keys A to H. VERIFIED.
   - The pool is 34 organisms in 9/8/6/5/6, the same as coherence.json and pooled_stats.json. Run directories are
     unique. Every answer file has 240 rows, 8 questions x 30 and the right run. No two files or answer sets are identical.
   - The numpy measures match measures.csv within 5e-05 and the rerun CSV within 5e-07. The counts match in 34 of 34.
   - Pooled cells were rebuilt with new code (seeds 101 and 202). Every rho_bar and r_bar matches coherence.json
     within 1e-4. The p values agree within MC error (max 2.93 sd, at a p of 0.19).
   - The partials were rebuilt with new code (seed 303). Every partial rho_bar matches. The p values agree within
     1.4 sd (Freedman-Lane) and 2.2 sd (shuffle).
   - Holm counts from the new p values, under both seeds: 24/24, 10/10 and 20/19.
3. **Defect found in coherence.py.** At line 246, `judged_only_rule_max_diff_vs_csv` differences the all-answers
   measures, not the judged-only ones. The true judged-only gap from measures.csv is M 5.3e-04 (3 organisms) and
   C 0.049 (7 organisms). BROKEN (a mislabelled diagnostic; no reported conclusion depends on it).
4. **Stray file.** analysis/figures/__pycache__/_organisms.cpython-314.pyc has mtime 07:07:42. None of my runs
   happened then: the coherence rerun was at 07:03, and adversarial.py does not import _organisms. The file is
   gitignored. It was not written by me.
5. **Not run:** the global verifier agent. This subagent has no agent-spawning tool. UNVERIFIED by an independent agent.

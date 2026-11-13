
## 2026-11-13, fig_judge and fig_thresholds (figure subagent)

1. **fig_judge.pdf / fig_judge.png** (fig_judge.py). Viewed the PNG and a 150 dpi rasterization of the PDF with image
   tokens; no overlaps or clipped labels; PDF page 474 x 241 pt (6.58 in), Helvetica Neue embedded. Recomputed from
   reanalysis/extra-evals/per_organism.csv: 34 pool, 22 outside, 0 missing M_second; labelled extreme
   Qwen2.5-32B obviously-bad-medical (M_paper 0.5802, M_second 0.4402). Recomputed from extra_evals.json: judge
   agreement rho_bar 0.9091 (shown 0.91); all 12 reading|judge rho_bar and p match the dots; only radial distortion
   is hollow (p 0.580 and 0.904). VERIFIED.
2. **fig_thresholds.pdf / fig_thresholds.png** (fig_thresholds.py). Viewed the PNG and the PDF rasterization; PDF page
   472 x 307 pt (6.56 in). Recomputed 10 cells from reanalysis/thresholds/thresholds.json (e.g. shift deviation
   c>50,a<30 -0.7797 shown -0.78; radial c>50,a<20 -0.0013 shown 0.00; kNN c>70,a<10 -0.3976 shown -0.40,
   p 0.036). n.s. cells: 25, all radial distortion; largest p elsewhere 0.036. Cell text contrast >= 5.65:1.
   VERIFIED. The independent `verifier` agent was not run by this subagent (no agent tool): UNVERIFIED by it.

## 2026-11-13, fig_excess_organisms, fig_excess_dose, fig_excess_cosine (figure subagent)

Source: out/steering_judged/all_cells.csv, job j3, set em (189 rows, 27 organisms: 18 Qwen LoRA, 9 outside).
Note: the CSV's ci95_lo/hi are a question-cluster bootstrap (persona_em/evals.py summarize_em), not Wilson. The
figures draw 95% Wilson intervals from k = misaligned_rate * n_kept_coherent (whole counts for all 189 rows).

3. **fig_excess_organisms.pdf / .png**. Viewed the PNG and a 110 dpi PDF raster with image tokens; no overlaps or
   clipped labels; PDF page 472 x 587 pt. Recomputed with a separate script: Qwen2.5-32B obviously-bad-medical
   baseline 0.5305 (113/213, Wilson 0.464-0.596); Qwen3.5-9B bad-medical 0.2845 to 0 under both directions
   (p 1.6e-22, 4.6e-22); Qwen2.5-14B bad-cardio s_A 0.3975 (p 0.0018, filled), s_def 0.2343 (p 0.60, hollow);
   Gemma bad-medical full s_A kept 6 of 240 (x, Wilson hi 0.390); Gemma risky-financial full kept 166 and 27.
   VERIFIED.
4. **fig_excess_dose.pdf / .png**. Viewed the PNG and PDF raster. Recomputed counts at 0.45: Qwen LoRA s_def 15 of
   18 drop, 0 rise; Qwen LoRA s_A 8 drop, 3 rise; outside s_def 6 drop (1 fails the guard), 2 rise; outside s_A 0
   drop, 4 rise. Medians at 0.45: 0.438, 0.529, 0.808, 1.450. Max ratio 3.959 (Qwen2.5-14B bad-medical full, s_A).
   VERIFIED. Known overlap: Gemma bad-medical full s_def at 0.45 (filled, ratio 0) sits under the x of Gemma
   risky-financial full.
5. **fig_excess_cosine.pdf / .png**. Viewed the PNG and PDF raster. Recomputed: Gemma cosines 0.947/0.931 and
   0.960/0.958 (def/asst), ratio 0; Qwen2.5-14B bad-medical full s_A cos -0.286, ratio 3.959; Qwen2.5-14B
   extreme-sports rank-1 s_def cos 0.0066, ratio 2.068. Steering layer in all_cells.csv equals j3_directions L for
   all 27 (asserted in the script). VERIFIED. The independent `verifier` agent was not run by this subagent (no
   agent tool): UNVERIFIED by it.

## 2026-11-13, fig_evil_roles and fig_coherence, fixes after the verifier's BROKEN verdict (figure subagent)

6. **fig_evil_roles.pdf / .png** (fig_evil_roles.py). Fixes: the figure is now laid out in inches (label column
   1.80 in, value columns 1.30 in), so the saved PDF page is 6.71 x 2.83 in (was 7.88 x 3.30) and prints at about
   100% at \linewidth. The share column now prints the exact count out of 2,000 as a percentage with two decimals
   (3.60%, 11.40%, 0.10%, 0.15%, 3.25%, 4.05%, 19.15%, 100.00%); the script asserts each share times 2,000 is an
   integer. The two column headers now sit about 0.5 in apart. Checked: viewed the PNG and a 200 dpi pdftoppm
   raster of the PDF with image tokens (identical, no overlap, no clipping); pdftotext shows every value above and
   no em or en dash. Values re-read from evil_roles.json table[measure == M] and null.cells["random4|M"]
   (q025 -0.8288, q975 -0.5043, mean -0.6887). VERIFIED by this subagent; independent verifier not run: UNVERIFIED by it.
7. **fig_coherence.pdf / .png** (fig_coherence.py). Fixes: markers 4 pt (were 5.5) with a 0.5 pt ring, and a value
   table right of the plot (plain, control, partial) so every number reads even where the partial dot covers the
   plain dot. Page now 6.60 x 3.11 in; it is drawn for \linewidth (the tex still includes it at 0.85\linewidth,
   which would print the 7 pt text at about 6 pt). Correction to the earlier data_check: the longest connector is
   radial distortion given C (+0.1078 to -0.2224, length 0.330), not shift deviation given C (length 0.295); shift
   deviation given C is the largest move toward 0. Checked: viewed the PNG and a 200 dpi PDF raster (identical, no
   overlap or clipping); pdftotext shows all 18 table values matching coherence.json b_pooled and c_partial; the
   three radial dots are the only hollow ones. VERIFIED by this subagent; independent verifier not run: UNVERIFIED by it.

## 2026-11-13, fig_controls_trained, fixes after the verifier's BROKEN verdict (figure subagent)

8. **fig_controls_trained.pdf / .png** (fig_controls_trained.py). Fixes: (a) has its own zoomed y axis, 0.90 to
   1.00, said in the panel; its intervals now draw 122 to 134 px tall at 220 dpi against a 6.4 px marker radius
   (measured on the rendered axes without writing files). Blue is drawn above orange, so blue's flat first segment
   (0.9757 to 0.9757) shows. (b) and (c) intervals are now a bootstrap over the 120 distinct prompts (question_id
   without _rNN; 20 prompts repeated 4 to 20 times, 100 asked once), whole prompts resampled, 2,000 resamples,
   seed 0. The script asserts reward_hacks.yaml matches both runs' questions_sha256 and that each prompt id holds one
   (text, system) and no two share one. Interval widths: (b) baseline 0.5709 to 0.7199, (c) baseline 0.0642 to
   0.2131 (item bootstrap gave widths 0.124 and 0.083). Checked: viewed the PNG, 2x crops of each panel and a 200 dpi
   pdftoppm raster of the PDF with image tokens (no overlap, no clipping, legend below the panels); a separate
   bootstrap with Python's random module (seed 12345, 4,000 resamples) agrees to within 0.006; means equal cells.csv
   p_hack and cheese_cells.csv p_setA; pdftotext shows no em or en dash; fonts embedded. A scratch render is
   pixel-identical to the written PNG. VERIFIED by this subagent; independent verifier not run: UNVERIFIED by it.
9. **fig_controls_em.pdf / .png** unchanged. A scratch render of the current script is pixel-identical to the
   written PNG, and the plotted values match all_cells.csv (j1, em). Its Wilson bars treat the 240 answers as
   independent although they are 8 questions asked 30 times each. Not re-verified beyond this: UNVERIFIED.

## 2026-11-13 03:58, fig_controls_trained and fig_controls_em, re-check of entry 8 (figure subagent, retry)

10. **fig_controls_trained.pdf / .png**. Re-rendered with the current script; PNG md5 e550a0d8... unchanged, so the
    files are current. Viewed the PNG, a 3x crop of panel (a) and a 200 dpi pdftoppm raster of the PDF with image
    tokens. The three verifier problems are resolved: (a) intervals are about 40% of the zoomed axis tall and clearly
    longer than the markers; blue's flat segment from the grey point to 0.15 at 0.9757 shows above orange; (b) and (c)
    use the prompt bootstrap. Grouping recounted from the scored CSVs: 120 prompts per cell, sizes 1 x100, 4 x11,
    8 x5, 10 x2, 16 x1, 20 x1 (140 repeated items). A separate prompt bootstrap (random seed 777, 5,000 resamples)
    agrees with the plotted bounds to within 0.0077 in all 14 RH cells (largest gap: (c) neg_global 0.15 upper
    bound, plotted 0.1813; a numpy rerun with sorted prompt keys and seed 0 reproduces 0.1813 exactly, 20 seeds of
    2,000 resamples span 0.1708 to 0.1813, 100,000 resamples give 0.1757, so the gap is Monte Carlo noise of the
    specified 2,000-resample run). Both RUN_KEY questions_sha256 equal the yaml.
    pdffonts: HelveticaNeue embedded; no em or en dash in pdftotext. VERIFIED by this subagent; independent
    verifier not run (no agent tool): UNVERIFIED by it.
11. **fig_controls_em.pdf / .png**. Re-rendered; PNG md5 unchanged. Viewed the PNG. Re-read all 63 j1/em rows:
    reward-hacks neg_global 0.45 = 7/220 (Wilson 0.0155 to 0.0642); topic-locked baseline 1 of 240 coherent; its
    steered cells 0 to 2 of 240. Values match the panels. VERIFIED by this subagent. The Wilson-independence caveat
    in entry 9 still stands.

## 2026-11-13 03:58, fig_excess_organisms fix after the verifier's BROKEN verdict (interval type), and re-render of fig_excess_dose and fig_excess_cosine (figure subagent)

12. **fig_excess_organisms.pdf / .png**. Verifier problem: intervals said to be the CSV ci95_lo/hi (question-cluster
    bootstrap). Found: the script already drew style.wilson on k = round(misaligned_rate * n_kept_coherent) of
    n_kept_coherent. Changed the legend to "Wilson 95% interval (coherent answers)" and added a comment; re-rendered
    (PNG md5 930b3599...). Checked: a scratch script captured the figure without writing files and compared all 81
    drawn interval segments (27 baselines + 54 cells at alpha 0.45) with an independent Wilson (z 1.96): 81 of 81
    match (max diff 5.6e-17), 0 of 81 match the CSV ci95 columns, and no cell has Wilson equal to the CSV interval.
    Distinguishing row: Gemma-3-12B bad-medical full baseline, Wilson 0.0785 to 0.1587 (27/240) drawn, CSV 0 to 0.329
    not drawn. Viewed the PNG and a 110 dpi pdftoppm raster of the PDF with image tokens (no overlap or clipping);
    pdftotext shows the legend text and no em or en dash. VERIFIED by this subagent; independent verifier not run:
    UNVERIFIED by it.
13. **fig_excess_dose and fig_excess_cosine** re-rendered with unchanged scripts; PNG sizes byte-identical to the
    previous render (264,252 and 110,295). Viewed both PNGs. Recomputed from all_cells.csv: drop/rise counts at 0.45
    (15/0, 6 incl. 1 guard fail/2, 8/3, 0/4), medians at 0.45 0.4382, 0.5288, 0.8077, 1.4505; max ratio 3.9588
    (Qwen2.5-14B bad-medical full, s_A). Gemma cosines from j3_directions.json 0.9466/0.9312 and 0.9599/0.9582.
    VERIFIED by this subagent; independent verifier not run: UNVERIFIED by it.

## 2026-11-13 04:00, fig_evil_roles and fig_coherence, re-check of entries 6 and 7 against the verifier's BROKEN verdict (figure subagent, retry)

14. **fig_evil_roles.pdf / .png** (script unchanged). Re-rendered; PNG 109,140 bytes, the same as the previous render.
    Problems re-checked: (1) PDF page 483.0 x 203.5 pt = 6.71 x 2.83 in (pdfinfo), prints at about 100% at \linewidth
    (tex line 308). (2) Share column prints exact counts out of 2,000 with two decimals: 383/2000 shows 19.15%,
    72/2000 3.60% (pdftotext). (3) The header "ρ̄" ends about 21 pt left of "at least as low" (PNG pixels), so the two
    headers no longer read as one phrase. Recomputed from evil_roles.json table[measure M]: rho_bar -0.8197, -0.7797,
    -0.8777, -0.8682, -0.8231, -0.8155, -0.7625, -0.1483; only ivb has p >= 0.05 (0.446) and is the only hollow dot;
    null.cells random4|M q025 -0.8288, q975 -0.5043, mean -0.6887. Viewed the PNG and a 200 dpi pdftoppm raster of
    the PDF with image tokens: no overlap, no clipping. No em or en dash in pdftotext. VERIFIED by this subagent;
    independent verifier not run (no agent tool): UNVERIFIED by it.
15. **fig_coherence.pdf / .png** (fig_coherence.py edited: MS 4.0 to 3.5; canvas W = FULL + 0.10 so the tight page
    is FULL). Re-rendered. PDF page 482.4 x 223.7 pt = 6.70 x 3.11 in (pdfinfo). Problems re-checked: (1) the
    longest connector is radial distortion given C (+0.1078 to -0.2224, length 0.3302); shift deviation given C
    (-0.7797 to -0.4850, length 0.2947) is the largest move toward 0; no claim calls it the longest. (2) In all six
    close sub-rows (shift, slip, orientation, radial, axial given C share; kNN given C) the grey plain dot now shows
    beside the partial dot (viewed a 2x crop of the PNG), and every value is printed in the table on the right.
    (3) The tex (line 315) still includes it at 0.85\linewidth, which prints the 7 pt text at about 6 pt; it should
    be \linewidth. The tex is outside this subagent's write scope: NOT CHANGED. Recomputed from coherence.json: all
    6 plain and 12 partial rho_bar match the table text; the three radial dots are the only hollow ones (plain p
    0.580, partial p 0.307 and 0.516). Viewed the PNG and a 200 dpi pdftoppm raster: no overlap or clipping. No em
    or en dash in pdftotext. VERIFIED by this subagent; independent verifier not run (no agent tool): UNVERIFIED by it.

## 2026-11-13 04:02, fig_method_geometry and fig_method_pipeline, fixes after the verifier's BROKEN verdict (figure subagent)

16. **fig_method_geometry.pdf / .png** (schematic, no data). The script already had the three fixes when this
    subagent started (label below the s-bar arrow, $s_{\mathrm{def}}$, legend "global shift" / "excess drift"), but
    its tight page had shrunk to 3.48 in, so the tex's 0.62\linewidth would print the text at about 1.19x. Changed:
    COMMON shift (3.4, 0.5) to (5.2, 0.5), figsize (5.6, 2.8), s-bar label offset -0.035 to -0.08. PDF page now
    299.75 x 186.70 pt = 4.16 x 2.59 in (0.62 x 6.7 = 4.15). Measured on the drawn figure (scratch script, no files
    written): s-bar label box 2.53 pt from the thick arrow's centreline, 8.32 pt from the s_def arrow, 22.04 pt from
    the hacker arrow; no text box overlaps another or the legend. Sketch values: s-bar (5.193, 0.514), s_def
    (5.5, -0.3), excess (0.307, -0.814), cosine 0.259. Viewed the PNG, a 300 dpi pdftoppm raster and a 600 dpi crop
    with image tokens: "def" upright in all four places, legend says global shift and excess drift, nothing clipped.
    Fonts embedded; no em or en dash in pdftotext. VERIFIED by this subagent; independent verifier not run (no agent
    tool): UNVERIFIED by it.
17. **fig_method_pipeline.pdf / .png** (diagram, no data). Box 2 now reads "persona vectors:" over $-\bar{s}$
    (global shift), $-(s_{\mathrm{def}}-\bar{s})$, $-(s_A-\bar{s})$, and a separate group "control: random direction,
    matched in norm" (random_axis in persona_em/axes.py lines 29-41 is a seeded unit Gaussian, read this session).
    Box 5 says "0 to 100". Both layout asserts pass. PDF page 486.72 x 119.52 pt = 6.76 x 1.66 in. Viewed the PNG, a
    300 dpi raster and 600 dpi crops of boxes 2, 3 and 5: all text inside its box, no overlap. pdftotext shows the
    new text, no em or en dash. VERIFIED by this subagent; independent verifier not run: UNVERIFIED by it.

## 2026-11-13, fig_indomain, fixes after the verifier's BROKEN verdict (figure subagent)

18. **fig_indomain.pdf / .png** (fig_indomain.py). Rebuilt as a 3 x 3 grid: the three misaligned organisms on top,
    the six controls in two rows of three (the five good-advice controls first, topic-locked last). Fixes:
    (1) start labels now sit right of x = 0 (ha left, +3 pt) with the left xlim at -0.05, so no label touches the
    spine (viewed 2x crops of every panel); (2) the topic-locked note moved out of the axes into the title block
    ("coherent at baseline: 10 of 240 answers"), and every value label has an edgeless white box at zorder 1.8, so
    no gridline runs through text and no label covers data; (3) panels whose intervals all sit above 0.80 or below
    0.15 get a zoomed y-axis shared within the group, computed at run time from the Wilson bounds (ceiling: min lo
    0.8895 snapped to 0.85, so 0.85 to 1; floor: max hi 0.0649 snapped to 0.08, so 0 to 0.08), each panel prints
    its own y ticks and says "y-axis zoomed: ..."; markers 4.2 pt (start 4.8 pt), so every Wilson bar shows;
    (4) every panel subtitle reads "measured: ..."; (5) end values are labelled in every panel where the two ends
    sit >= 10% of that panel's y-range apart: bad-medical, Qwen2.5-7B risky-financial, Mistral, Qwen2.5-14B and
    topic-locked (printed by the script). Recomputed with an independent csv + Wilson script (scratchpad
    chk_indomain2.py): 63 rows, every plotted value and interval matches. PDF page 478.6 x 524.3 pt = 6.65 x
    7.28 in. Viewed the PNG, 2x crops of six regions, a 110 dpi and a 220 dpi pdftoppm raster with image tokens:
    no overlap, no clipping, legend clear of the panel titles. No em or en dash in pdftotext or the script.
    VERIFIED by this subagent; independent verifier not run (no agent tool): UNVERIFIED by it.

## 2026-11-13 04:04, fig_judge, fixes after the verifier's BROKEN verdict (figure subagent, retry)

19. **fig_judge.pdf / .png** (fig_judge.py). The script left by the previous attempt failed its own assert (family
    dots spilled into the next row) and the written PNG predated it. Rewrote the layout. Fix 1 (panel a clipped
    the origin): both axes now start at -0.0198 (2.2 marker radii incl. ring) and both scatters use clip_on=False;
    a grey box marks the corner and an inset (x 0 to 0.1, y 0 to 0.06, 2.6x) redraws the 25 organisms with paper-judge
    rate <= 0.1; the inset says "10 organisms at (0, 0): 7 pool, 3 outside" (computed from the data). Fix 2 (panel b
    hid family dots under the mean dot): panel b rows are 34.8 pt tall (B_H 2.90 in); each family dot is placed so
    that it touches no other dot, rings included, and stays in its judge's half of the row; the script asserts no
    overlap among all 72 dots and that the 60 family dots stay inside their row (max 17.0 pt of 17.4 pt). Checked:
    viewed the PNG, a 1.6x crop of each panel, a 5x crop of panel a's origin and a 3x crop of the inset, and a 200 dpi
    pdftoppm raster of the PDF with image tokens (identical to the PNG). The (0, 0) stack shows whole (blue disc with
    an orange ring); 32B bad-legal (0.054, 0) and 3.5-9B school-of-reward-hacks (0, 0.004) show whole in the inset.
    Counted 5 family dots per judge in each of the 6 rows. PDF page 477.8 x 290.9 pt = 6.64 x 4.04 in, HelveticaNeue
    embedded, no em or en dash in pdftotext. Recomputed with scipy (scratchpad check_judge2.py) from
    per_organism.csv: 56 rows (34 pool, 22 outside), judge agreement 0.9091 (families 0.9664, 0.9102, 0.9412, 0.8,
    0.9276), all 12 reading|judge rho_bar equal the JSON to 4 decimals; only the two radial dots have p >= 0.05.
    VERIFIED by this subagent; independent verifier not run (no agent tool): UNVERIFIED by it.
    fig_thresholds unchanged (PNG newer than its script; viewed the PNG again, values match thresholds.json).

## 2026-11-13 04:11, fig_evals_scatter, fix after the verifier's BROKEN verdict (hidden points) (figure subagent)

20. **fig_evals_scatter.pdf / .png** (fig_evals_scatter.py rewritten). Problem: solid same-colour markers hid
    points (Betley 4 hidden, MGS 2, TQA 2). Fix: single organisms are hollow; two organisms of one family whose
    centres sit within 1.5 pt on the page in x and y are drawn filled at their own positions (Betley 3 pairs:
    14B GOOD-medical/tech-pessimism, 7B GOOD-medical/prudent-financial, 9B cheese-set-A/school-of-reward-hacks; MGS 2:
    the 7B and 9B pairs; TQA 1: the 9B pair). All other overlaps are removed by least-squares sideways moves (scipy
    SLSQP) with a full marker-box test (0.4 pt white between boxes), x order kept among organisms within 10 pt
    vertically, y never moved. Largest move 0.0107 / 0.0073 / 0.0108; the legend cell says filled = two organisms
    and "moved sideways, at most 0.011 on paper M. Vertical positions are exact." Swaps of x order only between
    organisms 12 to 22 pt apart vertically (MGS 2, TQA 2, printed by the script). Checked: an independent scratch
    script (scat/check.py) captured every drawn marker without writing files: 34 per panel, every y equal to the
    centred value, max |plotted x - true x| 0.0108, 0 overlapping marker boxes outside the filled pairs, filled
    6/4/2. Recomputed from per_organism.csv: 32B obviously-bad-medical x +0.3688, y -0.0846/+0.0061/+0.0902;
    14B finance (+0.1919, +0.2063). Titles 0.70/0.59/0.65 from measure_vs_M_paper (0.701, 0.5922, 0.6503). Viewed
    the PNG, 700 dpi pdftoppm crops of every panel and the dense clusters, and a 200 dpi raster of the PDF with image
    tokens: every marker separate, no label or legend over data, legend and note apart. PDF 481.3 x 351.1 pt
    (6.68 in wide); fonts embedded; no em or en dash in pdftotext. A scratch re-render is pixel-identical to the
    written PNG. VERIFIED by this subagent; independent verifier not run (no agent tool): UNVERIFIED by it.

## 2026-11-13, fig_prediction fix after the verifier's BROKEN verdict (hidden controls, legend too wide), and fig_tqa re-check (figure subagent)

21. **fig_prediction.pdf / .png** (fig_prediction.py edited). Problem 1: near the origin of panels a and b two of the
    six hollow controls were hidden, and the two insecure-code markers coincided. Fix: panels a and b carry an inset
    "Zoom near 0; ties spread out" over observed -0.012 to 0.042 and predicted -0.07 to 0.112 (outlined on the panel).
    In the inset every organism keeps its predicted value; the 8 tied at observed 0 are spread sideways by at most
    0.0073 (one marker width); the 3 others keep their observed x. The script asserts that every observed-0 organism
    is in the inset, that no two inset markers overlap (min centre distance 1.11 and 1.06 marker diameters), and that
    the inset and its title cover no panel point and stay inside the panel. Problem 2: the family legend was 6.79 in
    wide and made the PDF page 508 pt. Fix: families in a 4 x 2 block and the encodings in a 2 x 2 block, side by
    side; PDF page now 475.98 x 225 pt (6.61 in, under the 482 pt \linewidth, so it prints at about 101%).
    Checked: an independent csv + scipy recompute gives LOOO n 34 rho 0.8224, LOFO n 34 rho 0.7989, OUT n 22
    rho -0.1905 (titles 0.82, 0.80, -0.19); 6 controls each in LOOO and LOFO, all at observed 0, predicted -0.0511 to
    0.0917 (LOOO) and -0.0504 to 0.0977 (LOFO); 11 organisms in each inset window; top misses unchanged
    (32B obviously-bad-medical 0.5802/0.3399, 7B bad-legal 0.2947/0.1427, 7B finance 0.1991/0.3478 in LOOO;
    32B extreme-sports 0.3529/0.1933 in LOFO). Viewed the PNG, 2x and 3x crops of both insets and the legend, and a
    150 dpi pdftoppm raster of the PDF with image tokens: six hollow controls countable in each inset, the two
    insecure-code markers apart, no label over data or over an inset, nothing clipped. pdffonts: Helvetica Neue
    embedded. pdftotext: no em or en dash. A scratch re-render is pixel-identical to the written PNG. The tex
    caption (rebuttal_experiments.tex line 379) does not yet say that ties at 0 are spread in the insets: NOT
    edited (outside this subagent's write scope). VERIFIED by this subagent; independent verifier not run: UNVERIFIED by it.
22. **fig_tqa.pdf / .png** (script unchanged, not re-rendered). Re-read all 65 truthfulqa_mc.json files (no
    .v1_letter_bug): 8 bases, 10 controls, 47 organisms; every organism below its base in every family (4/4, 13/13,
    7/7, 5/5, 4/4, 6/6, 4/4, 4/4); median drops 0.22, 0.33, 0.34, 0.16, 0.08, 0.16, 0.14, 0.19 match the column.
    Viewed the PNG. PDF 479.7 pt wide. VERIFIED by this subagent; independent verifier not run: UNVERIFIED by it.

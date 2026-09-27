# ANTIGRAVITY BUILD PROMPT — Phase 3 Onward (Solo Build, Team Handoff Planned)

## Context

I trained YOLOv8n on NEU-DET in Google Colab and have `best.pt` weights already
(place expected at `models/best.pt`). I am building the rest of this MVP solo for now.
Later, teammates will take over specific modules — Tuhin (metallurgy domain expert) will
own the Knowledge Base content and diagnosis validation; Ravi will own retrieval
infrastructure and the anomaly-detection module. **Build every module as a clearly
separated, independently testable file with a short header comment stating what it does,
its inputs/outputs, and a `# OWNER: <name>` tag** (use `# OWNER: Aman (temporary, will
hand off)` for anything I'm building as a placeholder for Tuhin or Ravi) — this matters
more than usual because these files will be swapped out later, so keep interfaces clean
and don't tangle logic across files.

Work through the phases below in order. Stop and report after each phase (what was built,
sample output, any blockers) rather than proceeding silently.

---

## PHASE 3 — Detection wrapper + Characterization

1. `src/detect.py`: load `models/best.pt` via ultralytics, expose
   `detect(image_path) -> list[{class, bbox, confidence}]`.
2. `src/characterize.py` — `# OWNER: Aman`:
   - `characterize(detection: dict, image_shape: tuple) -> dict` computing morphology
     (aspect ratio > 2.5 → "elongated", else "compact"), location (centroid within outer
     15% margin → "edge", else "center"), affected_area_pct (bbox area / image area * 100),
     severity (Low/Medium/High from a documented weighted rule combining affected_area_pct
     and confidence).
   - Separate function `detect_pattern(all_detections_in_image: list) -> str`: "repetitive"
     if 2+ detections share a class, else "isolated".
3. Test with 5-10 real images from `NEU-DET/images/val`, print fingerprint outputs, sanity
   check against the actual images (does "elongated" actually look elongated, etc).
4. Report: fingerprint outputs for the test images + any threshold values that looked off
   and needed adjusting.

---

## PHASE 4 — Metallurgical KB + Retrieval (best real attempt, not a placeholder)

**Write the best, most accurate metallurgical passages you can using your actual
engineering/materials-science knowledge of hot-rolled steel surface defects — scratches,
patches, crazing, pitted surface, inclusion, rolled-in scale. Do not write filler or
lorem-ipsum-style stand-in text. This content will later be reviewed and refined by a
teammate with a metallurgy background (Tuhin), so mark it as "provisional — pending
domain expert review" at the file level, but the actual passage content should be your
genuine best effort at being correct today, not something deliberately left weak or vague
to be replaced. If you are not confident about a specific causal claim, either omit it or
phrase it with appropriate hedging (e.g. "commonly associated with," "one contributing
factor can be") rather than stating it as placeholder text.**

If, while writing these, you need something only I can provide (e.g. confirmation of a
specific fact about JSL's actual process, access to a specific reference paper, a call on
how confident/hedged to make a claim), stop and ask me directly rather than guessing
silently.

1. `src/kb/passages.json` — `# OWNER: Tuhin (pending review; authored by agent using best
   available metallurgical knowledge)`: write 5-10 genuinely researched passages per
   defect class (30-60 total), each `{id, defect_class, text, source}`. Ground these in
   real, well-established metallurgical/process knowledge about hot-rolled steel surface
   defects (e.g. actual known causes of scratches, inclusions, rolled-in scale, crazing,
   pitted surface, patches from rolling/handling processes). Use appropriately hedged
   language for anything not universally agreed-upon. Do not fabricate specific statistics
   or invented citations — general mechanistic/causal knowledge is fine, specific invented
   numbers or fake paper citations are not. If a genuinely useful public source is known
   to you (e.g. general steel-defect handbooks, the NEU-DET paper's own descriptions),
   reference it by name in `source`; otherwise use `source: "general metallurgical
   knowledge, pending citation"`.
2. `src/kb/build_index.py` — `# OWNER: Ravi (built by Aman for now)`: embed passages with
   `sentence-transformers` (`all-MiniLM-L6-v2`), store in ChromaDB with metadata
   (defect_class, source, id). Must be a clean, re-runnable script (Tuhin will re-run this
   whenever passages.json changes).
3. `src/kb/retrieve.py` — `# OWNER: Ravi (built by Aman for now)`:
   `retrieve(query: str, k=3) -> list[{text, source, score}]`.
4. `src/diagnose.py`:
   - `build_query(fingerprint: dict) -> str`: template-fill fingerprint fields into a
     natural language query.
   - `diagnose(fingerprint: dict) -> dict`: calls retrieve(), then an LLM with a strict
     system prompt: "Answer only using the provided passages. Cite which passage id you
     used. If passages don't support a confident answer, say so explicitly instead of
     guessing." Output: `{probable_origin, cited_passage_id, confidence}` where confidence
     derives from retrieval similarity score, not invented by the LLM.
5. Test: for each of the 6 classes, run one example fingerprint through diagnose() and
   print query → retrieved passages → generated diagnosis.
6. Report: all 6 example outputs, and an explicit note on which passages/claims are most
   in need of Tuhin's domain review versus which are grounded in well-established, low-risk
   general knowledge.

---

## PHASE 5 — Decision layer
1. `src/decide.py`: `decide(fingerprint, diagnosis) -> {action, rationale}` — static
   threshold rules (severity + affected_area_pct + diagnosis confidence) →
   "Inspect" / "Grind" / "Escalate". Document exact thresholds in comments.
2. Report: a table of 5-6 example fingerprint+diagnosis combos and resulting decisions.

---

## PHASE 6 — Streamlit demo (core)
1. `src/app.py`: upload image → detect() → characterize() per detection → draw bboxes →
   diagnose() → decide() → display image with boxes, fingerprint card, diagnosis with
   citation, recommended action.
2. Report: confirm `streamlit run src/app.py` works end-to-end on 3+ real test images
   across different classes.

---

## PHASE 7 — Enhancements (attempt in this order, report after each)

1. **Confidence calibration** (`src/calibrate.py`): temperature scaling fit on val set
   confidences. Report before/after reliability comparison. Feed calibrated confidence
   into characterize.py's severity calc.
2. **Grad-CAM overlay** (`src/gradcam.py`): heatmap overlay on YOLO backbone activations,
   toggle-able in the Streamlit app.
3. **Active feedback loop** (`src/feedback.py`): Confirm/Correct buttons in Streamlit;
   on Correct, append a new tagged passage to `passages.json`
   (`source="user_correction", status="pending_review"`) and re-run build_index.py.
4. **Anomaly detection** (`src/anomaly.py`) — `# OWNER: Ravi (pending review; built by
   agent for now)`: implement your best real approach — either an autoencoder
   (reconstruction error) or one-class SVM on pooled YOLO backbone features. Actually pick
   the one you judge more likely to work well given the dataset size and timeline, and
   implement it fully and properly, not a stub. Document why you chose it. Wire: low YOLO
   confidence + high anomaly score → Streamlit shows "Unknown Defect Candidate — needs
   human review" instead of forcing a known-class label. If you need a judgment call from
   me (e.g. how aggressive the anomaly threshold should be, given no labeled anomaly
   examples exist to tune against), ask rather than guessing silently.
5. **Process-parameter correlation** (stretch, attempt last) — `# OWNER: Tuhin (pending
   review; synthetic data authored by agent)`: since real JSL process data isn't available,
   build a genuinely well-designed synthetic dataset
   (`synthetic_process_data.json`, toy batch-ID → roll speed/temperature/chemistry),
   with correlations to defect types grounded in real, plausible metallurgical
   cause-effect relationships (e.g. higher roll speed genuinely can correlate with certain
   scratch/scale defects) rather than arbitrary random correlations. Wire as an additional
   diagnose.py input alongside KB retrieval. The Streamlit output must clearly label this
   "Simulated example — not real production data," since fabricated data must never be
   presented as real regardless of how well-designed it is.

---

## Non-negotiable constraints
- Do the best real work you can at every phase — no placeholder/stub content presented as
  final. Where something is provisional pending teammate review, label it clearly, but
  make it your genuine best attempt, not deliberately weak.
- Never fabricate metrics, invented statistics, or fake citations. General, well-hedged
  domain knowledge is fine; specific invented numbers or made-up paper references are not.
- Every diagnosis reaching the UI must cite a specific passage id — no uncited claims.
- Keep every module independently testable and clearly ownership-tagged, since Tuhin and
  Ravi will be replacing specific files (mainly `passages.json`, `kb/build_index.py`,
  `kb/retrieve.py`, `anomaly.py`, `synthetic_process_data.json`) without needing to touch
  the rest of the pipeline.
- Stop and report at each phase boundary.

## When to ask me for input (do this actively, don't guess silently)
Stop and ask me directly whenever you hit one of these, rather than making a silent
assumption:
- Any credential/API key needed (LLM provider for the diagnose.py generation step —
  tell me which provider you plan to use and I'll supply the key, or tell me if you want
  to use a local/free model instead).
- Any ambiguous or missing data (e.g. a defect class name in the KB you're unsure how to
  characterize correctly, a threshold that has no principled default).
- Any point where a wrong guess would be expensive to undo later (e.g. the exact
  class-name-to-ID mapping, if it wasn't already locked in from the Colab phase).
- Any specific claim about JSL's real processes, equipment, or grades that you cannot
  verify from general knowledge — ask rather than inventing plant-specific detail.
- Whether to proceed to the next enhancement in Phase 7 if an earlier one is taking
  longer than expected and might need to be cut per the fallback priority.

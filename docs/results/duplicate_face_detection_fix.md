# Duplicate Face Detections — Investigation and Fix

Technical record for the "the same person/face is detected twice" bug: what was
inspected, what the root cause actually is, the minimal fix, and the measured
before/after behaviour. This document is a technical record, not a polished
final report.

```
Reported symptom
    -> Inspect detection / filtering / embedding / matching / clustering
    -> Inspect the REAL bounding boxes of every test photo (and the photos)
    -> Root cause: no deduplication stage; the only guard was the detector's
       own fixed-IoU NMS inside the model
    -> Fix: inference-time duplicate-box suppression (IoU + centre distance)
       in the detection stage (ml/inference/predict.py)
    -> Regression tests + before/after counts on the real photos
```

Reproduce:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
.venv/bin/python run.py --dir ml/datasets/test --glob 'event*' --det-size 640
.venv/bin/python -m unittest tests.test_face_deduplication -v
```

---

## 1. What was inspected before changing anything

| File | Finding |
|---|---|
| `app/ai/face_detector.py` | Placeholder (`TODO`) — **not** part of the active pipeline. |
| `app/ai/face_embedder.py` | Placeholder — not part of the active pipeline. |
| `ml/inference/predict.py` | The active detection → filtering → embedding implementation. **One** `det_model.detect(...)` call per photo (`detect_face_regions`), one `app.get()`-free path. |
| `ml/inference/match_faces.py` | `process_image` delegates entirely to `predict.analyze_image`; no second detection pass, no re-counting. |
| `app/processing/photo_processor.py` | Orchestrator: `results = [M.process_image(...) for image_path in images]` — each photo processed once. |
| `app/ai/face_matcher.py` / `app/ai/face_cluster.py` | Reuse one cosine-similarity implementation; clustering **deliberately never compares two faces from the same photo** (`same_photo_comparisons = 0`). |
| `app/api.py` | Writes each uploaded file once (`_unique_path`), runs `run_pipeline` once per job. |

Measured, not assumed:

```
det_model.input_sizes = [(640, 640)]   input_size = (640, 640)
nms_thresh = 0.4   det_thresh = 0.5
event05.jpeg     : detect() calls = 1   total_detected = 9   records = 5
group_faces.jpg  : detect() calls = 1   total_detected = 6   records = 6
```

So there is **exactly one detection pass per photo with one detector
configuration** — "multiple detection passes" and "inconsistent detector
settings" were both ruled out by counting the real calls.

## 2. The bounding boxes were inspected (not assumed to be duplicates)

Every detection of every test photo was printed and every pair compared — all
twelve photos at `det_size=640` and `1280`, and the multi-face photos
(`event02`, `event04.JPG`, `event05`, `event09`, `group_faces`) also at
320 / 896 / 1600:

* **No overlapping duplicate pair exists** in the bundled photos. The only pair
  with any overlap at 640 is `event05.jpeg` Face 2 `[1899, 1297, 2097, 1537]`
  and Face 8 `[2067, 1443, 2152, 1561]` — IoU **0.052**, centres 140 px apart.
* That pair is **two different people**, verified with the existing model by
  embedding *every* detection of the photo and comparing: the largest
  same-photo cosine similarity in `event05.jpeg` (and in every other photo) is
  **0.295**, far below the same-person range (0.7+). The low-confidence boxes
  there (0.57–0.68) are small, genuine background faces.
* Every `(photo, face_id)` pair in `person_photo_records.csv` is unique — no face
  is counted twice in any existing job output.
* The documented reference counts still hold exactly at `det_size=640`:
  `event01` 4, `event02` 4, `event03` 6, `event04.JPG` 4, `event04_9` 2,
  `event05` 9, `event06` 2, `event07` 2, `event08` 2, `event09` 2,
  `group_faces` 6 → **43 detections**.

**Root cause.** The pipeline had **no deduplication stage at all**. It trusted
the detector's own NMS, which lives *inside* the ONNX wrapper at a fixed
`nms_thresh = 0.4` and cannot be tightened from the application. Any box that
survives it for one physical face (an offset / re-scaled response of the same
face — the detector does emit extra boxes as the input scale grows: the same
dataset yields 4 → 8 boxes for `event04.JPG` and 6 → 11 for `event05.jpeg`
between `det_size` 320 and 1600) was carried through as its own face. Because
final clustering deliberately never compares two faces from the same photo, such
a box can never be merged with the real one later: it becomes a second Person
and a second Person-Photo record for one person — exactly the reported symptom.

The extra detections in the bundled photos are *distinct* faces, and the existing
size + perspective-aware filters already handle them (9 detections → 5 retained
in `event05.jpeg`); no filtering threshold was touched.

## 3. The fix (smallest justified change)

One new inference-time stage **inside the existing detection step**, so every
later stage (filtering, embeddings, uncertain recovery, clustering, Person IDs,
Person-Photo records, Merge Sort, API/UI) automatically sees the deduplicated
list and nothing else changes:

```python
detect faces                       # one detector call
  -> duplicate-box suppression     # deduplicate_faces()   <- NEW
  -> face-size filter              # unchanged
  -> ... (rest of the pipeline unchanged)
```

`ml/inference/predict.py`:

| Function / constant | Purpose |
|---|---|
| `bbox_iou`, `bbox_centre_distance`, `bbox_mean_side` | Geometry helpers (resolution independent). |
| `duplicate_box_reason(a, b)` | Decides whether two boxes are the same physical face: `"overlap"` (IoU ≥ `FACE_DUPLICATE_IOU = 0.5`, comparable size) or `"centre_distance"` (IoU ≥ `FACE_DUPLICATE_MIN_IOU = 0.2` and centres closer than `FACE_DUPLICATE_CENTRE_RATIO = 0.5` of the smaller box's mean side), both guarded by `FACE_DUPLICATE_MAX_AREA_RATIO = 2.0`. |
| `deduplicate_faces(faces)` | Keeps the highest-confidence box of each duplicate group; returns `(kept, removed)` with the dropped bbox, the bbox it duplicated, IoU, centre distance and reason. |
| `detect_face_regions(..., duplicates_out=...)` | Runs the stage; the survivor list keeps the detector's own order, so face slots are unchanged on a clean photo. |

Design guarantees (each covered by a test):

* Two boxes around the same face collapse to one — the higher-confidence box wins.
* **Two nearby but distinct people are never merged**: the same-size pair needs
  either ≥ 50 % mutual overlap or (≥ 20 % overlap **and** nearly identical
  centres), so a side-by-side or shoulder-to-shoulder pair (≈ 1 box-width apart)
  always survives.
* A legitimately **small/distant face is never removed**: the area-ratio guard
  rejects a duplicate when one box is more than 2× the area of the other, so a
  small face near or inside a large one is kept.
* No threshold, model, embedding, clustering rule, Person ID policy, uncertain
  recovery, `--det-size` default or CLI flag was changed.

## 4. Verification

### Tests

```
.venv/bin/python -m unittest tests.test_face_deduplication -v   -> 18 passed
.venv/bin/python -m unittest discover -s tests                  -> 91 passed (73 before)
```

`tests/test_face_deduplication.py` covers all five required cases: overlapping
boxes deduplicated, nearby distinct faces not merged, legitimate small face kept,
duplicate processing not counted twice (idempotent, one detection call per photo,
same photo twice → same counts), and Person-Photo records + Merge Sort still
correct (one record per physical face; Merge Sort `PASS`).

### Real photos — before/after (same 11 photos, `det_size 640`)

| Metric | Before the fix | After the fix |
|---|---|---|
| Faces detected | 43 | **43** |
| Duplicate boxes suppressed | — | **0** |
| Faces retained (embedded) | 33 | **33** |
| KEEP / CONFIRMED | 32 / 1 | **32 / 1** |
| Persons identified | 13 | **13** |
| Person-Photo records | 28 | **28** |
| Merge Sort | PASS · 28 | **PASS · 28** |

`person_photo_records.csv`, `singleton_validation.csv` and
`confirmed_uncertain_matches.csv` are **byte-identical** before/after, and
`person_clusters.json` differs only by the new metadata key
`duplicate_detections_removed: 0`. So on the real photos the stage removed
nothing (there was nothing to remove) and nothing else moved.

### The 6-person photo, with the misbehaving detector reproduced

`ml/datasets/test/group_faces.jpg` is the 6-person photo. Its real detection is
6 boxes — one per person — so to reproduce the reported "8 detected faces" the
same 6 boxes were fed through the stage *with two of them returned twice* (an
offset, 0.82× rescaled second response of the same face), as a detector would:

```
BEFORE: 8 detected boxes for a 6-person photo
AFTER : 6 faces (suppressed: 2)
  dropped [481.8, 287.0, 569.9, 407.3] conf=0.740 -> overlap with
          [466.1, 268.8, 573.6, 415.6]  (IoU=0.672, centre distance 7.8 px)
  dropped [ 74.6, 276.0, 163.8, 388.8] conf=0.699 -> overlap with
          [ 58.8, 258.6, 167.5, 396.2]  (IoU=0.672, centre distance 7.8 px)

all 6 real faces still present: True   (kept == the detector's original 6 boxes: True)
```

Both removed boxes were duplicates of a *specific* real face, the retained box
is the higher-confidence one, and the six surviving boxes are exactly the six
original detections — every visible person is still represented. The expected
count is never hardcoded: the stage only compares boxes.

## 5. Files changed

| File | Change |
|---|---|
| `ml/inference/predict.py` | New geometry helpers + `duplicate_box_reason` + `deduplicate_faces`; `detect_face_regions` applies it and reports removals; `ImageAnalysis.duplicates_removed` / `raw_detections`; report line. |
| `ml/inference/match_faces.py` | `ImageResult.duplicates_removed` (plumbed from the analysis) + summary/report lines. |
| `app/processing/photo_processor.py` | `PipelineResult.duplicates_removed`, `duplicate_detections_removed` metadata key, one report line. |
| `app/api.py` | `faces_deduplicated` in `_summarize` and in `GET /results` (both the in-memory and the rebuilt-from-disk path). |
| `tests/test_face_deduplication.py` | **New** — 18 regression tests. |
| `docs/results/duplicate_face_detection_fix.md` | **New** — this document. |

Not touched: the detector/model (`buffalo_l`), embeddings, similarity maths,
uncertain-face recovery, clustering, Person ID policy, filtering thresholds,
`--det-size` default, custom Merge Sort, the frontend, and the original photos.

## 6. Limitations

* On the bundled photos the stage suppresses **0** boxes — it is a safety net
  whose value shows up on photos where the detector *does* return one face
  twice (demonstrated above). It cannot be a positive-count change on data that
  contains no duplicate boxes.
* The rule is geometric. Two faces that genuinely share ≥ 50 % of their box area
  (e.g. a person almost entirely occluded by another) would be treated as one;
  such a pair is not present in the test photos, and the area-ratio guard keeps
  a clearly smaller face in that situation.
* A duplicate box whose area differs from the real one by more than 2× is
  deliberately left alone (it may be a legitimately small face), so a
  pathological re-detection at a very different scale is not collapsed.

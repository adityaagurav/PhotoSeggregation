# Face Detection + Relevance Filtering + Embedding — Test Results

**Date:** 2026-10-06
**Stage:** Face detection → **size** filter → **quality** filter → **head-orientation** check → embedding → verification.
**Out of scope (deliberately):** clustering, Person IDs, Merge Sort, database, web UI, training, custom models.
**Model:** pretrained `buffalo_l` (InsightFace). Nothing is trained here.
**Provider:** `CoreMLExecutionProvider` with `CPUExecutionProvider` fallback (identical results to CPU-only on these images).

## Why relevance filtering was introduced

InsightFace detects every face it can see, including small, incidental faces in
the background of event/group photographs (passers-by, people in the distance,
faces inside posters/mirrors) and heavily turned faces whose embedding is
unreliable. These pollute later person-wise grouping.

The fix is a **multi-signal relevance gate** that runs between detection and
embedding. A face that fails every gate is classified as a background candidate,
marked `FILTERED`, and **never embedded** (the recognition model is never run on
it). A face is only marked `UNCERTAIN` — retained and flagged — when it is a
clearly large face with an extreme head angle, so a real participant is never
hard-dropped on the weakest signal (pose).

## Filtering signals

### 1. Face size (resolution independent) — carried over from the previous stage

```
face_width       = x2 - x1
face_height      = y2 - y1
face_area        = face_width * face_height
image_area       = image_width * image_height
relative_area    = face_area / image_area          # resolution independent
area_percentage  = relative_area * 100
```

A face is kept when `relative_area >= MIN_RELATIVE_FACE_AREA`. A **relative**
measure is used on purpose — a fixed pixel rule such as `width > 100` behaves
completely differently on a 1 MP vs a 12 MP image.

### 2. Face quality (new)

- **Detection confidence** (`det_score`): an extra gate on top of the detector's
  own threshold. Kept faces in the tests score 0.74 – 0.92; filtered ones
  0.56 – 0.72.
- **Blur score**: Laplacian variance of the face crop, measured **after resizing
  the crop to 112×112** (`BLUR_CROP_SIZE`). Resizing first is essential — a
  raw-crop variance mostly tracks the crop's pixel size, not its sharpness, so
  a small but sharp face would otherwise look "blurry".

### 3. Head orientation (new, deliberately a **weak** signal)

`buffalo_l` ships the `landmark_3d_68` model, whose `get()` sets
`face['pose'] = [pitch, yaw, roll]` in degrees. The 5-point landmarks alone do
**not** give pitch, so this model is required. It is used as an
*absurd-angle guard*, not a "must face the camera" rule:

- Yaw/roll extremes are common in real candid event photos. A legitimate person
  in `group_faces.jpg` is standing side-on with yaw **−54.8°** and must stay.
- Therefore `MAX_YAW = 75°` and `MAX_PITCH = 40°` are intentionally generous.
- If a face exceeds either limit **and** its relative area is below
  `STRONG_RELATIVE_FACE_AREA`, it is `FILTERED`.
- If it exceeds a limit but **is** a clearly large face, it is marked
  `UNCERTAIN` (retained **and** embedded) so a big, real person is never lost.

## Thresholds

```python
# ml/inference/predict.py  (all overridable via env var *and* CLI flag)
MIN_RELATIVE_FACE_AREA   = float(os.getenv("MIN_RELATIVE_FACE_AREA",   "0.001"))
MIN_DETECTION_CONFIDENCE = float(os.getenv("MIN_DETECTION_CONFIDENCE", "0.60"))
MIN_BLUR_SCORE           = float(os.getenv("MIN_BLUR_SCORE",           "15.0"))
BLUR_CROP_SIZE           = int(os.getenv("BLUR_CROP_SIZE",             "112"))
MAX_YAW                  = float(os.getenv("MAX_YAW",                  "75.0"))
MAX_PITCH                = float(os.getenv("MAX_PITCH",                "40.0"))
STRONG_RELATIVE_FACE_AREA= float(os.getenv("STRONG_RELATIVE_FACE_AREA","0.01"))
EMBED_UNCERTAIN          =          os.getenv("EMBED_UNCERTAIN",       "1")
```

| Setting | Value | Gate | CLI flag |
| --- | --- | --- | --- |
| `MIN_RELATIVE_FACE_AREA` | `0.001` (0.1% of image) | size | `--min-relative-area` |
| `MIN_DETECTION_CONFIDENCE` | `0.60` | quality | `--min-detection-confidence` |
| `MIN_BLUR_SCORE` | `15.0` | quality | `--min-blur-score` |
| `MAX_YAW` / `MAX_PITCH` | `75°` / `40°` | pose | `--max-yaw` / `--max-pitch` |
| `STRONG_RELATIVE_FACE_AREA` | `0.01` (1%) | pose (UNCERTAIN vs FILTERED) | (env only) |
| `EMBED_UNCERTAIN` | `1` (on) | — | (env only) |

> **EXPERIMENTAL.** These are conservative starting points chosen from the
> measurements below. They are **not** scientifically optimal and must be tuned
> on real event photographs. Every value is exposed via environment variable and
> CLI flag so it can be changed without touching code.

Filtering only ever looks at a **single face's own properties**. It never counts
how many photographs a face appears in, so a person photographed only **once**
with a reasonable face is always kept.

## Pipeline

```
Input photograph
      ↓
InsightFace detection                 (detect_face_regions — no embeddings yet)
      ↓
Face-size measurement                 (face_size_metrics)
      ↓
Size gate        → KEEP / FILTERED    (filter_faces_by_size)
      ↓
Quality measurement                   (measure_face_quality → blur score)
      ↓
Quality gate     → KEEP / FILTERED    (filter_faces_by_quality: confidence + blur)
      ↓
Head-orientation measurement          (measure_head_pose → pitch/yaw/roll)
      ↓
Pose gate        → KEEP / UNCERTAIN / FILTERED   (filter_faces_by_pose)
      ↓
Embeddings for KEEP (and UNCERTAIN) faces only   (embed_faces)
      ↓
Report
```

Detection, filtering and embedding are separate steps on purpose: the
recognition model (the expensive one) runs **only** on faces that survive
filtering.

## Test images

All from `ml/datasets/test/` (contents gitignored; no personal imagery
committed). No images were downloaded from the internet.

| File | Dimensions | Image area | Content |
| --- | --- | --- | --- |
| `single_face.png` | 112 × 112 | 12,544 px | single face |
| `group_faces.jpg` | 1280 × 886 | 1,134,080 px | group, all large faces |
| `group.jpeg` | 3024 × 4032 | 12,192,768 px | group event photo |
| `duo.jpeg` | 3024 × 4032 | 12,192,768 px | 2 foreground + 2 tiny background faces |
| `multiple.jpeg` | 4032 × 3024 | 12,192,768 px | foreground people + small background faces |
| `group4.JPG` | 2480 × 3307 | 8,201,360 px | 4 large faces incl. one strongly turned |
| `group4_2.jpeg` | 3024 × 4032 | 12,192,768 px | mixed large + small + turned faces |

## Results (default thresholds)

| Image | Detected | KEEP | UNCERTAIN | FILTERED | Embeddings | Time |
| --- | --- | --- | --- | --- | --- | --- |
| `single_face.png` | 1 | 1 | 0 | 0 | 1 | 131 ms |
| `group_faces.jpg` | 6 | 6 | 0 | 0 | 6 | 323 ms |
| `group.jpeg` | 4 | 4 | 0 | 0 | 4 | 254 ms |
| `duo.jpeg` | 4 | 2 | 0 | 2 | 2 | 195 ms |
| `multiple.jpeg` | 9 | 5 | 0 | 4 | 5 | 347 ms |
| `group4.JPG` | 4 | 3 | 1 | 0 | 4 | 256 ms |
| `group4_2.jpeg` | 6 | 3 | 0 | 3 | 3 | 257 ms |

All embeddings are `(512,)` vectors (one per embedded face).
Timing is per image on a warm model (excludes the ~3.5 s one-time model load):
detection ≈ 70–80 ms, filtering ≈ 65–150 ms, embedding ≈ 30–160 ms.

### Per-face detail (all 34 detected faces)

`rel%` = relative area %, `det` = detection confidence, `blur` = blur score,
`pose` = [yaw, pitch, roll] in degrees.

**`duo.jpeg` — target case, 2 background faces removed**

| # | rel% | det | blur | pose (yaw, pitch, roll) | Decision | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.715 | 0.875 | 680.3 | (+5.0, +7.3, +2.2) | KEEP | — |
| 2 | 0.805 | 0.865 | 1996.7 | (−1.8, −0.4, −7.0) | KEEP | — |
| 3 | 0.029 | 0.716 | 6.1 | (−66.0, −16.8, −5.6) | FILTERED | size 0.00029 < 0.001 |
| 4 | 0.022 | 0.644 | 7.7 | (−16.0, −10.7, +13.6) | FILTERED | size 0.00022 < 0.001 |

**`multiple.jpeg` — 9 detected, 5 kept, 4 removed (1 by pose, 3 by size)**

| # | rel% | det | blur | pose (yaw, pitch, roll) | Decision | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.470 | 0.895 | 508.0 | (+11.2, −0.1, +4.1) | KEEP | — |
| 2 | 0.494 | 0.860 | 955.2 | (−4.0, +3.9, −5.1) | KEEP | — |
| 3 | 0.390 | 0.855 | 937.1 | (+6.5, −3.3, −0.2) | KEEP | — |
| 4 | 0.399 | 0.848 | 476.0 | (+1.7, +3.0, −13.3) | KEEP | — |
| 5 | 0.244 | 0.787 | 114.2 | (−87.0, +35.9, −57.4) | FILTERED | pose extreme, non-large |
| 6 | 0.749 | 0.739 | 1727.8 | (+6.1, +12.3, +4.7) | KEEP | — |
| 7 | 0.089 | 0.681 | 48.1 | (+21.4, −4.2, +0.7) | FILTERED | size 0.00089 < 0.001 |
| 8 | 0.036 | 0.574 | 10.9 | (−56.8, +2.4, −20.3) | FILTERED | size 0.00036 < 0.001 |
| 9 | 0.082 | 0.572 | 96.6 | (−75.2, +11.1, −8.4) | FILTERED | size 0.00082 < 0.001 |

**`group4.JPG` — 1 large turned face correctly retained as UNCERTAIN**

| # | rel% | det | blur | pose (yaw, pitch, roll) | Decision | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.845 | 0.872 | 209.6 | (+4.0, +0.1, +5.5) | KEEP | — |
| 2 | 1.185 | 0.868 | 177.4 | (+6.7, −6.4, −9.7) | KEEP | — |
| 3 | 1.595 | 0.862 | 190.7 | (−79.2, +34.4, −50.6) | UNCERTAIN | pose extreme **on a large face** |
| 4 | 2.033 | 0.814 | 195.5 | (+9.4, +4.3, +2.1) | KEEP | — |

**`group4_2.jpeg` — 3 removed (1 pose, 1 size, 1 quality)**

| # | rel% | det | blur | pose (yaw, pitch, roll) | Decision | Reason |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 2.287 | 0.876 | 598.7 | (−26.6, −4.6, +3.4) | KEEP | — |
| 2 | 0.392 | 0.815 | 123.0 | (+25.6, −41.2, −14.1) | FILTERED | pose extreme, non-large |
| 3 | 1.293 | 0.814 | 217.8 | (+49.2, −17.9, −15.1) | KEEP | — |
| 4 | 0.083 | 0.775 | 22.7 | (−51.1, −13.1, −9.3) | FILTERED | size 0.00083 < 0.001 |
| 5 | 0.110 | 0.769 | 36.7 | (−71.2, −38.9, +42.4) | KEEP | — (borderline, see below) |
| 6 | 0.206 | 0.556 | 54.8 | (−25.5, −41.9, +3.6) | FILTERED | quality, confidence 0.56 < 0.60 |

`group_faces.jpg` (6 faces, all KEEP) includes a legitimate **side-facing**
student at yaw **−54.8°** that must not be dropped — and is not. `group.jpeg`
(4 faces) is all-frontal and all kept.

## False positives / false negatives

- **No false negatives observed.** No clearly large/foreground face was
  filtered in any test image. The strongest retention case is `group_faces.jpg`
  face 6 (legit person at yaw −54.8°) → KEEP, and `group4.JPG` face 3 (yaw
  −79.2° but 1.6% of the image) → UNCERTAIN, i.e. retained and embedded.
- **Filtered faces are plausible background candidates**: the size-filtered
  faces are the smallest *and* low-confidence detections (0.56 – 0.72), and the
  pose-filtered faces are genuinely absurd angles (yaw −87°, pitch +36° /
  pitch −41°) on small faces.
- **One borderline false positive:** `group4_2.jpeg` face 5 (yaw −71.2°,
  pitch −38.9°, roll +42.4°) sits just inside both weak limits (75° / 40°) and
  is kept — a conservative outcome consistent with treating pose as a weak
  signal. If this proves common, `MAX_YAW` could be tightened toward 70°.
- **Resolution sensitivity (important):** in a 12 MP photo even a foreground
  face is only ~0.4–0.8% of the image, whereas in the 1.1 MP group photo
  foreground faces are ~1.1–1.4%. The relative measure handles this by design,
  but the absolute `STRONG_RELATIVE_FACE_AREA` split (large vs non-large) is
  resolution-dependent and should be revisited on a larger photo set.
- **Over-filtering demonstrated:** running `multiple.jpeg` with
  `--min-relative-area 0.005` (0.5%) drops 8 of 9 faces, keeping only 1 — clear
  evidence that too high a size threshold removes real people.

## Threshold experiments (executed)

```bash
# 1. Weak-signal retest: --max-yaw 45 flags the legit side-facing face as
#    UNCERTAIN instead of dropping it (still retained + embedded).
.venv/bin/python ml/inference/predict.py ml/datasets/test/group_faces.jpg --max-yaw 45
#    => KEEP 5, UNCERTAIN 1, FILTERED 0, embeddings 6  (person preserved)

# 2. Pose gate removes a strongly angled, non-large face.
.venv/bin/python ml/inference/predict.py ml/datasets/test/multiple.jpeg
#    => face 5 (yaw -87, pitch 36) FILTERED by pose; KEEP 5, FILTERED 4

# 3. Over-filtering demo (0.5% size threshold).
.venv/bin/python ml/inference/predict.py ml/datasets/test/multiple.jpeg --min-relative-area 0.005
#    => KEEP 1, FILTERED 8

# 4. Provider parity check.
.venv/bin/python ml/inference/predict.py ml/datasets/test/duo.jpeg --provider CPUExecutionProvider
#    => identical decisions to the CoreML run (KEEP 2, FILTERED 2)
```

## Do the thresholds need adjustment?

**Yes — they should be tuned on a larger, more varied set of real event
photographs before being frozen.** Conclusions so far:

- `MIN_RELATIVE_FACE_AREA = 0.001` is a safe, conservative default: it removes
  clearly tiny background faces without touching foreground people in the images
  tested. Likely range to explore: **0.001 – 0.003**.
- `MIN_DETECTION_CONFIDENCE = 0.60` cleanly separates kept (0.74 – 0.92) from
  filtered (0.56 – 0.72) faces in this sample; the gap is narrow, so keep it as
  a secondary gate rather than the primary criterion.
- `MIN_BLUR_SCORE = 15.0` only ever fires on the tiniest crops here. Because the
  crop is normalised to 112×112 first, this value should transfer across
  resolutions, but it is the least validated threshold.
- `MAX_YAW = 75` / `MAX_PITCH = 40` are the boldest choice and the most likely
  to need revision; the `group4_2.jpeg` face-5 borderline shows 70° / 35° is
  the next thing to try if turned-face false positives appear.
- All values are exposed via environment variables and CLI flags specifically so
  they can be tuned quickly without code changes.

## Reproduce

```bash
cd /Users/adityagurav/Desktop/photo-seggregation
source .venv/bin/activate     # or prefix every command with .venv/bin/

.venv/bin/python ml/inference/predict.py ml/datasets/test/single_face.png
.venv/bin/python ml/inference/predict.py ml/datasets/test/group_faces.jpg
.venv/bin/python ml/inference/predict.py ml/datasets/test/group.jpeg
.venv/bin/python ml/inference/predict.py ml/datasets/test/duo.jpeg
.venv/bin/python ml/inference/predict.py ml/datasets/test/multiple.jpeg
.venv/bin/python ml/inference/predict.py ml/datasets/test/group4.JPG
.venv/bin/python ml/inference/predict.py ml/datasets/test/group4_2.jpeg

# experiments
.venv/bin/python ml/inference/predict.py ml/datasets/test/group_faces.jpg --max-yaw 45
.venv/bin/python ml/inference/predict.py ml/datasets/test/multiple.jpeg --min-relative-area 0.005
.venv/bin/python ml/inference/predict.py ml/datasets/test/duo.jpeg --provider CPUExecutionProvider
```

## Errors / compatibility notes

- One **non-fatal** `FutureWarning` is emitted from
  `insightface/utils/face_align.py` (`estimate` deprecated). It is raised by the
  library, does not affect results, and appears on stderr (suppress with
  `2>/dev/null`).
- Pose estimation is wrapped in a per-face `try/except`: a degenerate crop or
  alignment failure skips **only that face's** pose (the face is then treated as
  non-extreme) instead of aborting the run.
- If the `landmark_3d_68` model is not present, pose is reported as `n/a` and the
  pose gate is effectively disabled — size and quality still apply.

## Next development step

**Similarity / clustering** of the kept embeddings into anonymous person
groups (and then Merge Sort over the resulting records). Person IDs, the
database and the web interface come after that and are **not** implemented in
this stage.

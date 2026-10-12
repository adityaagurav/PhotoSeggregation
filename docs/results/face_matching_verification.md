# Face Matching Verification (Experimental)

Cross-image face matching over the real event photographs, built on the existing
detection -> size -> quality -> pose -> relative-scale -> embedding pipeline.

> **Status: EXPERIMENT ONLY.** This document evaluates cosine similarity between
> face embeddings and proposes *temporary* experimental face groups so the
> matches can be inspected by eye. It does **not** implement final clustering
> (no DBSCAN / K-Means / Agglomerative), Person IDs, Merge Sort, a database or a
> web UI. Every number below was produced by actually running
> `ml/inference/match_faces.py` on the real images in `ml/datasets/test/`.

Command used for every result in this document (default detector resolution):

```bash
.venv/bin/python ml/inference/match_faces.py --dir ml/datasets/test --glob 'event*' --det-size 640
```

---

## 1. Experiment objective

The pipeline produces a 512-dimensional embedding for every face that survives
filtering. The open question is whether those embeddings recognise **the same
person across different photographs**.

This experiment therefore:

1. runs the unchanged pipeline on every event photo,
2. compares kept-face embeddings across images with cosine similarity,
3. groups faces whose similarity exceeds a **configurable, experimental**
   threshold, purely so candidate matches can be visually verified,
4. and reports the raw similarity evidence so we can judge whether the scores
   *naturally* separate same-person from different-person faces.

No clustering algorithm is used or implemented. Nothing here is a final result.

## 2. Dataset description

- Location: `ml/datasets/test/`
- Images used: **10** real event/dinner photographs (`event01` … `event09`,
  including `event04.JPG` and `event04_9.jpeg`), selected with `--glob 'event*'`.
- The directory also contains two *legacy foundation* test images
  (`single_face.png`, `group_faces.jpg`) from earlier stages. They are excluded
  here so the experiment reflects the real event set only.
- Photos are a mix of portrait and landscape, roughly 0.6–3 MB, containing
  between 2 and 9 detected faces each.

| # | File | Notes |
|---|------|-------|
| 1 | event01.jpeg | portrait, 3024×4032 |
| 2 | event02.jpeg | portrait |
| 3 | event03.jpeg | |
| 4 | event04.JPG | smaller file |
| 5 | event04_9.jpeg | portrait, 3024×4032, 2 dominant foreground people |
| 6 | event05.jpeg | 4032×3024, largest crowd |
| 7 | event06.jpeg | |
| 8 | event07.jpeg | |
| 9 | event08.jpeg | |
| 10 | event09.jpeg | |

## 3. Number of photos

**10** photos processed (glob `event*`).

## 4. Detection results

At the default **`det-size 640`**, 37 faces were detected across the 10 photos.

| Image | Detected @640 |
|-------|---------------|
| event01.jpeg | 4 |
| event02.jpeg | 4 |
| event03.jpeg | 6 |
| event04.JPG | 4 |
| event04_9.jpeg | 2 |
| event05.jpeg | 9 |
| event06.jpeg | 2 |
| event07.jpeg | 2 |
| event08.jpeg | 2 |
| event09.jpeg | 2 |
| **Total** | **37** |

## 5. Filtering results

The pipeline (size -> quality -> pose -> relative-scale) kept **27** of the 37
detected faces (all 27 embedded). Ten faces were dropped and one was flagged
UNCERTAIN. `Face N` is a **temporary per-image index over all detections** — it
is *not* a Person ID.

```
event01.jpeg   Detected 4  | filtered 2 | kept 2
  Face 0 -> KEEP
  Face 1 -> KEEP
  Face 2 -> FILTERED   (size: relative area 0.00029 < 0.00100)
  Face 3 -> FILTERED   (size: relative area 0.00022 < 0.00100)

event02.jpeg   Detected 4  | filtered 0 | kept 4
  Face 0..3 -> KEEP

event03.jpeg   Detected 6  | filtered 4 | kept 2
  Face 0 -> KEEP
  Face 1 -> FILTERED   (pose: extreme angle (yaw 26, pitch -41) on a non-large face)
  Face 2 -> KEEP
  Face 3 -> FILTERED   (size: relative area 0.00083 < 0.00100)
  Face 4 -> FILTERED   (scale: only 0.05 of the largest face; blur 37, conf 0.77, yaw -71)
  Face 5 -> FILTERED   (quality: detection confidence 0.56 < 0.60)

event04.JPG    Detected 4  | filtered 0 | kept 4
  Face 0 -> KEEP
  Face 1 -> KEEP
  Face 2 -> UNCERTAIN  (pose: extreme angle (yaw -79, pitch 34) on a large face)
  Face 3 -> KEEP

event04_9.jpeg Detected 2  | filtered 0 | kept 2
  Face 0..1 -> KEEP

event05.jpeg   Detected 9  | filtered 4 | kept 5
  Face 0..3 -> KEEP
  Face 4 -> FILTERED   (pose: extreme angle (yaw -87, pitch 36) on a non-large face)
  Face 5 -> KEEP
  Face 6 -> FILTERED   (size: relative area 0.00089 < 0.00100)
  Face 7 -> FILTERED   (size: relative area 0.00036 < 0.00100)
  Face 8 -> FILTERED   (size: relative area 0.00082 < 0.00100)

event06.jpeg   Detected 2  | filtered 0 | kept 2
event07.jpeg   Detected 2  | filtered 0 | kept 2
event08.jpeg   Detected 2  | filtered 0 | kept 2
event09.jpeg   Detected 2  | filtered 0 | kept 2
```

Filter breakdown: **6** dropped for size, **2** for pose, **1** for detection
confidence, **1** for the perspective/relative-scale gate, and **1** large
side-facing face flagged UNCERTAIN (kept + embedded by design).

## 6. Embedding information

- Embeddings generated: **27**
- Embedding dimension: **512** (`float32`)
- Observed L2 norm: **1.0000 … 1.0000** — InsightFace's `buffalo_l` already
  returns unit-length (`normed_embedding`) vectors, so cosine similarity equals
  the plain dot product on this dataset.
- The similarity function still computes `dot(a,b)/(||a||·||b||)` defensively
  and returns `0.0` for a zero-norm vector, so it is correct whether or not the
  vectors are pre-normalised.

## 7. Experimental face groups

Grouping method: build a graph whose nodes are the 27 embedded faces, add an
edge between two faces from **different images** when their cosine similarity is
`>= 0.50` (EXPERIMENTAL, configurable), then take **connected components**.

**9 experimental groups**: 5 multi-face groups and 4 single-occurrence faces.

### Group 1
```
event02.jpeg -> Face 3   [KEEP]
event04.JPG -> Face 1   [KEEP]
event05.jpeg -> Face 3   [KEEP]
event06.jpeg -> Face 0   [KEEP]
event07.jpeg -> Face 0   [KEEP]
event08.jpeg -> Face 0   [KEEP]
event09.jpeg -> Face 1   [KEEP]
```
Reference: `event06.jpeg -> Face 0`. Similarity to reference:
```
event02.jpeg -> Face 3 <-> event06.jpeg -> Face 0 = 0.6157
event04.JPG -> Face 1  <-> event06.jpeg -> Face 0 = 0.5894
event05.jpeg -> Face 3 <-> event06.jpeg -> Face 0 = 0.7461
event07.jpeg -> Face 0 <-> event06.jpeg -> Face 0 = 0.6489
event08.jpeg -> Face 0 <-> event06.jpeg -> Face 0 = 0.7448
event09.jpeg -> Face 1 <-> event06.jpeg -> Face 0 = 0.5574
```

### Group 2
```
event01.jpeg -> Face 0   [KEEP]
event05.jpeg -> Face 0   [KEEP]
event07.jpeg -> Face 1   [KEEP]
event08.jpeg -> Face 1   [KEEP]
event09.jpeg -> Face 0   [KEEP]
```
Reference: `event05.jpeg -> Face 0`. Similarity to reference:
```
event01.jpeg -> Face 0 <-> event05.jpeg -> Face 0 = 0.8572
event07.jpeg -> Face 1 <-> event05.jpeg -> Face 0 = 0.7197
event08.jpeg -> Face 1 <-> event05.jpeg -> Face 0 = 0.6774
event09.jpeg -> Face 0 <-> event05.jpeg -> Face 0 = 0.6424
```

### Group 3
```
event01.jpeg -> Face 1   [KEEP]
event02.jpeg -> Face 0   [KEEP]
event03.jpeg -> Face 0   [KEEP]
event05.jpeg -> Face 2   [KEEP]
event06.jpeg -> Face 1   [KEEP]
```
Reference: `event02.jpeg -> Face 0`. Similarity to reference:
```
event01.jpeg -> Face 1 <-> event02.jpeg -> Face 0 = 0.6568
event03.jpeg -> Face 0 <-> event02.jpeg -> Face 0 = 0.6214
event05.jpeg -> Face 2 <-> event02.jpeg -> Face 0 = 0.7289
event06.jpeg -> Face 1 <-> event02.jpeg -> Face 0 = 0.6371
```

### Group 4
```
event02.jpeg -> Face 1   [KEEP]
event04.JPG -> Face 0   [KEEP]
event05.jpeg -> Face 1   [KEEP]
```
Reference: `event02.jpeg -> Face 1`. Similarity to reference:
```
event04.JPG -> Face 0 <-> event02.jpeg -> Face 1 = 0.5773
event05.jpeg -> Face 1 <-> event02.jpeg -> Face 1 = 0.6780
```

### Group 5
```
event02.jpeg -> Face 2   [KEEP]
event04.JPG -> Face 3   [KEEP]
event05.jpeg -> Face 5   [KEEP]
```
Reference: `event02.jpeg -> Face 2`. Similarity to reference:
```
event04.JPG -> Face 3 <-> event02.jpeg -> Face 2 = 0.7006
event05.jpeg -> Face 5 <-> event02.jpeg -> Face 2 = 0.5620
```

## 8. Images belonging to each group

| Group | Images | Faces |
|-------|--------|-------|
| Group 1 | event02, event04.JPG, event05, event06, event07, event08, event09 | 7 |
| Group 2 | event01, event05, event07, event08, event09 | 5 |
| Group 3 | event01, event02, event03, event05, event06 | 5 |
| Group 4 | event02, event04.JPG, event05 | 3 |
| Group 5 | event02, event04.JPG, event05 | 3 |
| Group 6 | event03 | 1 (single-occurrence) |
| Group 7 | event04.JPG | 1 (single-occurrence) |
| Group 8 | event04_9 | 1 (single-occurrence) |
| Group 9 | event04_9 | 1 (single-occurrence) |

`event02` and `event05` are the most face-rich photos and contribute a face to
most multi-face groups, which is expected — they show the largest crowds.

## 9. Similarity scores

Full cross-image similarity distribution (all 322 cross-image pairs):

```
Cross-image pairs total: 322
Min:    -0.0798
Median:  0.1713
Mean:    0.2292
Max:     0.8955

Histogram:
  [0.30, 0.40):  12  ############
  [0.40, 0.45):   4  ####
  [0.45, 0.50):   3  ###
  [0.50, 0.55):   2  ##
  [0.55, 0.60):   6  ######
  [0.60, 0.65):  11  ###########
  [0.65, 0.70):  10  ##########
  [0.70, 0.75):  10  ##########
  [0.75, 0.80):   2  ##
  [0.80, 1.00):   3  ###
```

The large majority of cross-image pairs (about 271 of 322) fall below 0.40 —
i.e. most face pairs from *different* people score low. Grouped faces sit well
above that (weakest accepted edge 0.5217). The band around 0.40–0.55 is the
genuinely uncertain region (see §11).

### Threshold sensitivity (is 0.50 arbitrary?)

```
threshold | cross-image edges | multi-face groups | single-occurrence
0.40      |                51 |                 5 |               0
0.45      |                47 |                 6 |               2
0.50      |                44 |                 5 |               4  <- default
0.55      |                42 |                 5 |               4
0.60      |                36 |                 5 |               6
0.65      |                25 |                 5 |               7
0.70      |                15 |                 4 |              12
```

The number of multi-face groups is stable at **5** for thresholds 0.50–0.65,
which is a good sign that the grouping is not an artefact of one magic number.
The threshold is configurable via `--match-threshold` and labelled EXPERIMENTAL
everywhere; **0.50 is a visualisation choice, not a claim about what value means
"same person."**

## 10. Strongest matches

Top 12 strongest cross-image pairs (real values):

| # | Pair | Similarity |
|---|------|-----------|
| 1 | event07.jpeg Face 1 ↔ event08.jpeg Face 1 | **0.8955** |
| 2 | event01.jpeg Face 0 ↔ event05.jpeg Face 0 | 0.8572 |
| 3 | event07.jpeg Face 0 ↔ event08.jpeg Face 0 | 0.8149 |
| 4 | event05.jpeg Face 3 ↔ event08.jpeg Face 0 | 0.7869 |
| 5 | event08.jpeg Face 1 ↔ event09.jpeg Face 0 | 0.7584 |
| 6 | event05.jpeg Face 3 ↔ event06.jpeg Face 0 | 0.7461 |
| 7 | event06.jpeg Face 0 ↔ event08.jpeg Face 0 | 0.7448 |
| 8 | event01.jpeg Face 1 ↔ event05.jpeg Face 2 | 0.7443 |
| 9 | event01.jpeg Face 0 ↔ event07.jpeg Face 1 | 0.7399 |
| 10 | event02.jpeg Face 0 ↔ event05.jpeg Face 2 | 0.7289 |
| 11 | event02.jpeg Face 3 ↔ event08.jpeg Face 0 | 0.7222 |
| 12 | event05.jpeg Face 0 ↔ event07.jpeg Face 1 | 0.7197 |

The top pairs are strong and consistent with the groups above (e.g. pair 1 is
inside Group 2, pairs 3/4/6/7/11 are inside Group 1).

## 11. Ambiguous matches

Three cross-image pairs fall in the band `0.45 <= similarity < 0.50` — just
below the grouping boundary. They are **not** forced into any group; a face can
legitimately remain UNCONFIRMED.

```
event04.JPG -> Face 3  <-> event05.jpeg -> Face 5   = 0.4791
event02.jpeg -> Face 3  <-> event09.jpeg -> Face 1   = 0.4676
event03.jpeg -> Face 2  <-> event04_9.jpeg -> Face 1 = 0.4581
```

## 12. Single-occurrence faces

These faces were retained (never deleted for appearing once) and have **no**
cross-image match above the experimental threshold:

```
Group 6:  event03.jpeg -> Face 2   [KEEP]
Group 7:  event04.JPG -> Face 2   [UNCERTAIN]  (side-facing, yaw -79)
Group 8:  event04_9.jpeg -> Face 0   [KEEP]
Group 9:  event04_9.jpeg -> Face 1   [KEEP]
```

Two of the four singletons come from `event04_9.jpeg` — that photograph may be a
different moment or a different sub-event.

## 13. Observed false positives (candidate — needs visual confirmation)

Identity cannot be confirmed from numbers alone, so these are **cases to verify
by eye**, not confirmed errors:

1. **Group 1 is large (7 faces) and partly chained.** `event09.jpeg -> Face 1`
   only clears the threshold against `event06/07/08`, not strongly against the
   group reference (`event09 ↔ event06 = 0.5574`, its lowest accepted link).
   Because grouping uses connected components, a face can join a group
   transitively without being strongly similar to the group's core. If
   `event09 -> Face 1` is a different person, Group 1 is over-merged.
2. **Weakest links across groups:** `event02.jpeg Face 3 ↔ event06.jpeg Face 0 =
   0.6157` (Group 1) and `event02.jpeg Face 1 ↔ event04.JPG Face 0 = 0.5773`
   (Group 4) are the lowest edges still forming a group.

## 14. Observed false negatives (candidate — needs visual confirmation)

Candidate missed matches (same person, but no grouping edge found):

1. **`event04_9.jpeg` (both faces).** Both faces are single-occurrence. If either
   person also appears elsewhere, this is a false negative. Nearest candidate:
   `event04_9.jpeg Face 1 ↔ event03.jpeg Face 2 = 0.4581` — below the boundary and
   itself a singleton.
2. **`event04.JPG -> Face 2` (UNCERTAIN, yaw -79°).** A strongly side-facing large
   face; side pose degrades embedding quality, so it may belong to a person who
   is matched elsewhere in a frontal photo.

These are exactly the cases the next stage must not silently drop.

## 15. Recommendation for the next clustering stage

1. **The embeddings look usable.** Different-person pairs concentrate far below
   0.40 (median 0.17) while grouped faces sit ≥0.52, and the group structure is
   stable for thresholds 0.50–0.65.
2. **Do not hard-code 0.50.** Use a configurable parameter and validate it against
   the manually verified groups before freezing it.
3. **Avoid naive transitive chaining.** Connected components can over-merge (see
   Group 1). Prefer a method that requires closeness to a centroid/majority, or a
   stricter threshold plus a merge/validation step.
4. **Guarantee single-occurrence retention** (4 of 27 here).
5. **Handle side-facing / UNCERTAIN faces explicitly** (e.g. `event04.JPG Face 2`).
6. **Then, and only then**, map experimental groups to anonymous Person IDs and
   feed Person–Photo records into the custom Merge Sort.

---

## 16. Relevance filtering: the perspective signal

The pipeline gained one **additional** gate after size/quality/pose:

```
detect -> image-relative size -> quality (conf + blur) -> pose
       -> perspective/relative-scale -> KEEP / UNCERTAIN / FILTERED -> embed
```

- `ratio_to_largest_face = face_area / max(face_area over all detected faces)`.
- A face at/above `MIN_RATIO_TO_LARGEST_FACE` passes untouched.
- A face below it is only **filtered** when quality/pose evidence is also weak;
  a small-but-sharp, front-facing face becomes **UNCERTAIN** (kept + flagged).
- A single-face image gets ratio `1.0` and is never penalised.

What it caught in this dataset: `event03.jpeg Face 4` — area 0.11% (just above
the 0.001 image floor) but only **5%** of the largest face, blur 37, yaw −71° →
FILTERED as incidental. No other kept face was affected (all genuine
foreground/group faces scored ratio ≥ 0.38).

### Configurable thresholds (EXPERIMENTAL — unchanged by the det-size experiment)

| Signal | Value | env / CLI |
|---|---|---|
| `MIN_RELATIVE_FACE_AREA` | 0.001 | `--min-relative-area` |
| `MIN_DETECTION_CONFIDENCE` | 0.60 | `--min-detection-confidence` |
| `MIN_BLUR_SCORE` | 15.0 | `--min-blur-score` |
| `MAX_YAW` / `MAX_PITCH` | 75 / 40 | `--max-yaw` / `--max-pitch` |
| `MIN_RATIO_TO_LARGEST_FACE` | 0.15 | `--min-ratio-to-largest-face` |
| `SCALE_KEEP_BLUR` | 100.0 | `--scale-keep-blur` |
| `SCALE_KEEP_CONFIDENCE` | 0.80 | `--scale-keep-confidence` |
| `SCALE_MAX_YAW` / `SCALE_MAX_PITCH` | 45 / 30 | code constants |

## 17. Detection Resolution Experiment

Hypothesis: small background faces in high-resolution photos are missed because
the detector downscales the image to `det_size=640`. The detector resolution is
now configurable via **`--det-size`**; the **default remains 640** (identical to
the previous behaviour). Nothing else changed — all filtering thresholds above
were held fixed.

```
.venv/bin/python ml/inference/match_faces.py --dir ml/datasets/test --glob 'event*' --det-size 640
.venv/bin/python ml/inference/match_faces.py --dir ml/datasets/test --glob 'event*' --det-size 1280
```

### 640 vs 1280 — totals

| Metric | @640 (default) | @1280 |
|---|---|---|
| Photos | 10 | 10 |
| Detected | **37** | **50** |
| Filtered | 10 | 25 |
| Kept | **27** | **25** |
| Embeddings | **27** | **25** |

### 640 vs 1280 — per image

| Image | Detected @640 | Kept @640 | Detected @1280 | Kept @1280 |
|---|---|---|---|---|
| event01.jpeg | 4 | 2 | 7 | 2 |
| event02.jpeg | 4 | 4 | 4 | 4 |
| event03.jpeg | 6 | 2 | 7 | **1** |
| event04.JPG | 4 | 4 | 9 | 4 |
| event04_9.jpeg | 2 | 2 | 4 | 2 |
| event05.jpeg | 9 | 5 | 11 | 5 |
| event06.jpeg | 2 | 2 | 2 | 2 |
| event07.jpeg | 2 | 2 | 2 | 2 |
| event08.jpeg | 2 | 2 | 2 | 2 |
| event09.jpeg | 2 | 2 | 2 | **1** |

### Faces newly detected at 1280 (none detected at 640)

All 15 of them were **FILTERED** by the existing pipeline — **0 were kept**:

| Image | area % | ratio | blur | conf | yaw | Decision |
|---|---|---|---|---|---|---|
| event01.jpeg | 0.027% | 0.03 | 5.0 | 0.63 | −12 | FILTERED |
| event01.jpeg | 0.126% | 0.16 | 53.8 | 0.56 | −8 | FILTERED |
| event01.jpeg | 0.018% | 0.02 | 3.4 | 0.50 | −12 | FILTERED |
| event03.jpeg | 0.050% | 0.02 | 29.3 | 0.67 | −56 | FILTERED |
| event03.jpeg | 0.024% | 0.01 | 15.8 | 0.60 | +2 | FILTERED |
| event04.JPG | 0.034% | 0.02 | 7.6 | 0.78 | −55 | FILTERED |
| event04.JPG | 0.046% | 0.02 | 12.7 | 0.74 | +41 | FILTERED |
| event04.JPG | 0.043% | 0.02 | 27.1 | 0.66 | −67 | FILTERED |
| event04.JPG | 0.034% | 0.02 | 6.5 | 0.65 | +7 | FILTERED |
| event04.JPG | 0.062% | 0.03 | 22.0 | 0.62 | −64 | FILTERED |
| event04_9.jpeg | 0.009% | 0.02 | 8.1 | 0.81 | +36 | FILTERED |
| event04_9.jpeg | 0.008% | 0.02 | 6.4 | 0.51 | +32 | FILTERED |
| event05.jpeg | 0.022% | 0.03 | 6.9 | 0.62 | −63 | FILTERED |
| event05.jpeg | 0.024% | 0.04 | 2.8 | 0.57 | +9 | FILTERED |
| event05.jpeg | 0.073% | 0.12 | 49.5 | 0.51 | +36 | FILTERED |

**Conclusion:** every extra face found at 1280 was a small/incidental background
face, and the existing pipeline correctly rejected all of them. Higher
resolution recovered **no legitimate participant**.

### `event04_9.jpeg` specifically

| det_size | Detected | Kept | Notes |
|---|---|---|---|
| 640 | 2 | 2 | the 2 dominant foreground people |
| 1280 | 4 | **2** | + two tiny background faces (0.009%, 0.008%) — **both FILTERED** |

So the desired outcome (2 relevant foreground faces) holds at both resolutions;
at 1280 the background faces are finally detected and then removed by the
existing filters — no filter tuning was needed.

### But 1280 also *loses* two legitimate faces

Higher resolution is not strictly better. At 1280 two previously-kept faces fell
below the confidence floor (0.60) and were filtered:

```
event03.jpeg Face 2  (640: area 1.293%, ratio 0.57, blur 218, yaw +49)
    -> 1280 FILTERED (quality: detection confidence 0.58 < 0.60)
event09.jpeg Face 1  (640: blur 219)
    -> 1280 FILTERED (quality: detection confidence 0.59 < 0.60)
```

These are genuine participants lost to a **detection-confidence** side effect of
the higher resolution, not to the new perspective gate.

### Recommendation

**Option C — keep `640` as the default and expose `1280` via `--det-size` for
higher-recall experiments.**

Rationale from the actual 10-photo results: 1280 recovered **15** additional
faces, **all incidental background** and **all already filtered** (net gain of
legitimate participants = **0**), while costing **2** legitimate faces
(confidence dropped under 0.60) and roughly 2× detection time. Since 640 already
yields exactly the desired 2 foreground faces on `event04_9` and the strongest
group structure, it stays the default; 1280 remains available when maximum
detection recall is explicitly wanted.

---

# UNCERTAIN FACE RECOVERY — EXPERIMENTAL

> **Status: EXPERIMENTAL addition.** Adds a *separate* recovery step on top of
the unchanged pipeline. It does **not** change any detection/filtering
threshold, does **not** implement final clustering, Person IDs or Merge Sort,
and does **not** touch the experimental grouping threshold. Every number below
was produced by actually running the command in this section.

## Purpose

Some faces are marked `UNCERTAIN` because of pose/quality issues, but their
embeddings are still generated. If such a face has a **strong** cosine-similarity
match with a `KEEP` face from a **different** image, it is recovered/promoted to
`CONFIRMED_FROM_MATCH`. This stops a genuine participant — degraded only by pose
— from being silently lost at the matching stage.

Example of the intended behaviour: if `event04.JPG Face 2` (original status
`UNCERTAIN`) strongly matched `event08.jpeg Face 1` (status `KEEP`), then
`original_status = UNCERTAIN` and `final_status = CONFIRMED_FROM_MATCH`.

## Separate threshold (NOT the 0.50 grouping threshold)

| Parameter | Value | Override |
|---|---|---|
| `UNCERTAIN_PROMOTION_THRESHOLD` | **0.70** | `--uncertain-promotion-threshold` / env `UNCERTAIN_PROMOTION_THRESHOLD` |

The promotion step deliberately does **not** reuse the experimental `0.50`
grouping threshold. Only a similarity `>= 0.70` promotes an `UNCERTAIN` face.

## Rules implemented (enforced in `recover_uncertain_faces`)

1. Only `UNCERTAIN` faces (`original_status`) are considered.
2. Their embedding must already exist (they are embedded by design).
3. They are compared against `KEEP` faces **only**.
4. The `KEEP` face must come from a **different** image (same-image pairs skipped).
5. The strongest eligible `KEEP` match is selected per `UNCERTAIN` face.
6. Promotion happens only when `similarity >= UNCERTAIN_PROMOTION_THRESHOLD`.
7. Otherwise the face remains `UNCERTAIN`.
8. `FILTERED` faces never receive embeddings and never participate.
9. No existing filtering threshold is modified.
10. Both `original_status` and `final_status` are preserved on every record.

## Command used

```bash
.venv/bin/python ml/inference/match_faces.py --dir ml/datasets/test --glob 'event*' --det-size 640
```

## Actual results (10 event photos, det-size 640)

| Metric | Value |
|---|---|
| Photos processed | 10 |
| Faces detected | 37 |
| Faces filtered | 10 |
| `KEEP` faces | 26 |
| `UNCERTAIN` faces | 1 |
| Embeddings generated | 27 |
| Uncertain faces examined | 1 |
| Uncertain faces promoted | **0** |
| Uncertain faces remaining | **1** |
| Same-image matches used | 0 (PASS) |
| FILTERED faces with an embedding | 0 (PASS) |

The single `UNCERTAIN` face did **not** reach the 0.70 promotion threshold.
Its strongest eligible cross-image `KEEP` candidate was:

```
event04.JPG -> Face 2   (original_status: UNCERTAIN)
  uncertainty_reason: pose: extreme angle (yaw -79, pitch 34) on a large face
  strongest cross-image KEEP candidate: event01.jpeg -> Face 1
  similarity: 0.4417   (< promotion threshold 0.70)
  final_status: UNCERTAIN   (unchanged)
```

There were **no promotions** in this run, so there are no promotion scores to
list. `event04.JPG Face 2` is the strongly side-facing large face already noted
in §13/§14; its pose-degraded embedding (`0.4417`) sits far below 0.70, so the
recovery rule correctly leaves it `UNCERTAIN` rather than forcing a match.

## Machine-readable output

`data/output/confirmed_uncertain_matches.csv` — one row per examined
`UNCERTAIN` face (1 row + header in this run):

```
photo, face_id, original_status, uncertainty_reason,
matched_photo, matched_face_id, similarity, promotion_threshold, final_status
```

Actual file content:

```
photo,face_id,original_status,uncertainty_reason,matched_photo,matched_face_id,similarity,promotion_threshold,final_status
event04.JPG,2,UNCERTAIN,"pose: extreme angle (yaw -79, pitch 34) on a large face",event01.jpeg,1,0.441662,0.70,UNCERTAIN
```

`matched_photo` / `matched_face_id` / `similarity` record the strongest eligible
cross-image `KEEP` candidate even when it does **not** clear the threshold;
`final_status` is what the recovery step decided. When a face is promoted,
`final_status` becomes `CONFIRMED_FROM_MATCH` and `matched_*` is the `KEEP` face
that caused the promotion.

---

## Machine-readable output

`data/output/face_matching_results.csv` (27 rows + header) — one row per kept
face, no names:

```
group_id, image_filename, face_index, similarity_to_group_reference,
detection_confidence, relative_face_area, quality_score, yaw, pitch
```

`similarity_to_group_reference` is the cosine similarity of the face to its
group's reference face (the reference face itself is `1.0`). `quality_score` is
the Laplacian-variance blur score. Group IDs (`Group_1` … `Group_9`) are
temporary experimental labels, not person identities.

---

## Next stage: FINAL person clustering + anonymous Person IDs + Merge Sort

The experimental connected-component groups above are **not** promoted to the
final system (they over-merge through transitive chaining). The final stage uses
a centroid/representative-based clustering rule, generates anonymous Person IDs
dynamically, produces Person-Photo records, and orders them with the custom
Merge Sort (DAA). See
[`person_clustering_and_merge_sort.md`](person_clustering_and_merge_sort.md) for
the approach, the configurable `FINAL_CLUSTER_THRESHOLD` (0.60), multi-person
photo handling, UNCERTAIN handling and the Merge Sort complexity.

This document's matching/uncertain-recovery results remain the input those
stages consume; nothing above was changed.

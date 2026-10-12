# Final Person Clustering + Anonymous Person IDs + Custom Merge Sort

Technical documentation / development history for the FINAL stage of the
photo-seggregation pipeline. This document is a technical record, not a polished
final report.

Built on top of the already-verified pipeline (detection → filtering →
embeddings → uncertain-face recovery); those stages were **not** modified.

```
Event photos
    -> Face Detection
    -> Face Filtering
    -> Face Embeddings
    -> Uncertain-face recovery (existing, unchanged)
    -> Final person clustering (centroid/representative)   <- this stage
    -> Anonymous Person IDs (Person #1, Person #2, ...)     <- this stage
    -> Person-Photo records                                 <- this stage
    -> Custom Merge Sort (DAA)                              <- this stage
    -> Person-wise photo galleries
```

Reproduce:

```bash
.venv/bin/python ml/inference/match_faces.py   --dir ml/datasets/test --glob 'event*' --det-size 640
.venv/bin/python run.py --dir ml/datasets/test --glob 'event*' --det-size 640
.venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```

## Application entry point (`run.py` / `app/main.py`)

The full pipeline is implemented **once** in the shared orchestrator
`app/processing/photo_processor.py`. Every entry point delegates to it, so the
application launcher and the ML-side runner cannot diverge:

* `python run.py` → `app/main.py:main` → `photo_processor.main`
* `python app/main.py` → same
* `python ml/inference/cluster_people.py` → thin wrapper over the same module
* `app/processing/photo_processor.run_pipeline(args)` → returns a `PipelineResult`
  for programmatic reuse.

`run.py` / `app/main.py` no longer raise `NotImplementedError`; they run
`detection → filtering → embeddings → uncertain recovery → clustering →
anonymous Person IDs → Person-Photo records → Merge Sort → galleries`,
write the output files and print the report. The Streamlit upload/naming UI
remains out of scope for this stage. Running against a directory with no
matching images prints an error and exits with code 1 (not a crash).

---

## 1. Final clustering approach

Representative/centroid-based **incremental** clustering
(`app/ai/face_cluster.py`):

1. Start with no clusters.
2. Process each usable face embedding in a deterministic order.
3. For every *eligible* cluster, compute the cosine similarity between the face
   and the cluster's representative (the L2-normalised mean of its members).
4. If the best similarity `>= FINAL_CLUSTER_THRESHOLD`, join that cluster.
5. Otherwise, start a new anonymous cluster.

The representative is re-normalised after every update so it stays unit-length
(embeddings from `buffalo_l` are already unit-norm; `compare_embeddings` computes
`dot(a,b)/(||a||·||b||)` defensively regardless).

## 2. Why simple connected components were NOT used as the final method

The earlier experimental grouping used connected components over cross-image
edges `>= 0.50`. That over-merges through **transitive chaining**:

```
A <-> B = 0.82   (edge kept)
B <-> C = 0.76   (edge kept)
A <-> C = 0.48   (weak, but A and C are still connected through B)
```

Single linkage puts A, B and C in one person even though A and C are barely
similar. The centroid rule is stricter (a face must be close to the cluster's
*central tendency*, not merely to one member), so chaining is reduced. This is
verified by `tests/test_face_clustering.py::test_centroid_does_not_chain_like_single_linkage`.

## 3. Final clustering threshold

| Parameter | Value | Where |
|---|---|---|
| `FINAL_CLUSTER_THRESHOLD` | **0.60** | `app/ai/face_cluster.py`, env `FINAL_CLUSTER_THRESHOLD`, CLI `--cluster-threshold` |

**Why 0.60 (and why it is NOT claimed to be optimal):** from the earlier
similarity experiment the number of cross-image edges falls `44 (0.50) → 36
(0.60) → 25 (0.65)`. The `0.50` value was only ever an experimental *display*
threshold. `0.60` keeps the clearly-separated matches while discarding most of
the noisy `0.40–0.55` band that contained the ambiguous pairs. It is a
**conservative initial value**, deliberately configurable, and must be validated
against manually verified groups before being frozen. It is not scientifically
optimal.

## 4. Why the threshold is configurable

Embeddings from a pretrained model have no principled universal cut-off — the
right value depends on the dataset, the pose distribution and the tolerance for
false merges vs false splits. Hard-coding one number would make the result look
authoritative when it is not. The threshold is therefore overridable
(`--cluster-threshold` / `FINAL_CLUSTER_THRESHOLD`), and every other stage
(grouping, uncertain recovery) keeps its own separate threshold.

## 5. How anonymous Person IDs are generated

`app/processing/person_manager.py`. IDs are formatted from a positional index
over the clusters, **in dynamic cluster-creation order**:

```
Person #1, Person #2, Person #3, ...
```

No names, no hardcoded identities, no filename-as-identity, and no assumption
that `Person #1` is any particular real person. There is exactly one formatting
site (`make_person_id`). A human can later map an anonymous ID to a name via the
(not-yet-built) web UI. `tests` and the run report check that every ID matches
`^Person #\d+$`.

## 6. How multi-person photos are represented

Internally there is a **flat list of Person-Photo records**
(`app/processing/record_generator.py`), one per (person, photo, face) link. A
photo is never assigned to only one person:

```
(Person #1, event05.jpeg, face 0)
(Person #3, event05.jpeg, face 1)
(Person #5, event05.jpeg, face 3)
```

Grouping into person-wise galleries happens *after* the Merge Sort by
traversing the sorted records and de-duplicating photos per person. This is
covered by `tests/test_pipeline.py::test_multi_person_photo_is_supported` and
`::test_photo_appears_once_per_person_not_exclusively`.

## 7. How UNCERTAIN faces are handled

The existing recovery mechanism is preserved unchanged
(`UNCERTAIN_PROMOTION_THRESHOLD = 0.70`). Effective statuses entering this stage:

* **KEEP** → can participate.
* **CONFIRMED_FROM_MATCH** → can participate (recovered by a strong cross-image
  KEEP match).
* **UNCERTAIN** (not promoted) → does **not** automatically become a person; it
  is excluded from clustering but **preserved** in `person_clusters.json` under
  `unresolved_uncertain`.
* **FILTERED** → never participates (it never received an embedding).

## 8. Same-photo rule

Two faces from the same photo are never used as evidence that they are the same
person. When evaluating a face from photo *P*, a cluster is eligible only if it
already contains a member from a **different** photo, and its representative is
recomputed **excluding** photo *P*. A cluster holding only same-photo members is
skipped. `ClusteringResult.same_photo_comparisons` reports the invariant (0).

## 9. Merge Sort integration and complexity

`app/algorithms/merge_sort.py` is a custom recursive Merge Sort (the DAA
component). It sorts the Person-Photo records by Person ID; traversal then builds
the person-wise galleries.

* Python's `sorted()` / `list.sort()` are **not** used in the algorithm.
* Ordering is numeric-aware: `Person #2` sorts before `Person #10`.
* The sort is stable, so photos keep their processing order within a person.
* **Time complexity: `O(n log n)`**, **auxiliary space: `O(n)`**, where `n` is
  the number of Person-Photo records.
* Merge Sort **does not** identify faces — it only orders records the AI stage
  already produced.

Covered by `tests/test_merge_sort.py` (including an independent oracle check and
numeric ordering beyond `#9`).

---

## Actual run (10 event photos, det-size 640)

Command:

```bash
.venv/bin/python run.py --dir ml/datasets/test --glob 'event*' --det-size 640
```

| Metric | Value |
|---|---|
| Photos processed | 10 |
| Faces detected | 37 |
| Faces retained (embeddings) | 27 |
| KEEP | 26 |
| CONFIRMED_FROM_MATCH | 0 |
| UNCERTAIN (unresolved) | 1 |
| Usable faces (KEEP + confirmed) | 26 |
| **Final Person IDs** | **8** |
| — multi-photo persons | 5 |
| — accepted singleton persons | 3 |
| — rejected singleton faces | 3 |
| Person-Photo records | 23 |
| Same-photo comparisons | 0 (PASS) |
| Filtered faces with embeddings | 0 (PASS) |
| Uncertain recovery | examined 1, promoted 0, remaining 1 |
| Merge Sort | 23 records, `O(n log n)`, Status PASS |

### Person-wise galleries (actual)

```
Person #1 : event01, event05, event07, event08, event09
Person #2 : event01, event02, event03, event05, event06
Person #3 : event02, event05
Person #4 : event02, event04.JPG
Person #5 : event02, event04.JPG, event05, event06, event07, event08
Person #6 : event04.JPG        (accepted singleton, Face 0)
Person #7 : event04_9          (accepted singleton, Face 0)
Person #8 : event04_9          (accepted singleton, Face 1)
```

### Multi-person photos (actual)

`event01.jpeg`, `event05.jpeg`, `event07.jpeg`, `event08.jpeg`, `event02.jpeg`,
`event06.jpeg` and `event04.JPG` belong to two or more people, and
`event04_9.jpeg` holds Person #7 and Person #8. `event03.jpeg` and `event09.jpeg`
now hold a single person because their only singleton face was rejected by the
strict gate (see §10). No photo is ever forced into a single person.

### Questionable clusters (reported, not hidden)

Comparing against the experimental connected-component groups:

* The centroid rule splits weakly-linked faces off the large clusters
  (`event09.jpeg Face 1` ≈0.557, `event04.JPG Face 0` ≈0.577, `event05.jpeg
  Face 5` ≈0.562). Since these three end up as singletons, the strict singleton
  gate then decides their fate: all three are **rejected** (see §10), so they do
  not become persons at all.
* Person #5 (6 photos) and Person #1 (5 photos) are the largest clusters and
  could still be over-merged; only manual inspection can confirm.

These are exactly the cases flagged for human verification; no result was
adjusted to look cleaner.

## 10. Strict singleton person qualification

A person who appears in only one photo must still be allowed to become a
`Person #N`, but with **no cross-photo evidence** supporting the cluster the
quality bar is stricter. After clustering, `app/ai/singleton_qualifier.py`:

* leaves a **multi-photo cluster completely untouched** (its individual faces
  are never re-checked — the cross-photo evidence is what makes it reliable);
* applies a strict gate to a **singleton cluster** (all faces from one photo).

Gate (all configurable via env):

| Signal | Threshold | Source |
|---|---|---|
| `SINGLETON_MIN_RELATIVE_FACE_AREA` | 0.002 | resolution-independent relative area |
| `SINGLETON_MIN_DETECTION_CONFIDENCE` | 0.80 | detection confidence |
| `SINGLETON_MIN_BLUR_SCORE` | 80.0 | existing Laplacian-variance blur |
| `SINGLETON_MAX_YAW` | 45° | existing pose estimate |
| `SINGLETON_MAX_PITCH` | 30° | existing pose estimate |

**Why the area floor is 0.002, not the suggested 0.01:** running with `0.01`
rejected **all six** singletons, including three that are frontal, sharp and
high-confidence (`event04.JPG Face 0` = 0.0084, `event04_9 Face 0` = 0.0023 and
`Face 1` = 0.0029) purely because those photos are ~12 MP. The floor is therefore
derived from the **existing** pipeline floor (`MIN_RELATIVE_FACE_AREA` = 0.001)
as `2 × 0.001` — strictly stricter than what the verified pipeline already
accepts, without discarding real foreground faces. No existing filtering
threshold was changed.

**Face visibility:** `buffalo_l` exposes **no** direct face-visibility / occlusion
attribute, so none is invented. Visibility is approximated only by the available
size, confidence, blur and pose signals; there is no true visibility metric.

A singleton that fails is labelled `SINGLETON_REJECTED` with every applicable
reason (`face_too_small`, `low_detection_confidence`, `excessive_blur`,
`excessive_yaw`, `excessive_pitch`, plus `multiple_quality_failures` when more
than one fires). It is **not** deleted (photo and face data stay on disk) and it
does **not** get a `Person #N`, so it appears in neither `person_clusters.json`
nor `person_photo_records.csv` — only in `singleton_validation.csv`.

### Actual singleton qualification (real measurements)

`Singleton clusters examined: 6 · Accepted: 3 · Rejected: 3`

```
event03.jpeg   Face 2  area 0.012934  conf 0.8142  blur 217.78  yaw  49.23  pitch -17.92  REJECTED (excessive_yaw)
event04.JPG    Face 0  area 0.008447  conf 0.8719  blur 209.60  yaw   3.95  pitch   0.08  ACCEPTED_SINGLETON
event04_9.jpeg Face 0  area 0.002311  conf 0.8633  blur 107.07  yaw   0.69  pitch -12.36  ACCEPTED_SINGLETON
event04_9.jpeg Face 1  area 0.002884  conf 0.8116  blur 246.32  yaw  -6.88  pitch  -5.75  ACCEPTED_SINGLETON
event05.jpeg   Face 5  area 0.007487  conf 0.7389  blur 1727.81 yaw   6.06  pitch  12.32  REJECTED (low_detection_confidence)
event09.jpeg   Face 1  area 0.088968  conf 0.6156  blur 218.71  yaw -30.87  pitch -11.97  REJECTED (low_detection_confidence)
```

Note `event09.jpeg Face 1` is a large face (8.9% of the image) but was rejected
on detection confidence (0.616 < 0.80) — a real measurement, not a fabricated
label. `event03.jpeg Face 2` was rejected on yaw (49.23° > 45°), i.e. genuinely
turned away, even though area/confidence/blur pass. Remaining `UNCERTAIN` faces
never reach this stage at all.

## 11. Known limitations

1. **Centroid drift.** The representative is the normalised mean of members.
   Adding a borderline face nudges the centroid toward it; with small clusters a
   single questionable face can move the decision. Re-normalising does not
   prevent drift. Documented rather than hidden.
2. **Order dependence.** Incremental clustering depends on the face processing
   order. The runner uses a deterministic order (photo order, face index), but a
   different order can change borderline merges.
3. **Threshold sensitivity.** `0.60` is a conservative starting point, not an
   optimum. Set/verify with `--cluster-threshold`.
4. **Unresolved UNCERTAIN faces** do not form persons and remain in
   `unresolved_uncertain`; they require manual review or a matching KEEP face to
   be recovered (and even then, only via the existing `0.70` promotion rule).
5. **No real identities.** Everything is anonymous; names are out of scope for
   this stage, as is the web UI.
6. **Verified on a small set.** 10 photos / 26 usable faces is a small sample.

## Machine-readable outputs

* `data/output/person_clusters.json` — `persons` (id + photos) plus
  `unresolved_uncertain` and metadata (thresholds, det-size, photo count).
* `data/output/person_photo_records.csv` — `person_id, photo, face_id,
  similarity, status` (Merge-Sorted by Person ID).
* `data/output/confirmed_uncertain_matches.csv` — owned by the existing uncertain
  recovery mechanism; re-generated by the same code, not replaced.
* `data/output/singleton_validation.csv` — one row per singleton face with
  `photo, face_id, relative_face_area, detection_confidence, blur_score, yaw,
  pitch, singleton_status, rejection_reason`.

## Files

* Changed/implemented: `app/algorithms/merge_sort.py`, `app/ai/face_matcher.py`,
  `app/ai/face_cluster.py`, `app/processing/person_manager.py`,
  `app/processing/record_generator.py`, `app/ai/singleton_qualifier.py`,
  `tests/test_merge_sort.py`, `tests/test_face_clustering.py`,
  `tests/test_pipeline.py`, `tests/test_singleton_qualifier.py`.
* Application entry point wired: `app/processing/photo_processor.py`,
  `app/main.py`, `run.py`; `ml/inference/cluster_people.py` is now a thin
  wrapper over the shared orchestrator.
* Not touched: `ml/inference/predict.py`, `ml/inference/match_faces.py`
  (uncertain recovery preserved).

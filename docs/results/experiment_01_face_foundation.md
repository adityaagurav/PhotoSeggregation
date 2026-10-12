# Experiment 01 — ML Foundation (Detection + Embeddings)

**Date:** 2026-10-01
**Scope:** Verify that a pretrained face model runs on this machine and can
detect faces and produce embeddings. Clustering, naming, Merge Sort and the web
UI are deliberately **not** part of this experiment.

## Environment

| Item | Value |
| --- | --- |
| Operating system | macOS (Darwin 25.6.0) |
| Architecture | arm64 (Apple Silicon) |
| Python | 3.13.3 (`.venv`) |
| InsightFace | 2.0 |
| OpenCV (`opencv-python`) | 5.0.0.93 |
| NumPy | 2.5.3 |
| ONNX Runtime | 1.30.0 |
| Available ORT providers | `CoreMLExecutionProvider`, `AzureExecutionProvider`, `CPUExecutionProvider` |
| **Provider used** | **`CPUExecutionProvider`** |
| Model pack | `buffalo_l` (pretrained, no training performed) |
| Model cache | `~/.insightface/models/buffalo_l` (outside the repository) |

No CUDA / NVIDIA packages are installed or required.

## Test images

Placed locally in `ml/datasets/test/` (that folder's contents are gitignored,
so no personal imagery is ever committed):

| File | Content | Size |
| --- | --- | --- |
| `single_face.png` | single face | 112 × 112 |
| `group_faces.jpg` | group / multi-face | 1280 × 886 |

These two files were copied from the sample images bundled inside the installed
InsightFace package; nothing was downloaded from the internet.

> To add your own images, drop 1 single-face and 1 multi-face photo into
> `ml/datasets/test/` and pass their paths to the command below.

## Results

| Check | Result |
| --- | --- |
| InsightFace import | ✅ |
| Buffalo_L load | ✅ (auto-downloaded on first run) |
| Test image processed | ✅ |
| Single-face detection | ✅ 1 face in `single_face.png` |
| Multi-face detection | ✅ 6 faces in `group_faces.jpg` |
| Embeddings generated | ✅ |
| Embedding dimensionality | **512** (one `(512,)` vector per detected face) |
| Detection confidence range | 0.866 – 0.920 |
| Similarity comparison | ✅ cosine similarity |

### Timing (CPU provider)

| Stage | Time |
| --- | --- |
| First model load (incl. one-time download of ~282 MB) | 116.75 s |
| Warm model load (cached weights) | 0.26 s |
| Single-face image inference | 0.14 s |
| 6-face image inference | 0.40 s |

### Similarity behaviour

Cosine similarity between embeddings, using a **configurable** threshold
(default `FACE_SIMILARITY_THRESHOLD = 0.5`, overridable via env var or the
`--threshold` CLI flag — never a magic number):

| Pair type | Mean | Min | Max |
| --- | --- | --- | --- |
| Same person (original photo vs. horizontally-mirrored copy of itself) | +0.9528 | +0.9311 | — |
| Different people (all pairs within the group photo) | +0.0323 | -0.0845 | +0.2134 |

Same-person scores (≈ 0.93–0.97) and different-person scores (≤ 0.21) are
cleanly separated, so a threshold near 0.5 is a reasonable starting point. The
exact value will be tuned experimentally when clustering is implemented.

## Errors / compatibility notes

- **None blocking.** One non-fatal warning appears during alignment:
  `FutureWarning: 'estimate' is deprecated ... use SimilarityTransform.from_estimate`
  from `insightface/utils/face_align.py`. It does not affect results.
- CoreML was available but not selected; CPU was chosen for reliability. CoreML
  can be tried later with `--provider CoreMLExecutionProvider`.

## Reproduce

```bash
# single-face image
.venv/bin/python ml/inference/run_model.py ml/datasets/test/single_face.png

# multi-face image + pairwise similarity
.venv/bin/python ml/inference/run_model.py ml/datasets/test/group_faces.jpg

# cross-image comparison
.venv/bin/python ml/inference/run_model.py ml/datasets/test/single_face.png ml/datasets/test/group_faces.jpg
```

## Conclusion

The AI foundation is verified: a pretrained model detects faces, produces
512-dimensional embeddings, and cosine similarity separates same- from
different-person faces. Ready for the next stage (clustering) on request.

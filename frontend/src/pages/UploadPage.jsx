import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import Navbar from "../components/Navbar";
import {
  IconArrowLeft,
  IconArrowRight,
  IconUpload,
  IconAlert,
  IconTrash,
  IconX,
} from "../components/icons";
import "./UploadPage.css";

const ACCEPT = ".jpg,.jpeg,.png,.webp,image/jpeg,image/png,image/webp";
const VALID_EXTENSIONS = ["jpg", "jpeg", "png", "webp"];
const VALID_MIME_TYPES = ["image/jpeg", "image/png", "image/webp"];

const API_BASE = "/api";
const POLL_INTERVAL_MS = 1000;
const POLL_MAX_MS = 20 * 60 * 1000;

const STEPS = [
  { n: "01", label: "Upload" },
  { n: "02", label: "Process" },
  { n: "03", label: "Results" },
];

// Same file = same name + size + last-modified time.
function fileKey(file) {
  return `${file.name}__${file.size}__${file.lastModified}`;
}

function isSupportedImage(file) {
  const type = (file.type || "").toLowerCase();
  const ext = file.name.toLowerCase().split(".").pop();
  return VALID_MIME_TYPES.includes(type) || VALID_EXTENSIONS.includes(ext);
}

function plural(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

async function extractError(response) {
  try {
    const data = await response.json();
    const detail = data && data.detail !== undefined ? data.detail : data;
    if (typeof detail === "string") return detail;
    if (detail && detail.message) {
      const rejected = Array.isArray(detail.rejected) ? detail.rejected : [];
      return [detail.message, ...rejected.map((r) => `${r.name}: ${r.reason}`)].join(" ");
    }
    return JSON.stringify(detail);
  } catch {
    return `Request failed (${response.status}).`;
  }
}

export default function UploadPage() {
  const [photos, setPhotos] = useState([]); // { id, key, file, url }
  const [messages, setMessages] = useState([]); // { type, text }
  const [dragging, setDragging] = useState(false);

  // Real backend run state (never fabricated).
  const [phase, setPhase] = useState("idle"); // idle | uploading | processing | done | error
  const [jobId, setJobId] = useState(null);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [elapsed, setElapsed] = useState(0);

  const inputRef = useRef(null);
  const dragDepth = useRef(0);
  const photosRef = useRef(photos);
  const aliveRef = useRef(true);

  useEffect(() => {
    photosRef.current = photos;
  }, [photos]);

  useEffect(() => {
    aliveRef.current = true;
    return () => {
      aliveRef.current = false;
      // Revoke every object URL when the page unmounts so previews don't leak.
      photosRef.current.forEach((photo) => URL.revokeObjectURL(photo.url));
    };
  }, []);

  // Genuine elapsed timer while a real request is in flight.
  useEffect(() => {
    if (phase !== "uploading" && phase !== "processing") return undefined;
    const started = Date.now();
    const id = setInterval(
      () => setElapsed(Math.floor((Date.now() - started) / 1000)),
      1000
    );
    return () => clearInterval(id);
  }, [phase]);

  const busy = phase === "uploading" || phase === "processing";

  function addFiles(fileList) {
    const incoming = Array.from(fileList || []);
    if (incoming.length === 0) return;

    const known = new Set(photos.map((photo) => photo.key));
    const additions = [];
    const rejected = [];
    const duplicates = [];

    for (const file of incoming) {
      if (!isSupportedImage(file)) {
        rejected.push(file.name);
        continue;
      }
      const key = fileKey(file);
      if (known.has(key)) {
        duplicates.push(file.name);
        continue;
      }
      known.add(key);
      additions.push({ key, file, url: URL.createObjectURL(file) });
    }

    if (additions.length > 0) {
      const stamp = Date.now();
      setPhotos((prev) => [
        ...prev,
        ...additions.map((addition, index) => ({
          ...addition,
          id: `${addition.key}#${stamp}-${index}`,
        })),
      ]);
    }

    const nextMessages = [];
    if (rejected.length > 0) {
      nextMessages.push({
        type: "error",
        text: `Unsupported file type: ${rejected.join(", ")}. Only JPG, JPEG, PNG, and WebP images are accepted.`,
      });
    }
    if (duplicates.length > 0) {
      nextMessages.push({
        type: "warn",
        text: `Skipped ${plural(duplicates.length, "duplicate file")} already added: ${duplicates.join(", ")}.`,
      });
    }
    setMessages(nextMessages);
  }

  function handleInputChange(event) {
    addFiles(event.target.files);
    // Reset so the same file can be picked again later (duplicates are still caught).
    event.target.value = "";
  }

  function removePhoto(id) {
    setPhotos((prev) => {
      const target = prev.find((photo) => photo.id === id);
      if (target) URL.revokeObjectURL(target.url);
      return prev.filter((photo) => photo.id !== id);
    });
  }

  function clearAll() {
    photos.forEach((photo) => URL.revokeObjectURL(photo.url));
    setPhotos([]);
    setMessages([]);
    setPhase("idle");
    setJobId(null);
    setResult(null);
    setError("");
  }

  function openPicker() {
    inputRef.current?.click();
  }

  // Drag and drop
  function handleDragEnter(event) {
    event.preventDefault();
    dragDepth.current += 1;
    setDragging(true);
  }

  function handleDragOver(event) {
    event.preventDefault();
    if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
  }

  function handleDragLeave(event) {
    event.preventDefault();
    dragDepth.current = Math.max(0, dragDepth.current - 1);
    if (dragDepth.current === 0) setDragging(false);
  }

  function handleDrop(event) {
    event.preventDefault();
    dragDepth.current = 0;
    setDragging(false);
    if (event.dataTransfer && event.dataTransfer.files.length > 0) {
      addFiles(event.dataTransfer.files);
    }
  }

  async function pollStatus(id) {
    const deadline = Date.now() + POLL_MAX_MS;
    while (Date.now() < deadline && aliveRef.current) {
      const response = await fetch(`${API_BASE}/status/${id}`);
      if (!response.ok) throw new Error(await extractError(response));
      const data = await response.json();
      if (data.status === "done" || data.status === "error") return data;
      await new Promise((resolve) => setTimeout(resolve, POLL_INTERVAL_MS));
    }
    if (!aliveRef.current) return null;
    throw new Error("Timed out waiting for the pipeline to finish.");
  }

  // Real flow: upload -> start pipeline -> poll real status -> show real results.
  async function handleProcess() {
    if (photos.length === 0 || busy) return;
    setResult(null);
    setError("");
    setJobId(null);
    setElapsed(0);
    setPhase("uploading");

    try {
      const form = new FormData();
      photos.forEach((photo) => form.append("files", photo.file, photo.file.name));
      const uploadResponse = await fetch(`${API_BASE}/upload`, {
        method: "POST",
        body: form,
      });
      if (!uploadResponse.ok) throw new Error(await extractError(uploadResponse));
      const uploadData = await uploadResponse.json();
      if (!aliveRef.current) return;

      setJobId(uploadData.job_id);
      if (Array.isArray(uploadData.rejected) && uploadData.rejected.length > 0) {
        setMessages(
          uploadData.rejected.map((item) => ({
            type: "warn",
            text: `Backend skipped ${item.name}: ${item.reason}`,
          }))
        );
      }

      setPhase("processing");
      const processResponse = await fetch(`${API_BASE}/process/${uploadData.job_id}`, {
        method: "POST",
      });
      if (!processResponse.ok) throw new Error(await extractError(processResponse));

      const finalStatus = await pollStatus(uploadData.job_id);
      if (finalStatus === null) return; // unmounted
      if (finalStatus.status === "error") {
        throw new Error(finalStatus.error || "The pipeline reported an error.");
      }
      setResult(finalStatus.result);
      setPhase("done");
    } catch (err) {
      if (!aliveRef.current) return;
      setError(err instanceof Error ? err.message : String(err));
      setPhase("error");
    }
  }

  const hasPhotos = photos.length > 0;
  const controlsDisabled = busy;

  return (
    <div className="page">
      <Navbar
        brandHref="/"
        links={[
          { label: "How It Works", href: "/#how" },
          { label: "About", href: "/#about" },
        ]}
      />

      <main className="upload-main">
        <div className="wrap">
          <Link className="back-link" to="/">
            <IconArrowLeft size={15} />
            Back to home
          </Link>

          <ol className="stepper" aria-label="Progress">
            {STEPS.map((step, index) => {
              const activeIndex =
                phase === "done" ? 2 : phase === "processing" || phase === "uploading" ? 1 : 0;
              return (
                <li
                  key={step.n}
                  className={
                    "stepper-item" +
                    (index === activeIndex ? " is-active" : "") +
                    (index < activeIndex ? " is-complete" : "")
                  }
                  aria-current={index === activeIndex ? "step" : undefined}
                >
                  <span className="stepper-num">{step.n}</span>
                  <span className="stepper-label">{step.label}</span>
                </li>
              );
            })}
          </ol>

          <div className="upload-head">
            <h1>Upload Event Photos</h1>
            <p className="upload-sub">
              Choose the photos from your event. We&rsquo;ll organize them into
              person-wise collections.
            </p>
          </div>

          <div
            className={"dropzone" + (dragging ? " is-dragging" : "")}
            onClick={busy ? undefined : openPicker}
            onDragEnter={handleDragEnter}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            role="button"
            tabIndex={0}
            aria-label="Drag and drop photos or browse files"
            onKeyDown={(event) => {
              if (!busy && (event.key === "Enter" || event.key === " ")) {
                event.preventDefault();
                openPicker();
              }
            }}
          >
            <input
              ref={inputRef}
              type="file"
              className="visually-hidden"
              multiple
              accept={ACCEPT}
              onChange={handleInputChange}
            />

            <span className="dropzone-icon" aria-hidden="true">
              <IconUpload size={26} />
            </span>
            <p className="dropzone-title">Drag and drop your photos here</p>
            <p className="dropzone-sub">or browse files</p>
            <button
              type="button"
              className="btn btn-secondary"
              disabled={busy}
              onClick={(event) => {
                event.stopPropagation();
                openPicker();
              }}
            >
              Browse Files
            </button>
            <p className="dropzone-hint">
              Select multiple JPG, JPEG, PNG, or WebP images.
            </p>
          </div>

          {messages.length > 0 && (
            <div className="messages">
              {messages.map((message, index) => (
                <p
                  key={index}
                  className={"message message-" + message.type}
                  role={message.type === "error" ? "alert" : "status"}
                >
                  <IconAlert size={15} />
                  <span>{message.text}</span>
                </p>
              ))}
            </div>
          )}

          <section className="selected" aria-label="Selected photos">
            <div className="selected-head">
              <div>
                <h2>Selected Photos</h2>
                <p className="selected-count">
                  {hasPhotos
                    ? `${plural(photos.length, "photo")} selected.`
                    : "No photos selected yet."}
                </p>
              </div>
              {hasPhotos && (
                <div className="selected-actions">
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    disabled={controlsDisabled}
                    onClick={openPicker}
                  >
                    Add More
                  </button>
                  <button
                    type="button"
                    className="btn btn-danger btn-sm"
                    disabled={controlsDisabled}
                    onClick={clearAll}
                  >
                    <IconTrash size={14} />
                    Clear All
                  </button>
                </div>
              )}
            </div>

            {hasPhotos ? (
              <ul className="thumbs-grid">
                {photos.map((photo) => (
                  <li className="thumb-card" key={photo.id}>
                    <div className="thumb-media">
                      <img src={photo.url} alt={photo.file.name} loading="lazy" />
                      <button
                        type="button"
                        className="thumb-remove"
                        aria-label={`Remove ${photo.file.name}`}
                        title="Remove"
                        disabled={controlsDisabled}
                        onClick={() => removePhoto(photo.id)}
                      >
                        <IconX size={13} />
                      </button>
                    </div>
                    <span className="thumb-name" title={photo.file.name}>
                      {photo.file.name}
                    </span>
                  </li>
                ))}
              </ul>
            ) : (
              <div className="empty-state">
                <span className="empty-icon" aria-hidden="true">
                  <IconUpload size={20} />
                </span>
                <p>No photos yet. Add images above to see live previews here.</p>
              </div>
            )}
          </section>

          <div className="process-bar">
            <div className="process-info">
              <span className="process-title">Ready when you are.</span>
              <span className="process-note">
                {hasPhotos
                  ? `${plural(photos.length, "image")} ready to process.`
                  : "Add at least one image to enable processing."}
              </span>
            </div>
            <button
              type="button"
              className="btn btn-primary"
              onClick={handleProcess}
              disabled={!hasPhotos || busy}
              aria-busy={busy}
            >
              {busy ? (
                <>
                  <span className="spinner" aria-hidden="true" />
                  {phase === "uploading" ? "Uploading…" : "Processing…"}
                </>
              ) : (
                <>
                  Process Photos
                  <IconArrowRight size={15} />
                </>
              )}
            </button>
          </div>

          {phase === "uploading" && (
            <div className="run-panel" role="status">
              <span className="spinner" aria-hidden="true" />
              <div>
                <p className="run-title">Uploading {plural(photos.length, "photo")}…</p>
                <p className="run-note">Sending the originals to the local backend.</p>
              </div>
            </div>
          )}

          {phase === "processing" && (
            <div className="run-panel" role="status">
              <span className="spinner" aria-hidden="true" />
              <div>
                <p className="run-title">
                  Running the AI + DAA pipeline&hellip; ({elapsed}s)
                </p>
                <p className="run-note">
                  Job <code>{jobId}</code> &middot; detection &rarr; filtering &rarr;
                  embeddings &rarr; clustering &rarr; anonymous Person IDs &rarr; Merge
                  Sort.
                </p>
              </div>
            </div>
          )}

          {phase === "error" && (
            <div className="run-panel run-panel-error" role="alert">
              <IconAlert size={16} />
              <div>
                <p className="run-title">Processing failed</p>
                <p className="run-note">{error}</p>
              </div>
            </div>
          )}

          {phase === "done" && result && (
            <section className="results" aria-label="Processing results">
              <div className="results-head">
                <h2>Processing Results</h2>
                <div className="results-head-actions">
                  <span className="results-job">
                    Job <code>{result.job_id}</code>
                  </span>
                  <Link
                    className="btn btn-primary btn-sm"
                    to={`/results/${result.job_id}`}
                  >
                    View Full Results
                    <IconArrowRight size={14} />
                  </Link>
                </div>
              </div>

              <dl className="run-grid">
                <div className="run-stat">
                  <dt>Photos uploaded</dt>
                  <dd>{result.photos_uploaded}</dd>
                </div>
                <div className="run-stat">
                  <dt>Photos processed</dt>
                  <dd>{result.photos_processed}</dd>
                </div>
                <div className="run-stat">
                  <dt>Faces detected</dt>
                  <dd>{result.faces_detected}</dd>
                </div>
                <div className="run-stat">
                  <dt>Faces retained</dt>
                  <dd>{result.faces_retained}</dd>
                </div>
                <div className="run-stat">
                  <dt>People identified</dt>
                  <dd>{result.people_identified}</dd>
                </div>
                <div className="run-stat">
                  <dt>Unresolved uncertain</dt>
                  <dd>{result.unresolved_uncertain_faces}</dd>
                </div>
                <div className="run-stat">
                  <dt>Person–Photo records</dt>
                  <dd>{result.person_photo_records}</dd>
                </div>
                <div className="run-stat">
                  <dt>Merge Sort</dt>
                  <dd>
                    {result.merge_sort.ordered ? "PASS" : "FAIL"}
                    <span className="run-stat-sub">
                      {" "}
                      · {result.merge_sort.records} records ·{" "}
                      {result.merge_sort.time_ms} ms
                    </span>
                  </dd>
                </div>
              </dl>

              <div className="run-outputs">
                <span className="run-outputs-label">Outputs</span>
                <ul>
                  <li>{result.outputs.person_clusters_json}</li>
                  <li>{result.outputs.person_photo_records_csv}</li>
                  <li>{result.outputs.confirmed_uncertain_matches_csv}</li>
                  <li>{result.outputs.singleton_validation_csv}</li>
                </ul>
              </div>
            </section>
          )}
        </div>
      </main>
    </div>
  );
}

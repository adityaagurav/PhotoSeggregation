import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  IconAlert,
  IconChevronRight,
  IconFolder,
  IconGrid,
  IconImage,
  IconLayers,
  IconList,
  IconRefresh,
  IconSearch,
  IconUpload,
  IconX,
} from "../components/icons";
import "./ResultsPage.css";

const API_BASE = "/api";
const POLL_INTERVAL_MS = 1500;
const POLL_MAX_MS = 20 * 60 * 1000;

const SECTIONS = [
  { id: "all", label: "All Results", Icon: IconLayers },
  { id: "event", label: "Event Photos", Icon: IconImage },
  { id: "uncertain", label: "Uncertain Faces", Icon: IconAlert },
];

const SECTION_LABELS = {
  all: "All Results",
  event: "Event Photos",
  uncertain: "Uncertain Faces",
};

function formatStat(value) {
  return value === null || value === undefined ? "—" : value;
}

async function extractError(response) {
  try {
    const data = await response.json();
    const detail = data && data.detail !== undefined ? data.detail : data;
    if (typeof detail === "string") return detail;
    if (detail && typeof detail === "object") {
      const parts = [];
      if (detail.message) parts.push(detail.message);
      if (detail.error) parts.push(detail.error);
      if (parts.length > 0) return parts.join(" ");
      return JSON.stringify(detail);
    }
    return JSON.stringify(detail);
  } catch {
    return `Request failed (${response.status}).`;
  }
}

function photoUrl(jobId, filename) {
  return `${API_BASE}/photos/${encodeURIComponent(jobId)}/${encodeURIComponent(
    filename
  )}`;
}

function plural(count, word) {
  return `${count} ${word}${count === 1 ? "" : "s"}`;
}

// One original uploaded photo, served byte-for-byte from the backend.
function PhotoTile({ jobId, filename }) {
  const [failed, setFailed] = useState(false);
  const src = photoUrl(jobId, filename);

  return (
    <figure className="ps-photo">
      <a
        className="ps-photo-thumb"
        href={src}
        target="_blank"
        rel="noreferrer"
        title={`Open original: ${filename}`}
      >
        {failed ? (
          <span className="ps-photo-fallback">
            <IconImage size={20} />
            <span>Unavailable</span>
          </span>
        ) : (
          <img
            src={src}
            alt={filename}
            loading="lazy"
            onError={() => setFailed(true)}
          />
        )}
      </a>
      <figcaption className="ps-photo-name" title={filename}>
        {filename}
      </figcaption>
    </figure>
  );
}

function PhotoRow({ jobId, filename }) {
  const [failed, setFailed] = useState(false);
  const src = photoUrl(jobId, filename);

  return (
    <li>
      <a className="ps-row" href={src} target="_blank" rel="noreferrer">
        <span className="ps-row-thumb">
          {failed ? (
            <IconImage size={15} />
          ) : (
            <img
              src={src}
              alt={filename}
              loading="lazy"
              onError={() => setFailed(true)}
            />
          )}
        </span>
        <span className="ps-row-name">{filename}</span>
        <IconChevronRight size={15} />
      </a>
    </li>
  );
}

// A person's representative face crop. The blue folder icon is always the
// fallback: when no crop is available (or the image fails to load) the card
// simply shows the folder icon, so a missing preview can never break the page.
function PersonThumb({ jobId, person, variant }) {
  const [failed, setFailed] = useState(false);
  const file =
    person && person.preview && person.preview.available ? person.preview.file : null;
  const src = file
    ? `${API_BASE}/previews/${encodeURIComponent(jobId)}/${encodeURIComponent(
        file
      )}`
    : null;
  const showFace = Boolean(src) && !failed;
  const image = showFace ? (
    <img src={src} alt="" loading="lazy" onError={() => setFailed(true)} />
  ) : null;

  if (variant === "row") {
    return showFace ? (
      <span className="ps-row-face">{image}</span>
    ) : (
      <IconFolder size={22} />
    );
  }

  return (
    <>
      <span className="ps-folder-icon" aria-hidden="true">
        <IconFolder size={66} />
      </span>
      {showFace && <span className="ps-folder-face">{image}</span>}
    </>
  );
}

function StatChip({ label, value }) {
  return (
    <span className="ps-chip">
      <span className="ps-chip-k">{label}</span>
      <span className="ps-chip-v">{formatStat(value)}</span>
    </span>
  );
}

function EmptyState({ Icon, title, note }) {
  return (
    <div className="ps-empty">
      <span className="ps-empty-icon" aria-hidden="true">
        <Icon size={22} />
      </span>
      <p className="ps-empty-title">{title}</p>
      {note && <p className="ps-empty-note">{note}</p>}
    </div>
  );
}

export default function ResultsPage() {
  const { jobId } = useParams();
  const aliveRef = useRef(true);

  const [phase, setPhase] = useState("loading"); // loading | processing | done | error
  const [jobStatus, setJobStatus] = useState("processing");
  const [data, setData] = useState(null);
  const [error, setError] = useState("");
  const [errorCode, setErrorCode] = useState(null);
  const [reloadKey, setReloadKey] = useState(0);

  // Explorer UI state.
  const [view, setView] = useState("all");
  const [selectedPersonId, setSelectedPersonId] = useState(null);
  const [query, setQuery] = useState("");
  const [layout, setLayout] = useState("grid");

  // Reset when the job id in the URL changes (documented "adjust state during
  // render" pattern) so the effect below only ever performs async work.
  const [prevJobId, setPrevJobId] = useState(jobId);
  if (jobId !== prevJobId) {
    setPrevJobId(jobId);
    setPhase("loading");
    setJobStatus("processing");
    setData(null);
    setError("");
    setErrorCode(null);
    setView("all");
    setSelectedPersonId(null);
    setQuery("");
  }

  useEffect(() => {
    aliveRef.current = true;
    let timer = null;
    const deadline = Date.now() + POLL_MAX_MS;

    async function load() {
      if (!jobId) {
        setError("No job id was provided in the URL.");
        setErrorCode(404);
        setPhase("error");
        return;
      }
      try {
        const response = await fetch(
          `${API_BASE}/results/${encodeURIComponent(jobId)}`
        );
        if (!aliveRef.current) return;

        if (response.status === 202) {
          const info = await response.json().catch(() => null);
          setJobStatus(
            (info && info.detail && info.detail.status) || "processing"
          );
          setPhase("processing");
          if (Date.now() < deadline) {
            timer = setTimeout(load, POLL_INTERVAL_MS);
          } else {
            setError("Timed out waiting for the results.");
            setErrorCode(202);
            setPhase("error");
          }
          return;
        }

        if (!response.ok) {
          setError(await extractError(response));
          setErrorCode(response.status);
          setPhase("error");
          return;
        }

        const payload = await response.json();
        setData(payload);
        setPhase("done");
      } catch (err) {
        if (!aliveRef.current) return;
        setError(err instanceof Error ? err.message : String(err));
        setErrorCode(null);
        setPhase("error");
      }
    }

    load();

    return () => {
      aliveRef.current = false;
      if (timer) clearTimeout(timer);
    };
  }, [jobId, reloadKey]);

  const persons = useMemo(
    () => (data && Array.isArray(data.persons) ? data.persons : []),
    [data]
  );
  const uncertain = useMemo(
    () =>
      data && Array.isArray(data.unresolved_uncertain)
        ? data.unresolved_uncertain
        : [],
    [data]
  );

  // Every uploaded original if the backend reports it; otherwise fall back to
  // the union of the photos that were associated with a person.
  const allPhotos = useMemo(() => {
    if (data && Array.isArray(data.photos) && data.photos.length > 0) {
      return data.photos;
    }
    const seen = new Set();
    persons.forEach((person) =>
      (person.photos || []).forEach((name) => seen.add(name))
    );
    return Array.from(seen).sort((a, b) => a.localeCompare(b));
  }, [data, persons]);

  const ready = phase === "done" && Boolean(data);
  const selectedPerson =
    persons.find((person) => person.person_id === selectedPersonId) || null;

  const q = query.trim().toLowerCase();
  const filteredPersons = q
    ? persons.filter(
        (person) =>
          (person.person_id || "").toLowerCase().includes(q) ||
          (person.photos || []).some((name) =>
            name.toLowerCase().includes(q)
          )
      )
    : persons;
  const filteredPhotos = q
    ? allPhotos.filter((name) => name.toLowerCase().includes(q))
    : allPhotos;
  const filteredUncertain = q
    ? uncertain.filter((item) =>
        `${item.photo || ""} ${item.reason || ""} ${item.final_status || ""} ${
          item.face_id ?? ""
        }`
          .toLowerCase()
          .includes(q)
      )
    : uncertain;

  const counts = ready
    ? { all: persons.length, event: allPhotos.length, uncertain: uncertain.length }
    : null;

  const mergeSort = data ? data.merge_sort || {} : {};
  const ordered = mergeSort.ordered;
  const verdict =
    ordered === true ? "PASS" : ordered === false ? "FAIL" : "Not verified";
  const verdictClass =
    ordered === true ? "is-pass" : ordered === false ? "is-fail" : "is-unknown";
  const daaNote =
    ordered === true
      ? "Verified: Person–Photo records are in ascending Person ID order in person_photo_records.csv."
      : ordered === false
        ? "The records are NOT in ascending Person ID order — the sort did not succeed."
        : "The backend could not verify sorted order from the records for this job.";

  const phaseLabel = {
    loading: "Loading",
    processing: "Processing",
    done: "Ready",
    error: "Failed",
  }[phase];

  function selectView(next) {
    setView(next);
    setSelectedPersonId(null);
    setQuery("");
  }

  function togglePerson(personId) {
    setSelectedPersonId((current) => (current === personId ? null : personId));
  }

  const title =
    view === "all" && selectedPerson
      ? selectedPerson.person_id
      : SECTION_LABELS[view];

  const selectedCount = selectedPerson
    ? (selectedPerson.photos || []).length
    : 0;

  let subtitleDetail;
  if (view === "all") {
    subtitleDetail = selectedPerson
      ? plural(selectedCount, "photo")
      : plural(persons.length, "person");
  } else if (view === "event") {
    subtitleDetail = plural(allPhotos.length, "photo");
  } else {
    subtitleDetail = plural(uncertain.length, "uncertain face");
  }

  return (
    <div className="ps-app">
      <aside className="ps-sidebar">
        <Link className="ps-brand" to="/">
          <span className="ps-brand-mark" aria-hidden="true">
            <IconFolder size={18} />
          </span>
          <span className="ps-brand-name">PhotoSegregator</span>
        </Link>

        <nav className="ps-nav" aria-label="Library">
          <p className="ps-nav-title">Library</p>
          {SECTIONS.map(({ id, label, Icon }) => (
            <button
              key={id}
              type="button"
              className={"ps-nav-item" + (view === id ? " is-active" : "")}
              onClick={() => selectView(id)}
              disabled={!ready}
              aria-current={view === id ? "page" : undefined}
            >
              <Icon size={16} />
              <span className="ps-nav-label">{label}</span>
              {counts && <span className="ps-nav-count">{counts[id]}</span>}
            </button>
          ))}
        </nav>

        <div className="ps-sidebar-foot">
          <div className="ps-foot-event">
            <span className="ps-foot-k">Event</span>
            <code className="ps-foot-id">{jobId || "—"}</code>
          </div>
          <Link className="ps-foot-link" to="/upload">
            <IconUpload size={14} />
            New upload
          </Link>
        </div>
      </aside>

      <main className="ps-main">
        <header className="ps-toolbar">
          <div className="ps-toolbar-left">
            <h1 className="ps-title">{title}</h1>
            <p className="ps-sub">
              Event <code className="ps-sub-id">{jobId || "—"}</code>
              <span aria-hidden="true"> · </span>
              {subtitleDetail}
            </p>
          </div>

          <div className="ps-toolbar-right">
            <label className="ps-search">
              <IconSearch size={15} />
              <input
                type="search"
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Search people or photos"
                disabled={!ready}
                aria-label="Search people or photos"
              />
              {query && (
                <button
                  type="button"
                  className="ps-search-clear"
                  onClick={() => setQuery("")}
                  aria-label="Clear search"
                >
                  <IconX size={13} />
                </button>
              )}
            </label>

            <div className="ps-segment" role="group" aria-label="View mode">
              <button
                type="button"
                className={layout === "grid" ? "is-active" : ""}
                onClick={() => setLayout("grid")}
                disabled={!ready}
                aria-pressed={layout === "grid"}
                title="Grid view"
              >
                <IconGrid size={15} />
              </button>
              <button
                type="button"
                className={layout === "list" ? "is-active" : ""}
                onClick={() => setLayout("list")}
                disabled={!ready}
                aria-pressed={layout === "list"}
                title="List view"
              >
                <IconList size={15} />
              </button>
            </div>

            <button
              type="button"
              className="ps-iconbtn"
              onClick={() => setReloadKey((key) => key + 1)}
              title="Refresh results"
              aria-label="Refresh results"
            >
              <IconRefresh size={15} />
            </button>
          </div>
        </header>

        <div className="ps-statusbar">
          <span className={"ps-pill is-" + phase}>
            <span className="ps-pill-dot" aria-hidden="true" />
            {phaseLabel}
          </span>

          {ready && (
            <span className={"ps-verify " + verdictClass} title={daaNote}>
              Merge Sort {verdict}
              {mergeSort.records !== null && mergeSort.records !== undefined && (
                <span className="ps-verify-sep">·</span>
              )}
              {mergeSort.records !== null && mergeSort.records !== undefined && (
                <span>{mergeSort.records} records</span>
              )}
            </span>
          )}

          {ready && (
            <div className="ps-chips">
              <StatChip label="Photos" value={data.photos_processed} />
              <StatChip label="Faces" value={data.faces_detected} />
              <StatChip label="Retained" value={data.faces_retained} />
              <StatChip label="People" value={data.people_identified} />
              <StatChip label="Uncertain" value={data.unresolved_uncertain_faces} />
              <StatChip label="Records" value={data.person_photo_records} />
            </div>
          )}
        </div>

        <div className="ps-content">
          {phase === "loading" && (
            <div className="ps-state" role="status">
              <span className="ps-spinner" aria-hidden="true" />
              <p className="ps-state-title">Loading results…</p>
              <p className="ps-state-note">Reading the job&rsquo;s result files.</p>
            </div>
          )}

          {phase === "processing" && (
            <div className="ps-state" role="status">
              <span className="ps-spinner" aria-hidden="true" />
              <p className="ps-state-title">The pipeline is still running…</p>
              <p className="ps-state-note">
                Status: {jobStatus}. This page updates automatically.
              </p>
            </div>
          )}

          {phase === "error" && (
            <div className="ps-state ps-state-error" role="alert">
              <span className="ps-state-icon" aria-hidden="true">
                <IconAlert size={22} />
              </span>
              <p className="ps-state-title">
                {errorCode === 404
                  ? "Unknown event"
                  : errorCode === 500
                    ? "This event could not be loaded"
                    : "Could not load results"}
              </p>
              <p className="ps-state-note">{error}</p>
              <div className="ps-state-actions">
                <Link className="ps-btn ps-btn-primary" to="/upload">
                  <IconUpload size={15} />
                  Upload photos
                </Link>
                <Link className="ps-btn" to="/">
                  Back to home
                </Link>
              </div>
            </div>
          )}

          {ready && view === "all" && (
            <>
              {filteredPersons.length === 0 ? (
                <EmptyState
                  Icon={IconFolder}
                  title={
                    persons.length === 0
                      ? "No people were identified in this run."
                      : `No folders match “${query}”.`
                  }
                  note={
                    persons.length === 0
                      ? "Process an event with recognizable faces to see person folders here."
                      : "Try a different name or filename."
                  }
                />
              ) : layout === "grid" ? (
                <div className="ps-folders">
                  {filteredPersons.map((person) => {
                    const count = (person.photos || []).length;
                    const isSelected = person.person_id === selectedPersonId;
                    return (
                      <button
                        key={person.person_id}
                        type="button"
                        className={
                          "ps-folder-card" + (isSelected ? " is-selected" : "")
                        }
                        onClick={() => togglePerson(person.person_id)}
                        aria-pressed={isSelected}
                      >
                        <span className="ps-folder-stack">
                          <PersonThumb
                            jobId={jobId}
                            person={person}
                            variant="stack"
                          />
                        </span>
                        <span className="ps-folder-name">{person.person_id}</span>
                        <span className="ps-folder-count">
                          {plural(count, "photo")}
                        </span>
                      </button>
                    );
                  })}
                </div>
              ) : (
                <ul className="ps-rows">
                  {filteredPersons.map((person) => {
                    const count = (person.photos || []).length;
                    const isSelected = person.person_id === selectedPersonId;
                    return (
                      <li key={person.person_id}>
                        <button
                          type="button"
                          className={"ps-row" + (isSelected ? " is-selected" : "")}
                          onClick={() => togglePerson(person.person_id)}
                          aria-pressed={isSelected}
                        >
                          <span className="ps-row-icon" aria-hidden="true">
                            <PersonThumb
                              jobId={jobId}
                              person={person}
                              variant="row"
                            />
                          </span>
                          <span className="ps-row-name">{person.person_id}</span>
                          <span className="ps-row-meta">
                            {plural(count, "photo")}
                          </span>
                          <IconChevronRight size={15} />
                        </button>
                      </li>
                    );
                  })}
                </ul>
              )}

              {selectedPerson && (
                <section className="ps-detail" aria-label="Person folder">
                  <header className="ps-detail-head">
                    <span className="ps-detail-icon" aria-hidden="true">
                      <IconFolder size={26} />
                    </span>
                    <div className="ps-detail-text">
                      <h2>{selectedPerson.person_id}</h2>
                      <p>{plural(selectedCount, "photo")}</p>
                    </div>
                    <button
                      type="button"
                      className="ps-btn ps-btn-ghost"
                      onClick={() => setSelectedPersonId(null)}
                    >
                      Close
                    </button>
                  </header>

                  {selectedCount > 0 ? (
                    <div className="ps-gallery">
                      {(selectedPerson.photos || []).map((filename) => (
                        <PhotoTile
                          key={filename}
                          jobId={jobId}
                          filename={filename}
                        />
                      ))}
                    </div>
                  ) : (
                    <p className="ps-detail-empty">
                      No accessible photo is associated with this person.
                    </p>
                  )}
                </section>
              )}
            </>
          )}

          {ready && view === "event" && (
            <>
              {filteredPhotos.length === 0 ? (
                <EmptyState
                  Icon={IconImage}
                  title={
                    allPhotos.length === 0
                      ? "No event photos are available for this job."
                      : `No photos match “${query}”.`
                  }
                />
              ) : layout === "grid" ? (
                <div className="ps-photos">
                  {filteredPhotos.map((filename) => (
                    <PhotoTile key={filename} jobId={jobId} filename={filename} />
                  ))}
                </div>
              ) : (
                <ul className="ps-rows">
                  {filteredPhotos.map((filename) => (
                    <PhotoRow key={filename} jobId={jobId} filename={filename} />
                  ))}
                </ul>
              )}
            </>
          )}

          {ready && view === "uncertain" && (
            <>
              {filteredUncertain.length === 0 ? (
                <EmptyState
                  Icon={IconAlert}
                  title={
                    uncertain.length === 0
                      ? "No unresolved uncertain faces."
                      : `No uncertain faces match “${query}”.`
                  }
                  note={
                    uncertain.length === 0
                      ? "Every detected face was either retained or resolved by the pipeline."
                      : undefined
                  }
                />
              ) : layout === "grid" ? (
                <div className="ps-uncertain-grid">
                  {filteredUncertain.map((item, index) => (
                    <article
                      className="ps-uncertain-card"
                      key={`${item.photo}-${item.face_id}-${index}`}
                    >
                      <span className="ps-uncertain-face">
                        Face {item.face_id}
                      </span>
                      <span className="ps-uncertain-photo" title={item.photo}>
                        {item.photo}
                      </span>
                      <span className="ps-uncertain-reason">
                        {item.reason || item.final_status || "Unresolved"}
                      </span>
                    </article>
                  ))}
                </div>
              ) : (
                <ul className="ps-uncertain-list">
                  {filteredUncertain.map((item, index) => (
                    <li
                      className="ps-uncertain-item"
                      key={`${item.photo}-${item.face_id}-${index}`}
                    >
                      <span className="ps-uncertain-face">
                        Face {item.face_id}
                      </span>
                      <span className="ps-uncertain-photo" title={item.photo}>
                        {item.photo}
                      </span>
                      <span className="ps-uncertain-reason">
                        {item.reason || item.final_status || "Unresolved"}
                      </span>
                    </li>
                  ))}
                </ul>
              )}
            </>
          )}
        </div>
      </main>
    </div>
  );
}

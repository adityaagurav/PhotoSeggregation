import { Link } from "react-router-dom";
import Navbar from "../components/Navbar";
import {
  IconPlus,
  IconArrowRight,
  IconArrowDown,
  IconCircle,
  IconImage,
  IconCrowd,
  IconGrid,
  IconSort,
  IconShield,
} from "../components/icons";
import "./LandingPage.css";

const steps = [
  { n: "01", title: "Upload", body: "Add your event photographs.", Icon: IconImage },
  { n: "02", title: "Detect", body: "AI locates faces across the collection.", Icon: IconCircle },
  { n: "03", title: "Organize", body: "Matching faces are grouped into anonymous people.", Icon: IconCrowd },
  { n: "04", title: "Explore", body: "Open each person's photo collection.", Icon: IconGrid },
];

const features = [
  {
    title: "AI Face Detection",
    body: "Finds faces across every photo in the event.",
    Icon: IconCircle,
  },
  {
    title: "Smart Grouping",
    body: "Photos of the same person are collected together, not scattered.",
    Icon: IconCrowd,
  },
  {
    title: "Multi-Person Photos",
    body: "A single photo can appear in several people's galleries.",
    Icon: IconGrid,
  },
  {
    title: "Merge Sort-Based Organization",
    body: "A custom Merge Sort keeps large collections ordered and predictable.",
    Icon: IconSort,
  },
];

const photoTiles = ["t1", "t2", "t3", "t4", "t5", "t6"];

const people = [
  { name: "Person #1", count: "5 photos", thumbs: ["t1", "t3", "t6"] },
  { name: "Person #2", count: "4 photos", thumbs: ["t2", "t5"] },
];

export default function LandingPage() {
  return (
    <div className="page" id="top">
      <Navbar brandHref="#top" />

      <section className="hero">
        <div className="wrap hero-grid">
          <div className="hero-copy">
            <span className="eyebrow">
              <IconPlus size={13} />
              AI-POWERED PHOTO ORGANIZATION
            </span>
            <h1>
              Find Every <span className="accent">Photo</span> You&rsquo;re In.
            </h1>
            <p className="lead">
              Upload your event photographs and let AI sort them into anonymous
              person-wise galleries &mdash; no tagging, no names, no manual albums.
            </p>
            <div className="cta-row">
              <Link className="btn btn-primary" to="/upload">
                Upload Event Photos
                <IconArrowRight size={15} />
              </Link>
              <a className="btn btn-secondary" href="#how">
                How It Works
                <IconArrowDown size={15} />
              </a>
            </div>
            <p className="hero-note">
              <span className="dot" aria-hidden="true" />
              Anonymous Person IDs &middot; no names required
            </p>
          </div>

          <div className="visual" aria-hidden="true">
            <div className="visual-top">
              <span className="visual-label">Photos &rarr; AI &rarr; People</span>
              <span className="visual-tag">Preview</span>
            </div>

            <div className="visual-flow">
              <div className="photo-grid">
                {photoTiles.map((tint) => (
                  <span className={"thumb " + tint} key={tint}>
                    <span className="thumb-face" />
                    <span className="thumb-box" />
                  </span>
                ))}
              </div>

              <span className="flow-arrow">
                <IconArrowRight size={16} />
              </span>

              <div className="ai-node">
                <span className="ai-chip">
                  <IconCircle size={20} />
                </span>
                <span className="ai-name">AI</span>
                <span className="ai-sub">Matching faces</span>
              </div>

              <span className="flow-arrow">
                <IconArrowRight size={16} />
              </span>

              <div className="people">
                {people.map((person) => (
                  <div className="person" key={person.name}>
                    <span className="person-avatar">
                      <IconCrowd size={15} />
                    </span>
                    <span className="person-meta">
                      <span className="person-name">{person.name}</span>
                      <span className="person-count">{person.count}</span>
                    </span>
                    <span className="person-thumbs">
                      {person.thumbs.map((tint) => (
                        <span className={"swatch " + tint} key={tint} />
                      ))}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      </section>

      <section className="section" id="how">
        <div className="wrap">
          <div className="section-head">
            <span className="section-kicker">How It Works</span>
            <h2>From a crowded gallery to clear groups.</h2>
            <p>Four steps, no manual tagging at any point.</p>
          </div>
          <ol className="steps">
            {steps.map(({ n, title, body, Icon }) => (
              <li className="step" key={n}>
                <span className="step-num">
                  <span>{n}</span>
                  <Icon size={14} />
                </span>
                <h3>{title}</h3>
                <p>{body}</p>
              </li>
            ))}
          </ol>
        </div>
      </section>

      <section className="section section-alt">
        <div className="wrap">
          <div className="section-head">
            <span className="section-kicker">What You Get</span>
            <h2>Built for real event photo collections.</h2>
            <p>Everything needed to turn a raw photo dump into organized folders.</p>
          </div>
          <div className="features">
            {features.map(({ title, body, Icon }) => (
              <article className="feature" key={title}>
                <span className="feature-icon">
                  <Icon size={19} />
                </span>
                <h3>{title}</h3>
                <p>{body}</p>
              </article>
            ))}
          </div>
        </div>
      </section>

      <section className="section" id="about">
        <div className="wrap">
          <div className="about">
            <span className="about-icon" aria-hidden="true">
              <IconShield size={22} />
            </span>
            <div className="about-copy">
              <h2>Organized, not identified.</h2>
              <p>
                Faces are grouped into anonymous <strong>Person IDs</strong> instead of
                names. Nobody is labelled as a real person and the original photographs
                stay on your machine &mdash; you simply get person-wise folders to review
                and download.
              </p>
            </div>
          </div>
        </div>
      </section>

      <section className="section">
        <div className="wrap">
          <div className="final-cta">
            <h2>Ready to find your photos?</h2>
            <p>Upload your event collection and let AI do the organizing.</p>
            <Link className="btn btn-primary" to="/upload">
              Upload Event Photos
              <IconArrowRight size={15} />
            </Link>
          </div>

          <footer className="footer">
            <span className="footer-brand">
              <span className="brand-mark small" aria-hidden="true">
                <svg
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.8"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <circle cx="12" cy="12" r="9" />
                  <path d="M8.5 14c.9 1 2.1 1.6 3.5 1.6s2.6-.6 3.5-1.6" />
                  <path d="M9 10h.01M15 10h.01" />
                </svg>
              </span>
              AI Photo Segregation
            </span>
            <span>AI-powered event photo organization</span>
          </footer>
        </div>
      </section>
    </div>
  );
}

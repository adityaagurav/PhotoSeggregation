import { useState } from "react";
import { Link } from "react-router-dom";
import "./Navbar.css";

const DEFAULT_LINKS = [
  { label: "How It Works", href: "#how" },
  { label: "About", href: "#about" },
];

// Renders either a router <Link> (for "/..." hrefs) or a plain anchor (for "#..." hrefs).
function NavLink({ href, children, onClick, className }) {
  if (href && href.startsWith("/")) {
    return (
      <Link to={href} onClick={onClick} className={className}>
        {children}
      </Link>
    );
  }
  return (
    <a href={href} onClick={onClick} className={className}>
      {children}
    </a>
  );
}

export default function Navbar({ brandHref = "/", links = DEFAULT_LINKS }) {
  const [menuOpen, setMenuOpen] = useState(false);
  const close = () => setMenuOpen(false);

  return (
    <header className="nav">
      <div className="wrap nav-inner">
        <NavLink
          href={brandHref}
          className="brand"
          onClick={close}
        >
          <span className="brand-mark" aria-hidden="true">
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.7"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <circle cx="12" cy="12" r="9" />
              <path d="M8.5 14c.9 1 2.1 1.6 3.5 1.6s2.6-.6 3.5-1.6" />
              <path d="M9 10h.01M15 10h.01" />
            </svg>
          </span>
          <span className="brand-name">AI Photo Segregation</span>
        </NavLink>

        <button
          type="button"
          className="nav-burger"
          aria-label="Toggle menu"
          aria-expanded={menuOpen}
          onClick={() => setMenuOpen((open) => !open)}
        >
          <span />
          <span />
          <span />
        </button>

        <nav className={"nav-links" + (menuOpen ? " is-open" : "")} aria-label="Section">
          {links.map((link) => (
            <NavLink key={link.label} href={link.href} onClick={close}>
              {link.label}
            </NavLink>
          ))}
        </nav>
      </div>
    </header>
  );
}

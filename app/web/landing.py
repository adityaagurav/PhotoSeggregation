"""Landing / home page for the AI Photo Segregation web app.

Redesigned presentation-only landing page. Upload, backend, processing, galleries
and all AI/ML pipeline code are intentionally untouched.

The page is rendered as a single assembled HTML document via Streamlit's
``st.markdown(..., unsafe_allow_html=True)``, so the browser receives real HTML/CSS
and renders the visual components instead of displaying markup source.

Start the web app with::

    .venv/bin/python -m streamlit run app/web/streamlit_app.py
"""

from __future__ import annotations

import streamlit as st

# ---------------------------------------------------------------------------
# Icons (inline SVG, monochrome, currentColor). Kept small so they can be
# dropped into headings, buttons and cards without extra requests.
# ---------------------------------------------------------------------------

_ICON_FACE = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/>'
    '<path d="M8.5 14c.9 1 2.1 1.6 3.5 1.6s2.6-.6 3.5-1.6"/>'
    '<path d="M9 10h.01M15 10h.01"/>'
    "</svg>"
)

_ICON_PLUS = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
    "<path d='M12 5v14M5 12h14'/>"
    "</svg>"
)

_ICON_CHEVRON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="2" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M5 12h14M13 6l6 6-6 6"/>'
    "</svg>"
)

_ICON_CIRCLE = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round">'
    '<circle cx="12" cy="12" r="9"/>'
    '<path d="M12 8v4l3 2"/>'
    "</svg>"
)

_ICON_SHIELD = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
    '<path d="M12 3 5 6v5c0 4.4 3 8.3 7 9.5 4-1.2 7-5.1 7-9.5V6l-7-3Z"/>'
    '<path d="m9.2 12 2 2 3.6-3.8"/>'
    "</svg>"
)

_ICON_THUMB = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
    "<rect x='3' y='3' width='18' height='18' rx='3'/>"
    "<circle cx='8.5' cy='8.5' r='1.6'/>"
    "<path d='M21 15l-5.2-5.2a3 3 0 0 0-4.2 0L2 21'/>"
    "</svg>"
)

_ICON_PERSON = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor"'
    ' stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">'
    "<path d='M16 19v-1a4 4 0 0 0-4-4H7a4 4 0 0 0-4 4v1'/>"
    "<circle cx='9.5' cy='8' r='3.2'/>"
    "<path d='M21 19v-1a4 4 0 0 0-3-3.85'/>"
    "<path d='M15.5 5.1a3.2 3.2 0 0 1 0 6.2'/>"
    "</svg>"
)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = f"""<style>
  html {{ scroll-behavior: smooth; }}

  /* Neutralise Streamlit chrome for a clean product page. */
  #MainMenu, header[data-testid="stHeader"], footer {{ display: none !important; }}
  [data-testid="stAppViewContainer"] {{ background: #f7f8fc; }}
  [data-testid="stAppViewContainer"] > .main {{ padding: 0 !important; }}
  .block-container {{ padding: 0 !important; max-width: 100% !important; }}
  [data-testid="stToolbar"] {{ display: none !important; }}

  :root {{
    --bg: #f7f8fc;
    --surface: #ffffff;
    --text: #0b0f1c;
    --muted: #5b6378;
    --faint: #8b93a6;
    --border: #e6e9f2;
    --accent: #6559e8;
    --accent-2: #9370f9;
    --accent-soft: #eef0ff;
    --card: #ffffff;
  }}

  .page, .page * {{ box-sizing: border-box; }}
  .page {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
      Helvetica, Arial, sans-serif;
    color: var(--text);
    background: var(--bg);
    -webkit-font-smoothing: antialiased;
    line-height: 1.5;
  }}
  .wrap {{ max-width: 1120px; margin: 0 auto; padding: 0 28px; }}

  /* ---------------- Navbar ---------------- */
  .nav {{
    position: sticky; top: 0; z-index: 60;
    background: rgba(247, 248, 252, 0.82);
    backdrop-filter: saturate(180%) blur(12px);
    border-bottom: 1px solid var(--border);
  }}
  .nav-inner {{
    display: flex; align-items: center; justify-content: space-between;
    height: 64px;
  }}
  .brand {{
    display: flex; align-items: center; gap: 10px;
    font-weight: 700; letter-spacing: -0.01em; font-size: 1.04rem;
    color: var(--text); text-decoration: none;
  }}
  .brand .mark {{
    width: 30px; height: 30px; border-radius: 9px; flex: 0 0 auto;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: grid; place-items: center; color: #fff;
  }}
  .brand .mark svg {{ width: 17px; height: 17px; }}
  .nav-links {{ display: flex; align-items: center; gap: 28px; }}
  .nav-links a {{
    color: var(--muted); text-decoration: none; font-size: 0.94rem;
    font-weight: 500; transition: color .15s ease;
  }}
  .nav-links a:hover {{ color: var(--text); }}
  .nav-toggle, .nav-burger {{ display: none; }}

  /* ---------------- Hero ---------------- */
  .hero {{
    padding: 66px 0 56px;
    background:
      radial-gradient(900px 520px at 82% -10%, rgba(101,89,232,.10), rgba(147,112,249,0) 60%),
      radial-gradient(760px 520px at 4% 110%, rgba(147,112,249,.08), rgba(101,89,232,0) 60%),
      var(--bg);
  }}
  .hero-grid {{
    display: grid;
    grid-template-columns: 1.02fr 0.98fr;
    gap: 54px;
    align-items: center;
  }}
  .eyebrow {{
    display: inline-flex; align-items: center; gap: 9px;
    font-size: 0.74rem; font-weight: 600; letter-spacing: 0.12em;
    text-transform: uppercase; color: var(--accent);
    background: var(--accent-soft); border: 1px solid #e2e1ff;
    padding: 6px 12px; border-radius: 999px; margin-bottom: 18px;
  }}
  .eyebrow svg {{ width: 13px; height: 13px; }}
  .hero h1 {{
    font-size: 2.95rem; line-height: 1.05; letter-spacing: -0.03em;
    margin: 0 0 16px; font-weight: 800;
  }}
  .hero h1 .grad {{
    background: linear-gradient(120deg, var(--accent), var(--accent-2));
    -webkit-background-clip: text; background-clip: text; color: transparent;
  }}
  .hero .lead {{
    font-size: 1.08rem; color: var(--muted); max-width: 34em;
    margin: 0 0 26px;
  }}
  .cta-row {{ display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }}
  .btn {{
    display: inline-flex; align-items: center; gap: 9px;
    font-size: 0.96rem; font-weight: 600; padding: 12px 20px;
    border-radius: 12px; text-decoration: none; cursor: pointer;
    border: 1px solid transparent;
    transition: transform .12s ease, box-shadow .2s ease, background .2s ease;
  }}
  .btn-primary {{
    color: #fff;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    box-shadow: 0 10px 24px rgba(101,89,232,.28);
  }}
  .btn-primary:hover {{
    transform: translateY(-1px);
    box-shadow: 0 14px 28px rgba(101,89,232,.34);
  }}
  .btn-ghost {{ color: var(--text); background: var(--surface); border-color: var(--border); }}
  .btn-ghost:hover {{ border-color: #cfd5e6; }}
  .hero-note {{
    margin-top: 14px; font-size: 0.82rem; color: var(--faint);
    display: flex; align-items: center; gap: 8px;
  }}
  .hero-note .dot {{
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent); opacity: .85;
  }}

  /* ---------------- Hero visual ---------------- */
  .visual {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 22px;
    padding: 20px;
    box-shadow: 0 26px 64px rgba(16, 20, 46, 0.08);
  }}
  .visual-head {{
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 14px;
  }}
  .visual-title {{
    font-size: .72rem; font-weight: 700; letter-spacing: .12em;
    text-transform: uppercase; color: var(--faint);
  }}
  .visual-badge {{
    font-size: .68rem; font-weight: 600; letter-spacing: .04em;
    color: var(--accent); background: var(--accent-soft);
    border: 1px solid #e3e1ff; padding: 4px 9px; border-radius: 999px;
    display: inline-flex; align-items: center; gap: 6px;
  }}
  .visual-badge svg {{ width: 11px; height: 11px; }}

  .flow {{
    display: flex; align-items: center; gap: 14px;
  }}
  .flow-col {{ flex: 1 1 auto; min-width: 0; }}
  .photo-grid {{
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 8px;
  }}
  .tile {{
    aspect-ratio: 1 / 1; border-radius: 10px; position: relative;
    overflow: hidden; border: 1px solid rgba(15, 23, 42, .06);
    background: #fff;
    box-shadow: 0 1px 2px rgba(16, 20, 46, .04);
  }}
  .tile::before {{
    content: "";
    position: absolute; inset: 0;
    background: linear-gradient(135deg, rgba(255,255,255,.55), rgba(255,255,255,0) 60%);
  }}
  .tile .face {{
    position: absolute; left: 50%; top: 42%;
    width: 30%; padding-bottom: 30%;
    transform: translate(-50%, -50%);
    border-radius: 50% 50% 46% 46%;
    background: rgba(255,255,255,.55);
  }}
  .tile .face-shadow {{
    position: absolute; left: 50%; top: 42%;
    width: 28%; padding-bottom: 28%;
    transform: translate(-50%, -50%);
    border-radius: 50% 50% 42% 42%;
    background: rgba(255,255,255,.32);
  }}
  .tile .grad {{
    position: absolute; inset: 0;
  }}
  .t1 .grad {{ background: linear-gradient(135deg, #c7d2fe, #818cf8); }}
  .t2 .grad {{ background: linear-gradient(135deg, #fbcfe8, #f0abfc); }}
  .t3 .grad {{ background: linear-gradient(135deg, #bae6fd, #38bdf8); }}
  .t4 .grad {{ background: linear-gradient(135deg, #bbf7d0, #4ade80); }}
  .t5 .grad {{ background: linear-gradient(135deg, #fde68a, #fbbf24); }}
  .t6 .grad {{ background: linear-gradient(135deg, #ddd6fe, #a78bfa); }}

  .connector {{
    flex: 0 0 auto; height: 2px;
    background: linear-gradient(90deg, var(--accent), var(--accent-2));
    position: relative;
  }}
  .connector::after {{
    content: "";
    position: absolute; left: 50%; top: 50%;
    width: 8px; height: 8px; border-radius: 50%;
    background: #fff;
    transform: translate(-50%, -50%);
    box-shadow: 0 0 0 3px rgba(101,89,232,.25);
  }}

  .ai-card {{
    flex: 0 1 auto; min-width: 0;
    background: linear-gradient(135deg, rgba(101,89,232,.10), rgba(147,112,249,.06));
    border: 1px solid #e7e3ff;
    border-radius: 16px;
    padding: 14px 12px;
    text-align: center;
    box-shadow: 0 10px 22px rgba(101,89,232,.16);
  }}
  .ai-card .chip {{
    width: 54px; height: 54px; margin: 0 auto 8px;
    border-radius: 15px;
    background: linear-gradient(135deg, var(--accent), var(--accent-2));
    display: grid; place-items: center; color: #fff;
    box-shadow: 0 10px 20px rgba(101,89,232,.32);
  }}
  .ai-card .chip svg {{ width: 24px; height: 24px; }}
  .ai-card .label {{
    font-size: .7rem; font-weight: 700; letter-spacing: .06em;
    color: var(--accent);
  }}
  .ai-card .sub {{
    font-size: .64rem; color: var(--faint); margin-top: 2px;
  }}

  .people {{
    display: flex; flex-direction: column; gap: 10px;
  }}
  .person {{
    display: flex; align-items: center; gap: 12px;
    padding: 10px 12px;
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
    transition: border-color .15s ease, box-shadow .15s ease;
  }}
  .person:hover {{
    border-color: #d6dcee;
    box-shadow: 0 6px 14px rgba(16, 20, 46, .05);
  }}
  .person .avatar {{
    width: 32px; height: 32px; border-radius: 9px; flex: 0 0 auto;
    background: linear-gradient(135deg, #ece8ff, #c7c9f7);
    display: grid; place-items: center; color: var(--accent);
  }}
  .person .avatar svg {{ width: 16px; height: 16px; }}
  .person .meta {{ flex: 1 1 auto; min-width: 0; }}
  .person .name {{ font-size: .84rem; font-weight: 700; }}
  .person .count {{ font-size: .72rem; color: var(--faint); margin-top: 1px; }}
  .person .thumbs {{
    display: flex; gap: 4px;
  }}
  .person .thumbs i {{
    width: 14px; height: 14px; border-radius: 4px; display: block;
  }}

  /* ---------------- Sections ---------------- */
  section {{ padding: 58px 0; }}
  .section-head {{ max-width: 600px; margin: 0 0 30px; }}
  .section-head h2 {{
    font-size: 1.8rem; letter-spacing: -0.02em; margin: 0 0 8px; font-weight: 800;
  }}
  .section-head p {{ color: var(--muted); margin: 0; font-size: 1.02rem; }}

  /* ---------------- How it works ---------------- */
  .steps {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
  }}
  .step {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 18px;
  }}
  .step .num {{
    font-size: .72rem; font-weight: 800; letter-spacing: .06em;
    color: var(--accent); margin-bottom: 10px;
    display: inline-flex; align-items: center; gap: 7px;
  }}
  .step .num svg {{ width: 14px; height: 14px; }}
  .step h3 {{ font-size: 1.04rem; margin: 0 0 5px; }}
  .step p {{ color: var(--muted); margin: 0; font-size: .9rem; }}

  /* ---------------- Features ---------------- */
  .features {{
    display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px;
  }}
  .card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 20px 18px;
    transition: transform .15s ease, box-shadow .2s ease, border-color .2s ease;
  }}
  .card:hover {{
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(16, 20, 46, .07);
    border-color: #dcdff0;
  }}
  .card .ic {{
    width: 42px; height: 42px; border-radius: 12px; margin-bottom: 14px;
    background: var(--accent-soft); color: var(--accent);
    display: grid; place-items: center;
  }}
  .card .ic svg {{ width: 20px; height: 20px; }}
  .card h3 {{ font-size: 1.02rem; margin: 0 0 6px; }}
  .card p {{ color: var(--muted); margin: 0; font-size: .9rem; }}

  /* ---------------- Privacy ---------------- */
  .privacy {{
    background: linear-gradient(135deg, #16192f, #231d4f);
    color: #fff; border-radius: 22px; padding: 36px 38px;
    display: flex; gap: 22px; align-items: center;
  }}
  .privacy .ic {{
    width: 48px; height: 48px; border-radius: 13px; flex: 0 0 auto;
    background: rgba(255, 255, 255, .10);
    display: grid; place-items: center;
  }}
  .privacy .ic svg {{ width: 24px; height: 24px; }}
  .privacy h2 {{
    font-size: 1.35rem; margin: 0 0 6px; letter-spacing: -0.01em;
  }}
  .privacy p {{
    margin: 0; color: rgba(255, 255, 255, .78);
    max-width: 50em; font-size: .94rem;
  }}

  /* ---------------- Final CTA ---------------- */
  .final {{
    text-align: center; background: var(--surface);
    border: 1px solid var(--border); border-radius: 22px;
    padding: 48px 24px;
  }}
  .final h2 {{ font-size: 1.8rem; margin: 0 0 8px; letter-spacing: -0.02em; }}
  .final p {{ color: var(--muted); margin: 0 0 24px; }}

  .footer {{
    padding: 30px 0 40px; color: var(--faint); font-size: .84rem;
    display: flex; justify-content: space-between; gap: 14px; flex-wrap: wrap;
    border-top: 1px solid var(--border); margin-top: 30px;
  }}

  /* ---------------- Responsive ---------------- */
  @media (max-width: 980px) {{
    .hero-grid {{ grid-template-columns: 1fr; gap: 36px; }}
    .hero h1 {{ font-size: 2.4rem; }}
    .steps, .features {{ grid-template-columns: repeat(2, 1fr); }}
    .flow {{ flex-direction: column; }}
    .connector {{ width: 2px; height: 24px; }}
    .connector::after {{ top: 50%; left: 50%; width: 8px; height: 8px; }}
    .ai-card {{ transform: none; }}
  }}
  @media (max-width: 720px) {{
    .wrap {{ padding: 0 20px; }}
    .hero {{ padding: 44px 0 34px; }}
    .hero h1 {{ font-size: 2rem; }}
    .hero .lead {{ font-size: .98rem; }}
    .steps, .features {{ grid-template-columns: 1fr; }}
    .nav-burger {{
      display: inline-flex; flex-direction: column; justify-content: center;
      gap: 4px; width: 40px; height: 40px; border-radius: 10px;
      border: 1px solid var(--border); background: var(--surface);
    }}
    .nav-burger span {{ display: block; height: 2px; width: 18px; margin: 0 auto;
      background: var(--text); border-radius: 2px; }}
    .nav-links {{
      position: absolute; left: 0; right: 0; top: 64px;
      flex-direction: column; align-items: flex-start; gap: 0;
      background: var(--surface); border-bottom: 1px solid var(--border);
      padding: 10px 20px; display: none;
    }}
    .nav-inner {{ position: relative; }}
    .nav-toggle:checked ~ .nav-links {{ display: flex; }}
    .nav-links a {{ padding: 12px 0; width: 100%; font-size: .98rem; }}
    .privacy {{ flex-direction: column; align-items: flex-start; padding: 28px 24px; }}
    .final {{ padding: 40px 20px; }}
    .footer {{ flex-direction: column; }}
  }}
</style>"""


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

_HTML = f"""<div class="page" id="top">
  <div class="nav">
    <div class="wrap nav-inner">
      <a class="brand" href="#top">
        <span class="mark">{_ICON_FACE}</span>
        AI Photo Segregation
      </a>
      <input type="checkbox" id="nav-toggle" class="nav-toggle" />
      <label for="nav-toggle" class="nav-burger" aria-label="Toggle menu">
        <span></span><span></span><span></span>
      </label>
      <nav class="nav-links">
        <a href="#how">How It Works</a>
        <a href="#about">About</a>
      </nav>
    </div>
  </div>

  <header class="hero">
    <div class="wrap hero-grid">
      <div class="hero-copy">
        <span class="eyebrow">{_ICON_PLUS} AI-POWERED PHOTO ORGANIZATION</span>
        <h1>Find Every <span class="grad">Photo</span> You're In.</h1>
        <p class="lead">
          Upload your event photos and let AI automatically organize them into
          anonymous person-wise galleries.
        </p>
        <div class="cta-row">
          <a class="btn btn-primary" href="#upload">Upload Event Photos {_ICON_CHEVRON}</a>
          <a class="btn btn-ghost" href="#how">See How It Works {_ICON_CHEVRON}</a>
        </div>
        <p class="hero-note">
          <span class="dot"></span> Anonymous Person IDs &middot; No names required
        </p>
      </div>

      <div class="visual">
        <div class="visual-head">
          <span class="visual-title">Photos &rarr; AI &rarr; People</span>
          <span class="visual-badge">{_ICON_CIRCLE} Processing</span>
        </div>
        <div class="flow">
          <div class="flow-col">
            <div class="photo-grid">
              <div class="tile t1"><span class="grad"></span><span class="face"></span></div>
              <div class="tile t2"><span class="grad"></span><span class="face-shadow"></span></div>
              <div class="tile t3"><span class="grad"></span><span class="face"></span></div>
              <div class="tile t4"><span class="grad"></span><span class="face-shadow"></span></div>
              <div class="tile t5"><span class="grad"></span><span class="face"></span></div>
              <div class="tile t6"><span class="grad"></span><span class="face-shadow"></span></div>
            </div>
          </div>

          <div class="connector"></div>

          <div class="ai-card">
            <div class="chip">{_ICON_CIRCLE}</div>
            <div class="label">AI</div>
            <div class="sub">Matches &amp; groups</div>
          </div>

          <div class="connector"></div>

          <div class="flow-col">
            <div class="people">
              <div class="person">
                <span class="avatar">{_ICON_PERSON}</span>
                <span class="meta"><span class="name">Person #1</span><br>
                  <span class="count">5 photos</span></span>
                <span class="thumbs">
                  <span class="tile t1" style="width:14px;height:14px;border-radius:4px;display:block;background:linear-gradient(135deg,#c7d2fe,#818cf8)"></span>
                  <span class="tile t3" style="width:14px;height:14px;border-radius:4px;display:block;background:linear-gradient(135deg,#bae6fd,#38bdf8)"></span>
                  <span class="tile t6" style="width:14px;height:14px;border-radius:4px;display:block;background:linear-gradient(135deg,#ddd6fe,#a78bfa)"></span>
                </span>
              </div>
              <div class="person">
                <span class="avatar">{_ICON_PERSON}</span>
                <span class="meta"><span class="name">Person #2</span><br>
                  <span class="count">4 photos</span></span>
                <span class="thumbs">
                  <span class="tile t2" style="width:14px;height:14px;border-radius:4px;display:block;background:linear-gradient(135deg,#fbcfe8,#f0abfc)"></span>
                  <span class="tile t5" style="width:14px;height:14px;border-radius:4px;display:block;background:linear-gradient(135deg,#fde68a,#fbbf24)"></span>
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </header>

  <section id="how">
    <div class="wrap">
      <div class="section-head">
        <h2>How It Works</h2>
        <p>From hundreds of photos to your personal gallery.</p>
      </div>
      <div class="steps">
        <div class="step">
          <span class="num">01 {_ICON_PLUS}</span>
          <h3>Upload</h3>
          <p>Add your event photos.</p>
        </div>
        <div class="step">
          <span class="num">02 {_ICON_CIRCLE}</span>
          <h3>Detect</h3>
          <p>AI finds faces across the collection.</p>
        </div>
        <div class="step">
          <span class="num">03 {_ICON_PERSON}</span>
          <h3>Organize</h3>
          <p>Matching faces are grouped into anonymous people.</p>
        </div>
        <div class="step">
          <span class="num">04 {_ICON_CHEVRON}</span>
          <h3>Explore</h3>
          <p>Browse each person's photo collection.</p>
        </div>
      </div>
    </div>
  </section>

  <section>
    <div class="wrap">
      <div class="section-head">
        <h2>Built for event photo collections.</h2>
        <p>Everything you need to turn a crowded photo dump into clear groups.</p>
      </div>
      <div class="features">
        <div class="card">
          <div class="ic">{_ICON_FACE}</div>
          <h3>AI Face Detection</h3>
          <p>Find relevant faces across your event photos.</p>
        </div>
        <div class="card">
          <div class="ic">{_ICON_PERSON}</div>
          <h3>Smart Grouping</h3>
          <p>Photos of the same person are grouped together.</p>
        </div>
        <div class="card">
          <div class="ic">{_ICON_CHEVRON}</div>
          <h3>Multi-Person Photos</h3>
          <p>A single photo can appear in multiple people's galleries.</p>
        </div>
        <div class="card">
          <div class="ic">{_ICON_SHIELD}</div>
          <h3>Anonymous by Design</h3>
          <p>People are represented using Person # IDs rather than names.</p>
        </div>
      </div>
    </div>
  </section>

  <section id="about">
    <div class="wrap">
      <div class="privacy">
        <span class="ic">{_ICON_SHIELD}</span>
        <div>
          <h2>Your photos. Organized anonymously.</h2>
          <p>People are represented using anonymous Person IDs instead of names.</p>
        </div>
      </div>
    </div>
  </section>

  <section>
    <div class="wrap">
      <div class="final">
        <h2>Ready to find your photos?</h2>
        <p>Upload your event collection and let AI do the organizing.</p>
        <a class="btn btn-primary" href="#upload">Upload Event Photos {_ICON_CHEVRON}</a>
      </div>
      <div class="footer">
        <span>AI Photo Segregation</span>
        <span>AI-powered event photo organization</span>
      </div>
    </div>
  </section>
</div>"""


def render() -> None:
    """Render the redesigned landing page (presentation only; no upload/backend yet)."""
    st.set_page_config(
        page_title="AI Photo Segregation",
        page_icon=":material/insights:",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_CSS, unsafe_allow_html=True)
    st.markdown(_HTML, unsafe_allow_html=True)


if __name__ == "__main__":
    render()

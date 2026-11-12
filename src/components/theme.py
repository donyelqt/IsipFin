from __future__ import annotations

import html
from collections.abc import Iterator, Sequence
from contextlib import contextmanager

import streamlit as st

__all__ = [
    "ACCENT",
    "ACCENT_DIM",
    "ACCENT_SOFT",
    "ACCENT_STRONG",
    "BG",
    "BORDER",
    "BORDER_STRONG",
    "CSS_CLASSES",
    "ERROR",
    "FONT_IMPORT_URL",
    "FONT_MONO",
    "FONT_SANS",
    "RADIUS",
    "RADIUS_LG",
    "RADIUS_SMALL",
    "SUCCESS",
    "SURFACE",
    "SURFACE_2",
    "TEXT",
    "TEXT_DIM",
    "TEXT_MUTED",
    "TOKENS",
    "WARNING",
    "badge",
    "caption_note",
    "eyebrow",
    "figure",
    "footer_note",
    "hero_mark",
    "hero_sub",
    "hero_title",
    "inject_theme",
    "label",
    "panel",
    "rail_brand",
    "rail_label",
    "rail_status_card",
    "spacer",
]


BG = "#0A0A0B"
SURFACE = "#131316"
SURFACE_2 = "#1B1B20"
BORDER = "#2A2A31"
BORDER_STRONG = "#3A3A44"

TEXT = "#F5F5F6"
TEXT_MUTED = "#9A9AA3"
TEXT_DIM = "#6C6C76"

ACCENT = "#FF7A1A"
ACCENT_STRONG = "#FF9347"
ACCENT_DIM = "rgba(255,122,26,0.14)"
ACCENT_SOFT = "rgba(255,122,26,0.10)"

SUCCESS = "#3FB950"
WARNING = "#D29922"
ERROR = "#F85149"

RADIUS = "10px"
RADIUS_LG = "16px"
RADIUS_SMALL = "8px"

FONT_SANS = (
    '"IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", '
    'Roboto, "Helvetica Neue", Arial, sans-serif'
)
FONT_MONO = (
    '"IBM Plex Mono", ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, '
    '"Liberation Mono", monospace'
)

FONT_IMPORT_URL = "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap"

TOKENS: dict[str, str] = {
    "--isip-bg": BG,
    "--isip-surface": SURFACE,
    "--isip-surface-2": SURFACE_2,
    "--isip-border": BORDER,
    "--isip-border-strong": BORDER_STRONG,
    "--isip-text": TEXT,
    "--isip-text-muted": TEXT_MUTED,
    "--isip-text-dim": TEXT_DIM,
    "--isip-accent": ACCENT,
    "--isip-accent-strong": ACCENT_STRONG,
    "--isip-accent-dim": ACCENT_DIM,
    "--isip-accent-soft": ACCENT_SOFT,
    "--isip-success": SUCCESS,
    "--isip-warning": WARNING,
    "--isip-error": ERROR,
    "--isip-radius": RADIUS,
    "--isip-radius-lg": RADIUS_LG,
    "--isip-radius-small": RADIUS_SMALL,
    "--isip-font-sans": FONT_SANS,
    "--isip-font-mono": FONT_MONO,
}

CSS_CLASSES = (
    "isip-panel",
    "isip-panel-2",
    "isip-flush",
    "isip-label",
    "isip-figure",
    "isip-figure-muted",
    "isip-figure-lg",
    "isip-badge",
    "isip-badge-success",
    "isip-badge-warn",
    "isip-badge-error",
    "isip-badge-accent",
    "isip-badge-neutral",
    "isip-chip",
    "isip-chip-accent",
    "isip-rule",
    "isip-spacer",
    "isip-note",
    "isip-mark",
    "isip-hero-mark",
    "isip-hero",
    "isip-hero-title",
    "isip-hero-accent",
    "isip-hero-sub",
    "isip-eyebrow",
    "isip-footer",
    "isip-rail-brand",
    "isip-rail-wordmark",
    "isip-rail-subtitle",
    "isip-rail-names",
    "isip-rail-label",
    "isip-rail-card",
    "isip-rail-card-head",
    "isip-rail-card-title",
    "isip-rail-card-line",
)


def _token_block() -> str:
    lines = [f"    {name}: {value};" for name, value in TOKENS.items()]
    return ":root {\n" + "\n".join(lines) + "\n}"


_STYLESHEET_BODY = """
/* --- base --------------------------------------------------------------- */

/*
 * Streamlit's own theme is not switched off — the app keeps native widgets
 * (buttons, selectboxes, sliders, chat input) and their behaviour. This block
 * overrides the *skin* and adds our own primitives on top of it.
 */

.stApp,
.stAppViewContainer,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
[data-testid="stMainBlockContainer"],
[data-testid="stSidebarContent"],
.stMainBlockContainer {
    background-color: var(--isip-bg);
}

/*
 * Base type. `font-family` is inherited by Streamlit's emotion-styled widgets
 * too, which is why the whole app reads as one family after this single rule.
 */
.stApp {
    font-family: var(--isip-font-sans);
    color: var(--isip-text);
}

/* Streamlit's markdown containers sit on the page colour, not transparent:
   without this they flash light between the webfont swap. */
[data-testid="stMarkdownContainer"],
.stMarkdownContainer {
    font-family: var(--isip-font-sans);
    color: var(--isip-text);
}

/* Headings: tighter tracking than Streamlit's default, weight not size, so the
   hierarchy survives the small main-pane width. */
h1, h2, h3, h4 {
    font-family: var(--isip-font-sans);
    color: var(--isip-text);
    letter-spacing: -0.01em;
}

h1 { font-weight: 600; }
h2 { font-weight: 600; }
h3 { font-weight: 500; }

a {
    color: var(--isip-accent);
    text-decoration: none;
}

a:hover {
    color: var(--isip-accent-strong);
    text-decoration: underline;
}

hr {
    border-color: var(--isip-border);
}

/* Native input skin: the one place the 8px radius is prescribed. */
input, textarea, [data-baseweb="input"], [data-baseweb="textarea"] {
    font-family: var(--isip-font-sans) !important;
    border-radius: var(--isip-radius-small) !important;
    color: var(--isip-text);
}

[data-baseweb="input"]:focus-within,
[data-baseweb="textarea"]:focus-within {
    border-color: var(--isip-border-strong) !important;
    box-shadow: none !important;
}

/* --- sidebar ------------------------------------------------------------ */

/*
 * Sidebar rail. NOTE: the width is deliberately NOT set here.
 * Streamlit computes the rail's width in JS and applies it as an *inline*
 * style on `stSidebar` (from `initial_sidebar_state=<int>` in set_page_config,
 * or the user's drag), and no stylesheet rule — not even `!important` — beats an
 * inline style. The rail therefore gets its 260px from
 * `st.set_page_config(initial_sidebar_state=260)` in app.py, and this block
 * only owns the surface, the hairline and the padding.
 */
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background-color: var(--isip-surface);
    border-right: 1px solid var(--isip-border);
}

[data-testid="stSidebarContent"] {
    padding: 1.25rem 1rem;
}

/* --- primary action ----------------------------------------------------- */

/*
 * The primary CTA. One rule, one target: `st.button(..., type="primary")` renders
 * as `.stBaseButton-primary`, which Streamlit itself styles for the light
 * theme. This restyles the shell and the label, and keeps the hover/active
 * states inside the same spec.
 *
 * A rounded rectangle on the shared 10px radius, not a pill: the reference layout
 * this rail is modelled on uses a rectangular CTA, and a pill there reads as a
 * different component than the nav rows directly beneath it.
 */
button.stBaseButton-primary,
[data-testid="stBaseButton-primary"],
button[data-testid="stBaseButton-primary"] {
    background-color: var(--isip-accent) !important;
    border: 1px solid var(--isip-accent) !important;
    border-radius: var(--isip-radius) !important;
    color: #0A0A0B !important;
    font-family: var(--isip-font-sans) !important;
    font-weight: 600 !important;
    transition:
        background-color 120ms ease,
        border-color 120ms ease;
}

button.stBaseButton-primary:hover,
[data-testid="stBaseButton-primary"]:hover {
    background-color: var(--isip-accent-strong) !important;
    border-color: var(--isip-accent-strong) !important;
}

button.stBaseButton-primary:active,
[data-testid="stBaseButton-primary"]:active {
    background-color: var(--isip-accent) !important;
    border-color: var(--isip-accent) !important;
}

button.stBaseButton-primary:disabled,
[data-testid="stBaseButton-primary"]:disabled {
    background-color: var(--isip-surface-2) !important;
    border-color: var(--isip-border) !important;
    color: var(--isip-text-dim) !important;
    cursor: not-allowed;
}

/* Secondary buttons: surface fill, accent text, same radius rule as inputs. */
button.stBaseButton-secondary,
[data-testid="stBaseButton-secondary"] {
    background-color: var(--isip-surface-2) !important;
    border: 1px solid var(--isip-border) !important;
    border-radius: var(--isip-radius-small) !important;
    color: var(--isip-text) !important;
    font-family: var(--isip-font-sans) !important;
    transition:
        background-color 120ms ease,
        border-color 120ms ease;
}

button.stBaseButton-secondary:hover,
[data-testid="stBaseButton-secondary"]:hover {
    border-color: var(--isip-border-strong) !important;
    color: var(--isip-text);
}

/* --- chat transcript ---------------------------------------------------- */

/*
 * The transcript is the app's main surface. Chat bubbles go borderless (the
 * chat component already separates roles by avatar); the accent bar is what
 * marks an assistant answer apart from a user question.
 */
[data-testid="stChatMessage"] {
    background: transparent;
    border: none;
    padding: 0.25rem 0 0.75rem 0;
}

[data-testid="stChatMessageContent"] {
    font-family: var(--isip-font-sans);
    color: var(--isip-text);
}

/*
 * Role differentiation. `st.chat_message` takes no `key=`, so the role cannot
 * be carried on the chat node itself; and `st.html` calls are sibling element
 * containers, so a `<div>` opened in one call cannot be closed in a later one.
 * The working path is the container Streamlit already keyed: `st.container`'s
 * `key` becomes a `st-key-<key>` class on its `stElementContainer` element, and
 * the caller renders *inside* that container. The chat pane therefore does
 *
 *     with st.chat_message(name="assistant"):
 *         with st.container(border=False, key="isip-answer"):
 *             st.markdown(answer_md)
 *
 * which lands the rule on the element that actually wraps the answer.
 *
 * The match is a SUBSTRING selector, not an exact one, and not a prefix one
 * either. Streamlit puts the whole key into ONE class token —
 * `st-key-isip-answer-7` — alongside its own tokens, so an exact
 * `.st-key-isip-answer` selector misses the suffixed key, and the token is not
 * even first in the attribute: measured in Streamlit 1.65, a keyed element
 * reads `class="stElementContainer element-container st-key-isip-answer-7
 * st-emotion-cache-… e6x7pi3"`, so `[class^="st-key-…"]` matches nothing at
 * all. `[class*="st-key-isip-answer"]` matches the token wherever Streamlit
 * decides to order it, and however the caller makes the key unique.
 *
 * The same substring match is also the hazard: it is a wide net, so a widget
 * keyed `isip-answer-retry` inherits this accent border. The prefixes are
 * reserved by their owning module constants and must not be reused by any other
 * component — a retry button is `isip-chat-retry`, the rail's nav is
 * `isip-nav-*` (sidebar), hero buttons are `isip-suggestion-*` (chat), and the
 * conversation rows are `isip-session-*`. Pick a new prefix rather than
 * borrowing one.
 */
[class*="st-key-isip-answer"] {
    border-left: 2px solid var(--isip-accent);
    padding-left: 0.85rem;
}

[class*="st-key-isip-question"] {
    padding-left: 0.15rem;
}

/* User questions sit one step back from the answers: dimmer, no rule. */
[class*="st-key-isip-question"] [data-testid="stMarkdownContainer"] {
    color: var(--isip-text-muted);
}

/* Same two roles for callers that own the class themselves (the panel
   marker's wrapper, a manually built card), so both paths render alike. */
.isip-answer {
    border-left: 2px solid var(--isip-accent);
    padding-left: 0.85rem;
}

.isip-question {
    padding-left: 0.15rem;
}

.isip-question [data-testid="stMarkdownContainer"] {
    color: var(--isip-text-muted);
}

/* Tables inside an answer: mono figures, hairlines, no zebra noise. */
.isip-figure table,
[data-testid="stMarkdownContainer"] table {
    font-family: var(--isip-font-mono);
    font-size: 0.82rem;
    border-collapse: collapse;
    width: 100%;
}

[data-testid="stMarkdownContainer"] table th,
[data-testid="stMarkdownContainer"] table td {
    border: 1px solid var(--isip-border);
    padding: 0.35rem 0.55rem;
    text-align: left;
}

[data-testid="stMarkdownContainer"] table th {
    background-color: var(--isip-surface-2);
    color: var(--isip-text);
    font-weight: 500;
}

[data-testid="stMarkdownContainer"] table tr td {
    color: var(--isip-text-muted);
}

/*
 * The composer. `st.chat_input` is pinned to the bottom by Streamlit's own
 * `stBottomBlockContainer`; this block owns the dock's surface and gives it the
 * large radius, so the one input the user types into is the softest, most
 * obviously *the* control on the canvas.
 */
[data-testid="stBottomBlockContainer"] {
    background-color: var(--isip-bg);
    padding-bottom: 0.85rem;
}

[data-testid="stChatInput"] {
    background-color: var(--isip-surface);
    border: 1px solid var(--isip-border);
    border-radius: var(--isip-radius-lg);
    padding: 0.3rem;
    transition:
        border-color 140ms ease,
        box-shadow 140ms ease;
}

[data-testid="stChatInput"]:focus-within {
    border-color: var(--isip-border-strong);
    box-shadow: 0 0 0 3px var(--isip-accent-soft);
}

[data-testid="stChatInputTextArea"] {
    background-color: transparent !important;
    border: none !important;
    color: var(--isip-text) !important;
    font-family: var(--isip-font-sans) !important;
    font-size: 0.95rem !important;
    padding: 0.55rem 0.65rem !important;
    box-shadow: none !important;
}

[data-testid="stChatInputTextArea"]::placeholder {
    color: var(--isip-text-dim);
}

[data-testid="stChatInputSubmitButton"] {
    background-color: var(--isip-accent);
    color: #0A0A0B;
    border-radius: var(--isip-radius-small);
}

[data-testid="stChatInputSubmitButton"]:hover {
    background-color: var(--isip-accent-strong);
}

/* --- our primitives ----------------------------------------------------- */

/* The panel: the 10px-radius card every widget group sits in. */
.isip-panel {
    background-color: var(--isip-surface);
    border: 1px solid var(--isip-border);
    border-radius: var(--isip-radius);
    padding: 0.9rem 1rem;
    margin-bottom: 0.75rem;
}

.isip-panel-2 {
    background-color: var(--isip-surface-2);
}

/*
 * The flush variant: for the sidebar's brand row, footer row and any other
 * group that must NOT draw a second card inside the rail's own surface. The
 * sidebar already paints a card behind itself; a panel inside it would read as
 * a hole. `panel(..., extra_classes="isip-flush")` selects it.
 */
.isip-flush {
    background: transparent;
    border: none;
    padding: 0;
    margin-bottom: 0.5rem;
}

/* Uppercase monospace field label — the "PREVIOUS CONVERSATIONS" voice. */
.isip-label {
    font-family: var(--isip-font-sans);
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    color: var(--isip-text-dim);
    margin: 0 0 0.35rem 0;
}

/* Monospace figures: every number, id, page or score the UI shows. */
.isip-figure {
    font-family: var(--isip-font-mono);
    font-size: 0.82rem;
    font-variant-numeric: tabular-nums;
    color: var(--isip-text);
}

.isip-figure-muted {
    color: var(--isip-text-muted);
}

.isip-figure-lg {
    font-size: 1.05rem;
    font-weight: 500;
}

/* Status pill. Built on the 8px radius so chips and inputs share one scale. */
.isip-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-family: var(--isip-font-sans);
    font-size: 0.66rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    padding: 0.16rem 0.5rem;
    border-radius: var(--isip-radius-small);
    border: 1px solid var(--isip-border-strong);
    background-color: var(--isip-surface-2);
    color: var(--isip-text-muted);
    white-space: nowrap;
}

.isip-badge::before {
    content: "";
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background-color: currentColor;
    flex: 0 0 auto;
}

.isip-badge-success {
    color: var(--isip-success);
    border-color: color-mix(in srgb, var(--isip-success) 45%, transparent);
    background-color: color-mix(in srgb, var(--isip-success) 12%, var(--isip-surface));
}

.isip-badge-warn {
    color: var(--isip-warning);
    border-color: color-mix(in srgb, var(--isip-warning) 45%, transparent);
    background-color: color-mix(in srgb, var(--isip-warning) 12%, var(--isip-surface));
}

.isip-badge-error {
    color: var(--isip-error);
    border-color: color-mix(in srgb, var(--isip-error) 45%, transparent);
    background-color: color-mix(in srgb, var(--isip-error) 12%, var(--isip-surface));
}

.isip-badge-accent {
    color: var(--isip-accent);
    border-color: color-mix(in srgb, var(--isip-accent) 45%, transparent);
    background-color: var(--isip-accent-dim);
}

.isip-badge-neutral {
    color: var(--isip-text-muted);
}

/* Small neutral chip for metadata (page numbers, score buckets, ids). */
.isip-chip {
    display: inline-flex;
    align-items: center;
    font-family: var(--isip-font-mono);
    font-size: 0.7rem;
    color: var(--isip-text-muted);
    background-color: var(--isip-surface-2);
    border: 1px solid var(--isip-border);
    border-radius: var(--isip-radius-small);
    padding: 0.1rem 0.4rem;
}

.isip-chip-accent {
    color: var(--isip-accent);
    border-color: color-mix(in srgb, var(--isip-accent) 45%, transparent);
    background-color: var(--isip-accent-dim);
}

/* A hairline rule, for separating rows inside one panel. */
.isip-rule {
    border: none;
    border-top: 1px solid var(--isip-border);
    margin: 0.6rem 0;
}

/* Restrained vertical rhythm: one class, one size, used everywhere. */
.isip-spacer {
    height: var(--isip-spacer-size, 0.75rem);
}

/* Caption-style note under a block. */
.isip-note {
    font-family: var(--isip-font-sans);
    font-size: 0.72rem;
    line-height: 1.45;
    color: var(--isip-text-dim);
    margin: 0.3rem 0 0 0;
}

/* --- the brand mark ----------------------------------------------------- */

/*
 * One mark, two sizes. The tile is pure CSS — a gradient squircle with two
 * `--isip-bg` bars stamped on it, i.e. a ledger line — because the design
 * system forbids structural emoji and ships no image assets: a glyph would
 * depend on a font the container may not have. The bar geometry is in
 * percentages so the hero variant only has to change the box.
 */
.isip-mark {
    display: inline-block;
    position: relative;
    flex: 0 0 auto;
    width: 30px;
    height: 30px;
    border-radius: var(--isip-radius-small);
    background-image: linear-gradient(155deg, var(--isip-accent-strong), var(--isip-accent));
}

.isip-mark::before,
.isip-mark::after {
    content: "";
    position: absolute;
    left: 50%;
    transform: translateX(-50%);
    height: 2px;
    border-radius: 999px;
    background-color: var(--isip-bg);
}

.isip-mark::before {
    top: 33%;
    width: 46%;
}

.isip-mark::after {
    top: 55%;
    width: 28%;
}

/* --- the rail's fixed rows ---------------------------------------------- */

/*
 * Brand row, section label and the bottom identity/status card. Three separate
 * emitters rather than one fragment because Streamlit renders each `st.html`
 * call as its own element container: a `<div>` opened in one call cannot be
 * closed around widgets a later call emits in between.
 */

.isip-rail-brand {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.1rem 0.15rem 0.9rem 0.15rem;
}

.isip-rail-names {
    display: flex;
    flex-direction: column;
    min-width: 0;
    line-height: 1.25;
}

.isip-rail-wordmark {
    font-family: var(--isip-font-sans);
    font-size: 0.95rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--isip-text);
}

.isip-rail-subtitle {
    font-family: var(--isip-font-sans);
    font-size: 0.68rem;
    color: var(--isip-text-dim);
}

/* The muted upper-case section label ("PREVIOUS CONVERSATIONS"). Sits a step
   below `isip-label` in size, because the rail is the app's densest column. */
.isip-rail-label {
    font-family: var(--isip-font-sans);
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--isip-text-dim);
    margin: 0.45rem 0 0.4rem 0.15rem;
}

/* The bottom card: raised one step off the rail's surface so it reads as an
   object status rather than another list row. */
.isip-rail-card {
    background-color: var(--isip-surface-2);
    border: 1px solid var(--isip-border);
    border-radius: var(--isip-radius);
    padding: 0.7rem 0.8rem;
    margin-top: 0.6rem;
}

.isip-rail-card-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 0.5rem;
    margin-bottom: 0.4rem;
}

.isip-rail-card-title {
    font-family: var(--isip-font-sans);
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--isip-text);
}

.isip-rail-card-line {
    font-family: var(--isip-font-sans);
    font-size: 0.72rem;
    line-height: 1.45;
    color: var(--isip-text-muted);
}

/* --- the rail's nav buttons --------------------------------------------- */

/*
 * Four `st.button`s keyed `isip-nav-<view>`. Streamlit's button is a centred
 * flex row, which is right for the pill CTA and wrong for a nav item: these go
 * left-aligned on the rail's own surface, sharing the 8px radius of every other
 * row control in the rail. The rules are scoped to the reserved key token so no
 * other button on the page can pick them up, and Streamlit's own `stSidebarNav`
 * is a different testid entirely.
 *
 * The *selected* view is the one the sidebar renders as `type="primary"`, and
 * Streamlit marks that with `data-testid="stBaseButton-primary"` — NOT with a
 * `.stBaseButton-primary` class, measured in 1.65. That distinction is
 * load-bearing here: the flat-item rules below are `:not(...)`-scoped to the
 * primary, because clearing the background unconditionally would leave the
 * selected item as dark-on-dark text with its fill cancelled out. The selected
 * item keeps the pill rule's fill from the primary-action section above and only
 * takes the squared corners from here, so an active nav item reads as a row
 * rather than as the app's CTA.
 */
[class*="st-key-isip-nav"] button {
    justify-content: flex-start !important;
    text-align: left !important;
    border-radius: var(--isip-radius-small) !important;
    font-weight: 500 !important;
}

/*
 * Streamlit's own button markup centres through TWO nested wrappers:
 * `button > div(justify:center) > span(justify:center)`. Left-aligning the button
 * element alone is not enough — measured in the browser, the labels still sat on a
 * centre line, so "Documents" and "Diagnostics" began at different x positions and
 * the nav read as an off-grid list. Both wrappers are pulled to the start and given
 * the full width the alignment needs to be visible.
 */
[class*="st-key-isip-nav"] button > div,
[class*="st-key-isip-nav"] button > div > span {
    justify-content: flex-start !important;
    width: 100% !important;
}

[class*="st-key-isip-nav"] button:not([data-testid="stBaseButton-primary"]) {
    background-color: transparent !important;
    border-color: transparent !important;
    color: var(--isip-text-muted) !important;
}

[class*="st-key-isip-nav"] button:not([data-testid="stBaseButton-primary"]):hover {
    background-color: var(--isip-surface-2) !important;
    border-color: var(--isip-border) !important;
    color: var(--isip-text) !important;
}

/* --- the hero (empty-state canvas) -------------------------------------- */

/*
 * The canvas's empty state stacks centred fragments: mark, title, subtext,
 * eyebrow, the chip grid, then the footer line under the composer. Each
 * fragment is its own `st.html` container, so centring is per-fragment
 * (`text-align`) plus `margin: auto` for the width-limited ones, rather than a
 * flex parent that cannot exist across containers.
 */

.isip-hero {
    text-align: center;
    margin: 0.5rem 0 0.35rem 0;
}

.isip-hero-mark {
    display: block;
    width: 46px;
    height: 46px;
    margin: 0 auto;
    border-radius: var(--isip-radius-lg);
    box-shadow: 0 8px 24px var(--isip-accent-soft);
}

.isip-hero-mark::before,
.isip-hero-mark::after {
    height: 3px;
}

.isip-hero-title {
    font-family: var(--isip-font-sans);
    font-size: clamp(1.55rem, 3vw, 2.15rem);
    font-weight: 600;
    letter-spacing: -0.02em;
    line-height: 1.15;
    color: var(--isip-text);
    text-align: center;
    margin: 0.2rem 0 0.55rem 0;
}

/* The accented word in the title (the reference layout's coloured name). */
.isip-hero-accent {
    color: var(--isip-accent);
}

.isip-hero-sub {
    max-width: 33rem;
    margin: 0 auto;
    font-family: var(--isip-font-sans);
    font-size: 0.9rem;
    line-height: 1.55;
    color: var(--isip-text-muted);
    text-align: center;
}

/* The small upper-case kicker above the chip grid. */
.isip-eyebrow {
    font-family: var(--isip-font-sans);
    font-size: 0.62rem;
    font-weight: 600;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: var(--isip-text-dim);
    text-align: center;
    margin: 1.4rem 0 0.5rem 0;
}

/* One muted status line under the composer — the app's last word on a page. */
.isip-footer {
    font-family: var(--isip-font-sans);
    font-size: 0.7rem;
    line-height: 1.4;
    color: var(--isip-text-dim);
    text-align: center;
    margin: 0.55rem 0 0.25rem 0;
}

/* --- the hero's demo chips ---------------------------------------------- */

/*
 * Six `st.button`s keyed `isip-suggestion-<n>`, laid out by the caller as a
 * `st.columns(3)` grid. A Streamlit button centres a nowrap label, which is the
 * wrong shape for a suggestion: these go left-aligned with the icon leading, and
 * are allowed to wrap, because the demo questions are full sentences and a third
 * of the canvas is narrow. The height floor keeps the six cells in a grid
 * looking like cells even when one label wraps to two lines.
 */
[class*="st-key-isip-suggestion"] button {
    justify-content: flex-start !important;
    text-align: left !important;
    white-space: normal !important;
    min-height: 3.2rem !important;
    height: auto !important;
    padding: 0.45rem 0.7rem !important;
    line-height: 1.3 !important;
}

[class*="st-key-isip-suggestion"] button [data-testid="stMarkdownContainer"] {
    text-align: left;
    white-space: normal;
}

/* The icon leads the label in the accent colour, the way the mark does — except
   on the selected item, where the fill is already the accent. */
[class*="st-key-isip-nav"] button [data-testid="stIconMaterial"],
[class*="st-key-isip-suggestion"] button [data-testid="stIconMaterial"] {
    color: var(--isip-accent) !important;
    font-size: 1.05rem !important;
}

[class*="st-key-isip-nav"] button[data-testid="stBaseButton-primary"]
    [data-testid="stIconMaterial"] {
    color: #0A0A0B !important;
}

/* --- native widgets recast onto the palette ----------------------------- */

[data-testid="stSelectbox"] div[data-baseweb="select"] > div,
[data-testid="stMultiSelect"] div[data-baseweb="select"] > div {
    background-color: var(--isip-surface-2) !important;
    border-color: var(--isip-border) !important;
    border-radius: var(--isip-radius-small) !important;
    color: var(--isip-text) !important;
    font-family: var(--isip-font-sans) !important;
}

[data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
    background-color: var(--isip-accent) !important;
}

[data-testid="stFileUploaderDropzone"] {
    background-color: var(--isip-surface) !important;
    border: 1px dashed var(--isip-border-strong) !important;
    border-radius: var(--isip-radius) !important;
    color: var(--isip-text-muted) !important;
}

[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--isip-accent) !important;
}

/* Alerts: keep Streamlit's semantics, drop its saturated light-theme colors. */
[data-testid="stAlert"] {
    border-radius: var(--isip-radius-small);
    font-family: var(--isip-font-sans);
}

/* Expanders (the debug panel): 8px radius, hairline border, no shadow. */
[data-testid="stExpander"] details,
[data-testid="stExpander"] summary {
    border-radius: var(--isip-radius-small) !important;
}

[data-testid="stExpander"] {
    border: 1px solid var(--isip-border) !important;
    border-radius: var(--isip-radius-small) !important;
    background-color: var(--isip-surface) !important;
}

/* Toasts: surface, hairline, radius from the scale. */
[data-testid="stToast"] {
    background-color: var(--isip-surface-2);
    border: 1px solid var(--isip-border-strong);
    border-radius: var(--isip-radius-small);
    color: var(--isip-text);
    font-family: var(--isip-font-sans);
}

/* Progress bar: the accent is the only moving, colored thing in the UI. */
[data-testid="stProgressBarTrack"] > div {
    background-color: var(--isip-accent) !important;
}

/* Spinner: same accent, so "working" always reads as the same color. */
[data-testid="stSpinner"] > div {
    border-top-color: var(--isip-accent) !important;
    border-right-color: var(--isip-accent) !important;
}

/* --- motion ------------------------------------------------------------- */

/*
 * One easing curve and one duration for every interactive surface. Streamlit
 * already animates its own widgets; these cover ours plus hover transitions.
 * Nothing in this app animates more than 160ms — motion is feedback, not decor.
 */
.isip-panel,
.isip-badge,
.isip-chip {
    transition:
        background-color 140ms ease,
        border-color 140ms ease,
        color 140ms ease;
}

@media (prefers-reduced-motion: reduce) {
    * {
        transition-duration: 0.01ms !important;
        animation-duration: 0.01ms !important;
    }
}
"""


def inject_theme() -> None:
    block = f"<style>\n{_FONT_IMPORT_CSS}\n{_token_block()}\n{_STYLESHEET_BODY}\n</style>"
    st.markdown(block, unsafe_allow_html=True)


_FONT_IMPORT_CSS = f'@import url("{FONT_IMPORT_URL}");'


def _escape(text: object | None) -> str:
    if text is None:
        return ""
    return html.escape(str(text), quote=True)


def _class_list(*names: str | None) -> str:
    return " ".join(name for name in names if name)


def _emit(markup: str, *, help_text: str | None = None) -> None:
    st.html(markup)
    if help_text:
        st.caption(help_text)


@contextmanager
def panel(
    title: str | None = None,
    *,
    kind: str | None = None,
    extra_classes: str | None = None,
    key: str | None = None,
) -> Iterator[st.delta_generator.DeltaGenerator]:
    classes = _class_list("isip-panel", "isip-panel-2" if kind == "raised" else None, extra_classes)
    container = st.container(border=False, key=key)
    if title:
        label(title)
    with container:
        st.html(f'<div class="{classes}"></div>')
        yield container


def label(text: str) -> None:
    _emit(f'<div class="{_class_list("isip-label")}">{_escape(text)}</div>')


def figure(
    value: object,
    *,
    muted: bool = False,
    large: bool = False,
    prefix: str | None = None,
    suffix: str | None = None,
) -> None:
    classes = _class_list(
        "isip-figure",
        "isip-figure-muted" if muted else None,
        "isip-figure-lg" if large else None,
    )
    body = _escape(prefix) + _escape(value) + _escape(suffix)
    _emit(f'<div class="{classes}">{body}</div>')


def badge(text: str, *, kind: str = "neutral", extra_classes: str | None = None) -> None:
    modifier = f"isip-badge-{kind}"
    if modifier not in CSS_CLASSES:
        modifier = "isip-badge-neutral"
    classes = _class_list("isip-badge", modifier, extra_classes)
    _emit(f'<span class="{classes}">{_escape(text)}</span>')


def caption_note(text: str) -> None:
    _emit(f'<div class="{_class_list("isip-note")}">{_escape(text)}</div>')


def spacer(size: str | None = None) -> None:
    style = f' style="--isip-spacer-size: {_escape(size)}"' if size else ""
    _emit(f'<div class="{_class_list("isip-spacer")}"{style}></div>')


def rail_brand(name: str, subtitle: str | None = None) -> None:
    names = (
        f'<span class="{_class_list("isip-rail-names")}">'
        f'<span class="{_class_list("isip-rail-wordmark")}">{_escape(name)}</span>'
    )
    if subtitle:
        names += f'<span class="{_class_list("isip-rail-subtitle")}">{_escape(subtitle)}</span>'
    names += "</span>"
    _emit(
        f'<div class="{_class_list("isip-rail-brand")}">'
        f'<span class="{_class_list("isip-mark")}" aria-hidden="true"></span>'
        f"{names}"
        "</div>"
    )


def rail_label(text: str) -> None:
    _emit(f'<div class="{_class_list("isip-rail-label")}">{_escape(text)}</div>')


_STATUS_BADGE_KINDS: dict[str, str] = {
    "READY": "success",
    "INDEXED": "success",
    "NO INDEX": "warn",
    "EMPTY": "warn",
    "ERROR": "error",
}


def rail_status_card(title: str, lines: Sequence[str], badge: str | None = None) -> None:
    head = f'<span class="{_class_list("isip-rail-card-title")}">{_escape(title)}</span>'
    if badge:
        kind = _STATUS_BADGE_KINDS.get(badge.strip().upper(), "neutral")
        head += (
            f'<span class="{_class_list("isip-badge", f"isip-badge-{kind}")}">'
            f"{_escape(badge)}</span>"
        )
    body = "".join(
        f'<div class="{_class_list("isip-rail-card-line")}">{_escape(line)}</div>'
        for line in lines
        if line
    )
    _emit(
        f'<div class="{_class_list("isip-rail-card")}">'
        f'<div class="{_class_list("isip-rail-card-head")}">{head}</div>'
        f"{body}"
        "</div>"
    )


def hero_mark() -> None:
    _emit(
        f'<div class="{_class_list("isip-hero")}">'
        f'<span class="{_class_list("isip-mark", "isip-hero-mark")}" aria-hidden="true"></span>'
        "</div>"
    )


def hero_title(plain: str, accent: str | None = None) -> None:
    body = _escape(plain)
    if accent and accent in plain:
        before, _, after = plain.partition(accent)
        body = (
            f"{_escape(before)}"
            f'<span class="{_class_list("isip-hero-accent")}">{_escape(accent)}</span>'
            f"{_escape(after)}"
        )
    _emit(f'<div class="{_class_list("isip-hero-title")}">{body}</div>')


def hero_sub(text: str) -> None:
    _emit(f'<div class="{_class_list("isip-hero-sub")}">{_escape(text)}</div>')


def eyebrow(text: str) -> None:
    _emit(f'<div class="{_class_list("isip-eyebrow")}">{_escape(text)}</div>')


def footer_note(text: str) -> None:
    _emit(f'<div class="{_class_list("isip-footer")}">{_escape(text)}</div>')

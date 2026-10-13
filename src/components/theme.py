"""IsipFin design system: token definitions, stylesheet injection, render helpers.

This module is the single owner of the app's visual language (ADR-009 keeps the UI
Streamlit-native, so the language is delivered as one injected ``<style>`` block
rather than a build step or a component library). Everything downstream —
``src/components/chat.py``, ``src/components/sidebar.py``,
``src/components/upload.py`` — imports from here so the palette, the radius rule
and the type stack are defined exactly once.

What this module exports
------------------------

Injection
    :func:`inject_theme` — emit the whole stylesheet, on *every* run. It must
    never be memoised: Streamlit removes any element a rerun does not re-send, so
    a stylesheet emitted only once is destroyed by the user's first click and the
    app reverts to bare Streamlit chrome (measured in a real browser — see
    :func:`inject_theme`). ``st.html`` sends style-only content to the *event*
    container, so the call costs no vertical space, but ``st.html`` also drops
    the ``<style>`` element itself, which is why the block goes through
    ``st.markdown(..., unsafe_allow_html=True)``.

Tokens
    :data:`BG`, :data:`SURFACE`, :data:`SURFACE_2`, :data:`BORDER`,
    :data:`BORDER_STRONG`, :data:`TEXT`, :data:`TEXT_MUTED`, :data:`TEXT_DIM`,
    :data:`ACCENT`, :data:`ACCENT_STRONG`, :data:`ACCENT_DIM`,
    :data:`ACCENT_SOFT`, :data:`SUCCESS`, :data:`WARNING`, :data:`ERROR`,
    :data:`RADIUS`, :data:`RADIUS_LG`, :data:`RADIUS_SMALL`, :data:`FONT_SANS`,
    :data:`FONT_MONO`, plus :data:`TOKENS` (the name → value mapping the CSS
    block is rendered from).

Render helpers
    :func:`panel`, :func:`label`, :func:`figure`, :func:`badge`,
    :func:`caption_note`, :func:`spacer` — small, side-effect-light HTML emitters
    that share the token values and the radius rule. They are conveniences, not a
    gateway: a component that prefers to write its own markup may import the
    class names (:data:`CSS_CLASSES`) or the tokens directly instead.

Layout helpers
    :func:`rail_brand`, :func:`rail_label`, :func:`rail_status_card` describe the
    rail's three fixed rows (brand mark, section label, workspace card);
    :func:`hero_mark`, :func:`hero_title`, :func:`hero_sub`, :func:`eyebrow`,
    :func:`footer_note` describe the canvas's centred empty state and its footer
    line. They are the same kind of emitter as the render helpers — one HTML
    fragment each, escaped and classified — but they carry the *placement*
    (centred, upper-case, muted) that would otherwise be re-invented per caller.

The rules the stylesheet enforces
---------------------------------
1. **One radius scale.** ``--isip-radius`` (10px) for panels, cards and the primary
   CTA, ``--isip-radius-lg`` (16px) for the composer dock and the brand marks, and
   ``--isip-radius-small`` (8px) for inputs and inline chips. No other size exists,
   on purpose.
2. **Two type roles.** IBM Plex Sans for prose, IBM Plex Mono for every figure,
   table, id and code-ish string — so a reader can tell a number from a word at a
   glance. ``--isip-font-sans`` carries a system fallback stack, so the app still
   renders (and still passes contrast) when Google Fonts is unreachable.
3. **No structural emoji.** Status is carried by :func:`badge` labels and the
   color tokens; the iconless markup keeps the UI accessible and grep-clean.
4. **Reserved widget-key prefixes.** Five prefixes are styled by selector, so
   any *other* widget keyed with one of them silently inherits that styling:
   ``isip-answer…`` / ``isip-question…`` (the chat pane's role containers, and
   the only two that add a border), ``isip-nav…`` (the rail's nav buttons),
   ``isip-suggestion…`` (the hero's demo chips) and ``isip-session…`` (the
   conversation rows). The owning component's module constant is the definition;
   every other component must pick a new prefix rather than borrowing one.
5. **Zero business logic.** This module renders tokens and containers. It never
   reads the store, prompts, retrieval or the LLM gateway (ADR-009's boundary).
"""

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


# --- the palette (black + orange premium financial dark theme) ------------------

#: Near-black base, deliberately not ``#000``: layered surfaces need a floor they
#: can separate from, and pure black collapses ``--isip-surface`` against it.
BG = "#0A0A0B"
#: Panels and cards.
SURFACE = "#131316"
#: Raised rows inside a panel (a citation snippet inside a citation card).
SURFACE_2 = "#1B1B20"
#: Hairlines. ``--isip-border-strong`` is for focused controls and the CTA.
BORDER = "#2A2A31"
BORDER_STRONG = "#3A3A44"

TEXT = "#F5F5F6"
#: Secondary prose (labels, metadata). Contrast on ``BG`` ≈ 6.6:1.
TEXT_MUTED = "#9A9AA3"
#: Tertiary text (timestamps, ids). Contrast on ``SURFACE`` ≈ 3.4:1 — large
#: text and non-essential only, never a figure a user must read.
TEXT_DIM = "#6C6C76"

#: The accent. Orange is the only saturated hue in the system, which is what
#: makes it read as "the action" without a second accent competing for it.
ACCENT = "#FF7A1A"
ACCENT_STRONG = "#FF9347"
#: Tinted wash for selected rows / hovered list items / the citation marker chip.
ACCENT_DIM = "rgba(255,122,26,0.14)"
#: The faintest accent wash: a focus ring / halo behind a control, where
#: ``ACCENT_DIM`` would read as a *selected* state rather than an indication.
ACCENT_SOFT = "rgba(255,122,26,0.10)"

#: Status colors. Kept to the three the UI actually states (ok / warn / error);
#: nothing else is allowed to use them.
SUCCESS = "#3FB950"
WARNING = "#D29922"
ERROR = "#F85149"

#: The one radius scale. Radii is the *only* place a new size would show up.
RADIUS = "10px"
#: The large radius: the composer dock and the brand marks, the two places the
#: design is deliberately softer than a card so they read as *the* focal point.
RADIUS_LG = "16px"
RADIUS_SMALL = "8px"

#: Prose font. The ``-apple-system``/``Segoe UI`` front of the stack means the
#: Windows dev box and the Linux container both get a native sans before the
#: webfont arrives; IBM Plex Sans is the named intent either way.
FONT_SANS = (
    '"IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", '
    'Roboto, "Helvetica Neue", Arial, sans-serif'
)
#: Figures, tables, ids. Same reasoning: the fallbacks are the platform's real
#: monospace stacks, so columns of numbers stay aligned offline.
FONT_MONO = (
    '"IBM Plex Mono", ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, '
    '"Liberation Mono", monospace'
)

#: The Google Fonts request. Two families, one import, one place to change it.
#: A ``display=swap`` keeps text visible while the webfont loads.
FONT_IMPORT_URL = "https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans:wght@400;500;600&display=swap"

#: CSS custom-property name → value. The stylesheet is rendered from this one
#: mapping, so the Python constants and the CSS block can never drift.
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

#: Class names the injected stylesheet defines, exported so a sibling component
#: can emit its own markup against the same rules instead of re-declaring them.
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


# --- the stylesheet ------------------------------------------------------------


def _token_block() -> str:
    """The ``:root`` custom properties, one per line, from :data:`TOKENS`."""
    lines = [f"    {name}: {value};" for name, value in TOKENS.items()]
    return ":root {\n" + "\n".join(lines) + "\n}"


#: Everything after the tokens. Written as a module constant rather than an
#: f-string so the braces in CSS stay literal and readable.
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


# --- injection -----------------------------------------------------------------


def inject_theme() -> None:
    """Emit the design-system stylesheet (tokens + rules) into the current run.

    Called by ``app.py`` on every run, before any component renders, and that
    *every run* is the whole point: Streamlit removes every element a rerun does
    not re-send, so a once-per-session guard does not save a duplicate — it
    deletes the stylesheet. Measured in a real browser on a first load: exactly
    one ``<style>`` node carrying this block, ``--isip-accent`` live, the rail
    label upper-cased; after a single unrelated button click, zero such nodes,
    every custom property gone, and the app back on Streamlit's own chrome. The
    first click a user makes therefore used to strip the design system. Emitting
    unconditionally costs one re-sent element per run and keeps the CSS in place.

    Injection goes through ``st.markdown(..., unsafe_allow_html=True)`` rather
    than ``st.html``, and that is measured, not stylistic: verified in a real
    browser, ``st.html`` renders its container but drops the ``<style>`` element,
    so the whole design system silently did not apply (the app still *looked*
    dark only because Streamlit's own default theme is dark). ``st.html`` is
    still correct for the helper markup below, which does land.

    The base palette is ALSO set natively in ``.streamlit/config.toml``. That is
    not redundant: Streamlit paints its own widgets (sliders, inputs, the canvas)
    from its theme tokens, so CSS alone leaves them on the default red accent
    while our surfaces are orange.

    The fonts arrive as a single ``@import`` inside the same block. That is the
    documented Streamlit way (there is no build step, and a ``<link>`` to a
    local file would be an unread path in the container); the cost is that the
    request serializes behind the stylesheet, which is acceptable at two
    families and is why the fallback stacks exist.
    """
    block = (
        "<style>\n" f"{_FONT_IMPORT_CSS}\n" f"{_token_block()}\n" f"{_STYLESHEET_BODY}\n" "</style>"
    )
    st.markdown(block, unsafe_allow_html=True)


#: ``@import`` must be the first rule in a stylesheet, ahead of ``:root``.
_FONT_IMPORT_CSS = f'@import url("{FONT_IMPORT_URL}");'


# --- HTML primitives -----------------------------------------------------------


def _escape(text: object | None) -> str:
    """Escape ``text`` for HTML, rendering ``None`` as the empty string.

    Every caller passes text that may legitimately be absent (an optional
    prefix or suffix, an unknown parameter), and ``str(None)`` would leak the
    literal "None" into the UI — the classic Streamlit-rendered-null bug.
    """
    if text is None:
        return ""
    return html.escape(str(text), quote=True)


def _class_list(*names: str | None) -> str:
    """Join the non-empty class names into one ``class`` attribute value."""
    return " ".join(name for name in names if name)


def _emit(markup: str, *, help_text: str | None = None) -> None:
    """Write one raw HTML fragment.

    ``st.html`` is the only Streamlit primitive that lets us control the DOM
    (``st.markdown(..., unsafe_allow_html=True)`` would re-parse it as markdown),
    and it sanitizes with DOMPurify — which is why every caller passes escaped
    text and never unsanitized document content.
    """
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
    """A bordered card around a group of widgets, as a ``with`` block.

    Renders ``st.container(border=False)`` and puts a zero-height marker
    ``<div>`` inside it carrying the ``isip-panel`` class, so the card's
    background, border and radius come from the injected stylesheet rather than
    Streamlit's own container skin (which is tuned for the light theme). The
    marker div is what makes this work: ``st.container`` emits no element of its
    own that we can attribute, so the card's box is drawn on an element we own
    and the widgets stream in beside it.

    Parameters
    ----------
    title:
        Optional section heading, rendered as the monospace uppercase
        :func:`label` above the container's contents.
    kind:
        Optional panel modifier. ``"raised"`` lifts the surface to
        ``--isip-surface-2``; any other value is ignored.
    extra_classes:
        Extra CSS classes appended to ``isip-panel``. ``"isip-flush"`` is the
        convention for the sidebar's brand and footer rows, which must not draw
        a second card inside the rail's own surface.
    key:
        A widget key for the underlying container, passed straight through to
        ``st.container``. Uniqueness is the caller's job, as usual.

    Yields
    ------
    The container delta generator, to render widgets into.
    """
    classes = _class_list("isip-panel", "isip-panel-2" if kind == "raised" else None, extra_classes)
    container = st.container(border=False, key=key)
    if title:
        label(title)
    with container:
        st.html(f'<div class="{classes}"></div>')
        yield container


def label(text: str) -> None:
    """Render an uppercase monospace field label (``PREVIOUS CONVERSATIONS``)."""
    _emit(f'<div class="{_class_list("isip-label")}">{_escape(text)}</div>')


def figure(
    value: object,
    *,
    muted: bool = False,
    large: bool = False,
    prefix: str | None = None,
    suffix: str | None = None,
) -> None:
    """Render a figure (number, id, page, score) in the monospace stack.

    Every numeric surface in the UI goes through here so ``tabular-nums``
    applies and the reader can tell figures from prose at a glance.

    Parameters
    ----------
    value:
        The value to render. Escaped, so a filename or an id is safe to pass
        straight through.
    muted:
        Use the muted text color (for secondary figures beside a primary one).
    large:
        The larger figure size, for a headline number such as a revenue total.
    prefix, suffix:
        Optional text placed before / after the value, same styling.
    """
    classes = _class_list(
        "isip-figure",
        "isip-figure-muted" if muted else None,
        "isip-figure-lg" if large else None,
    )
    body = _escape(prefix) + _escape(value) + _escape(suffix)
    _emit(f'<div class="{classes}">{body}</div>')


def badge(text: str, *, kind: str = "neutral", extra_classes: str | None = None) -> None:
    """Render a status pill: the only status icon the UI uses.

    Parameters
    ----------
    text:
        The status word (``indexed``, ``missing``, ``retry``). Rendered
        uppercase by CSS; no emoji is ever added here.
    kind:
        One of ``success``, ``warn``, ``error``, ``accent`` or ``neutral``
        (default). Unknown values fall back to ``neutral`` rather than raising,
        so a new status string cannot break the render path.
    extra_classes:
        Extra CSS classes appended to the badge.
    """
    modifier = f"isip-badge-{kind}"
    if modifier not in CSS_CLASSES:
        modifier = "isip-badge-neutral"
    classes = _class_list("isip-badge", modifier, extra_classes)
    _emit(f'<span class="{classes}">{_escape(text)}</span>')


def caption_note(text: str) -> None:
    """Render a dim caption-style note under a block (hints, fallback reasons)."""
    _emit(f'<div class="{_class_list("isip-note")}">{_escape(text)}</div>')


def spacer(size: str | None = None) -> None:
    """Insert one fixed-height gap in the vertical rhythm.

    Parameters
    ----------
    size:
        Any CSS length (``0.75rem``). Defaults to ``0.75rem``, the stylesheet's
        own ``--isip-spacer-size``.
    """
    style = f' style="--isip-spacer-size: {_escape(size)}"' if size else ""
    _emit(f'<div class="{_class_list("isip-spacer")}"{style}></div>')


# --- layout helpers: the rail's rows -------------------------------------------


def rail_brand(name: str, subtitle: str | None = None) -> None:
    """Render the rail's brand row: the mark, then the wordmark stack.

    Parameters
    ----------
    name:
        The wordmark (``IsipFin``).
    subtitle:
        Optional second line under the wordmark, in the dim text colour.
    """
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
    """Render the rail's upper-case muted section label (``PREVIOUS CONVERSATIONS``).

    Sibling of :func:`label` rather than a call to it: the label lives in the
    rail, which is the app's densest column, so it sits one size smaller and
    owns its own margin.
    """
    _emit(f'<div class="{_class_list("isip-rail-label")}">{_escape(text)}</div>')


#: How a rail status word maps onto a badge kind. The card takes one free-form
#: ``badge`` string — the sidebar states ``READY`` or ``NO INDEX`` and nothing
#: else — so the mapping lives here instead of in the signature. An unlisted
#: word renders the neutral pill rather than raising, for the same reason
#: :func:`badge` falls back: a new status string must not break the rail.
_STATUS_BADGE_KINDS: dict[str, str] = {
    "READY": "success",
    "INDEXED": "success",
    "NO INDEX": "warn",
    "EMPTY": "warn",
    "ERROR": "error",
}


def rail_status_card(title: str, lines: Sequence[str], badge: str | None = None) -> None:
    """Render the rail's bottom identity/status card.

    Parameters
    ----------
    title:
        The card's heading (``Local workspace``).
    lines:
        The facts to state, one muted line each. Blank entries are skipped, so a
        caller can assemble the list conditionally without filtering it.
    badge:
        Optional status word for the head row (``READY``). Mapped onto a badge
        kind by :data:`_STATUS_BADGE_KINDS`; an unlisted word renders neutral.
    """
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


# --- layout helpers: the hero and its footer -----------------------------------


def hero_mark() -> None:
    """Render the centered hero mark: the brand tile at hero size, no wordmark."""
    _emit(
        f'<div class="{_class_list("isip-hero")}">'
        f'<span class="{_class_list("isip-mark", "isip-hero-mark")}" aria-hidden="true"></span>'
        "</div>"
    )


def hero_title(plain: str, accent: str | None = None) -> None:
    """Render the hero's centered headline, optionally two-toned.

    Parameters
    ----------
    plain:
        The whole title text.
    accent:
        A substring of ``plain`` to render in the accent colour. When it does not
        occur in ``plain`` (or is ``None``) the whole title renders in the text
        colour — a caller with a stale word gets a plain title, not a broken one,
        and the markup stays one coherent fragment either way.
    """
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
    """Render the hero's muted subtext, centered and width-limited."""
    _emit(f'<div class="{_class_list("isip-hero-sub")}">{_escape(text)}</div>')


def eyebrow(text: str) -> None:
    """Render a small centered upper-case label (``TRY A DEMO``)."""
    _emit(f'<div class="{_class_list("isip-eyebrow")}">{_escape(text)}</div>')


def footer_note(text: str) -> None:
    """Render the one small muted status line under the composer."""
    _emit(f'<div class="{_class_list("isip-footer")}">{_escape(text)}</div>')

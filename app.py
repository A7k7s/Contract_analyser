"""
Contract Analysis System
========================
A lightweight NLP-powered contract analysis tool using spaCy (en_core_web_sm)
for Named Entity Recognition. No transformers or heavy models required.

Usage:
    streamlit run app.py

Dependencies:
    pip install streamlit spacy pymupdf
    python -m spacy download en_core_web_sm
"""

import re
import json
import io
import streamlit as st
import spacy
from spacy import displacy

# ─────────────────────────────────────────────
#  Page configuration (must be first Streamlit call)
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Contract Analysis System",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────
#  Custom CSS – premium dark/glass aesthetic
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    color: #e2e8f0;
}

/* ── Header banner ── */
.page-header {
    background: linear-gradient(90deg, #1a1a2e, #16213e, #0f3460);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 2rem;
    border: 1px solid rgba(255,255,255,0.08);
    box-shadow: 0 8px 32px rgba(0,0,0,0.4);
}
.page-header h1 {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(90deg, #a78bfa, #60a5fa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0 0 0.4rem 0;
}
.page-header p {
    color: #94a3b8;
    margin: 0;
    font-size: 1rem;
}

/* ── Cards ── */
.glass-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 14px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    backdrop-filter: blur(10px);
    box-shadow: 0 4px 20px rgba(0,0,0,0.3);
}

/* ── Entity pill badges ── */
.entity-pill {
    display: inline-block;
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.15);
    border-radius: 999px;
    padding: 0.25rem 0.75rem;
    margin: 0.2rem;
    font-size: 0.85rem;
    font-weight: 500;
    color: #e2e8f0;
    transition: all 0.2s ease;
}

/* Per-type color accents */
.pill-person  { border-color: #a78bfa; color: #c4b5fd; }
.pill-org     { border-color: #60a5fa; color: #93c5fd; }
.pill-date    { border-color: #34d399; color: #6ee7b7; }
.pill-money   { border-color: #fbbf24; color: #fde68a; }
.pill-loc     { border-color: #f87171; color: #fca5a5; }
.pill-rule    { border-color: #fb923c; color: #fdba74; }

/* ── Metric boxes ── */
.metric-row { display: flex; gap: 1rem; flex-wrap: wrap; margin-bottom: 1.5rem; }
.metric-box {
    flex: 1;
    min-width: 110px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    padding: 1rem 1.2rem;
    text-align: center;
}
.metric-box .m-value { font-size: 2rem; font-weight: 700; }
.metric-box .m-label { font-size: 0.78rem; color: #94a3b8; margin-top: 0.2rem; }

/* ── Section titles ── */
.section-title {
    font-size: 1rem;
    font-weight: 600;
    letter-spacing: 0.05em;
    text-transform: uppercase;
    color: #94a3b8;
    margin: 1.2rem 0 0.6rem 0;
}

/* ── Streamlit overrides ── */
div[data-testid="stTextArea"] textarea {
    background: #f8fafc !important;
    border: 1px solid rgba(0,0,0,0.1) !important;
    border-radius: 10px !important;
    color: #000000 !important;
}
div[data-testid="stFileUploader"] {
    background: rgba(255,255,255,0.04) !important;
    border: 1px dashed rgba(255,255,255,0.2) !important;
    border-radius: 10px !important;
    padding: 1rem !important;
}
div[data-testid="stTabs"] [data-baseweb="tab"] {
    font-size: 0.9rem;
    font-weight: 500;
}
/* Buttons */
div.stButton > button {
    background: linear-gradient(90deg, #7c3aed, #2563eb) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: opacity 0.2s ease !important;
    width: 100%;
}
div.stButton > button:hover { opacity: 0.88 !important; }

/* Download buttons */
div.stDownloadButton > button {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-size: 0.85rem !important;
    width: auto !important;
}

/* displaCy highlight container */
.displacy-container {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    line-height: 2.2;
    font-size: 0.95rem;
    color: #e2e8f0;
    overflow-x: auto;
}
.displacy-container mark.entity {
    border-radius: 4px;
    padding: 0.1em 0.3em;
    font-size: 0.88em;
}

/* Info/warning boxes */
.info-box {
    background: rgba(96,165,250,0.1);
    border-left: 3px solid #60a5fa;
    padding: 0.7rem 1rem;
    border-radius: 6px;
    font-size: 0.88rem;
    color: #93c5fd;
    margin-bottom: 0.8rem;
}
.warn-box {
    background: rgba(251,191,36,0.1);
    border-left: 3px solid #fbbf24;
    padding: 0.7rem 1rem;
    border-radius: 6px;
    font-size: 0.88rem;
    color: #fde68a;
}
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
#  1. MODEL LOADING
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_model() -> spacy.language.Language:
    """
    Load the spaCy small English model (en_core_web_sm).
    Cached so it only loads once across Streamlit re-runs.
    Returns the loaded nlp pipeline, or None on failure.
    """
    try:
        nlp = spacy.load("en_core_web_sm")
        return nlp
    except OSError:
        return None


# ═══════════════════════════════════════════════════════════════
#  2. ENTITY EXTRACTION
# ═══════════════════════════════════════════════════════════════

# Mapping from spaCy label → display category
LABEL_MAP = {
    "PERSON":   "Names",
    "ORG":      "Names",
    "DATE":     "Dates",
    "TIME":     "Dates",
    "MONEY":    "Money",
    "GPE":      "Locations",
    "LOC":      "Locations",
    "FAC":      "Locations",
}

# Labels we want to keep (spaCy raw label filter)
ALLOWED_LABELS = set(LABEL_MAP.keys())


def extract_entities(text: str, nlp) -> dict:
    """
    Run spaCy NER on the input text and return entities grouped by category.

    Args:
        text: Raw contract / legal text.
        nlp:  Loaded spaCy Language model.

    Returns:
        dict with keys: 'Names', 'Dates', 'Money', 'Locations'
        Each value is a list of unique entity strings.
        Also includes 'doc' (the spaCy Doc object) for downstream rendering.
    """
    doc = nlp(text)

    grouped: dict = {
        "Names":     [],
        "Dates":     [],
        "Money":     [],
        "Locations": [],
        "_doc":      doc,   # private — used for displaCy
    }

    seen = set()
    for ent in doc.ents:
        if ent.label_ not in ALLOWED_LABELS:
            continue
        normalized = ent.text.strip()
        key = (ent.label_, normalized.lower())
        if key in seen or not normalized:
            continue
        seen.add(key)
        category = LABEL_MAP[ent.label_]
        grouped[category].append(normalized)

    return grouped


# ═══════════════════════════════════════════════════════════════
#  3. RULE-BASED EXTRACTION
# ═══════════════════════════════════════════════════════════════

# Regex patterns for contract-specific clauses
_EFFECTIVE_DATE_PATTERNS = [
    r"effective\s+(?:as\s+of\s+|date[:\s]+)([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
    r"effective\s+(?:as\s+of\s+|date[:\s]+)(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
    r"(?:this\s+agreement|this\s+contract)\s+(?:is\s+)?effective\s+(?:as\s+of\s+)?([A-Za-z]+\s+\d{1,2},?\s+\d{4})",
    r"(?:commencing|commencement\s+date)[:\s]+([A-Za-z]+\s+\d{1,2},?\s+\d{4}|\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4})",
]

_CONTRACT_VALUE_PATTERNS = [
    r"(?:contract\s+value|total\s+(?:consideration|amount|value|price|fee))[:\s]+[\$£€]?\s*([\d,]+(?:\.\d{1,2})?)",
    r"(?:aggregate\s+amount|maximum\s+amount)[:\s]+[\$£€]?\s*([\d,]+(?:\.\d{1,2})?)",
    r"(?:pay|pays|paid|payment\s+of)\s+[\$£€]\s*([\d,]+(?:\.\d{1,2})?)",
    r"[\$£€]\s*([\d,]+(?:\.\d{1,2})?)\s+(?:per\s+(?:annum|year|month)|annually)",
]

_TERMINATION_PATTERNS = [
    r"(?:term(?:ination)?\s+(?:of\s+)?(?:this\s+)?(?:agreement|contract))[:\s]+(\d+\s+(?:year|month|day)s?)",
    r"(?:initial\s+term|term\s+of\s+(?:the\s+)?agreement)[:\s]+(\d+\s+(?:year|month|day)s?)",
]


def rule_based_extraction(text: str) -> dict:
    """
    Apply domain-specific regex rules to extract contract clauses
    that spaCy NER may miss or mis-classify.

    Args:
        text: Raw contract text.

    Returns:
        dict with keys 'Effective Date', 'Contract Value', 'Contract Term'.
        Values are lists of matched strings (deduplicated).
    """
    lower_text = text.lower()
    results = {
        "Effective Date":  [],
        "Contract Value":  [],
        "Contract Term":   [],
    }

    seen_ed, seen_cv, seen_ct = set(), set(), set()

    for pat in _EFFECTIVE_DATE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val = m.group(1).strip()
            if val.lower() not in seen_ed:
                seen_ed.add(val.lower())
                results["Effective Date"].append(val)

    for pat in _CONTRACT_VALUE_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            raw = m.group(1).strip()
            val = f"${raw}" if not raw.startswith(("$", "£", "€")) else raw
            if val.lower() not in seen_cv:
                seen_cv.add(val.lower())
                results["Contract Value"].append(val)

    for pat in _TERMINATION_PATTERNS:
        for m in re.finditer(pat, text, re.IGNORECASE):
            val = m.group(1).strip()
            if val.lower() not in seen_ct:
                seen_ct.add(val.lower())
                results["Contract Term"].append(val)

    return results


# ═══════════════════════════════════════════════════════════════
#  4. DISPLACY HIGHLIGHT RENDERING
# ═══════════════════════════════════════════════════════════════

# Custom colors per entity label for the displaCy SVG
DISPLACY_COLORS = {
    "PERSON":  "#c084fc",   # purple
    "ORG":     "#60a5fa",   # blue
    "DATE":    "#34d399",   # green
    "TIME":    "#34d399",
    "MONEY":   "#fbbf24",   # amber
    "GPE":     "#f87171",   # red
    "LOC":     "#fb923c",   # orange
    "FAC":     "#fb923c",
}


def highlight_text(doc) -> str:
    """
    Generate displaCy HTML markup for the given spaCy Doc.
    Only entities in ALLOWED_LABELS are rendered.

    Args:
        doc: spaCy Doc object (from extract_entities).

    Returns:
        HTML string with entity spans highlighted.
    """
    # Filter doc.ents to only the labels we care about
    filtered_ents = [e for e in doc.ents if e.label_ in ALLOWED_LABELS]

    # Build the rendered HTML
    html = displacy.render(
        doc,
        style="ent",
        page=False,
        options={
            "ents": list(ALLOWED_LABELS),
            "colors": DISPLACY_COLORS,
        },
    )
    # Wrap in our custom styled div
    return f'<div class="displacy-container">{html}</div>'


# ═══════════════════════════════════════════════════════════════
#  5. DOWNLOAD HELPERS
# ═══════════════════════════════════════════════════════════════

def build_download_json(entities: dict, rule_entities: dict) -> bytes:
    """
    Serialize extracted and rule-based entities into a JSON byte string
    suitable for st.download_button.
    """
    payload = {
        "ner_entities": {
            k: v for k, v in entities.items() if not k.startswith("_")
        },
        "rule_based_entities": rule_entities,
    }
    return json.dumps(payload, indent=2, ensure_ascii=False).encode("utf-8")


def build_download_txt(entities: dict, rule_entities: dict) -> bytes:
    """
    Serialize extracted entities into a plain-text byte string.
    """
    lines = ["CONTRACT ANALYSIS — EXTRACTED ENTITIES", "=" * 45, ""]

    for category, items in entities.items():
        if category.startswith("_") or not items:
            continue
        lines.append(f"[{category.upper()}]")
        for item in items:
            lines.append(f"  • {item}")
        lines.append("")

    lines.append("[RULE-BASED FINDINGS]")
    for clause, items in rule_entities.items():
        if items:
            lines.append(f"  {clause}:")
            for item in items:
                lines.append(f"    – {item}")
    return "\n".join(lines).encode("utf-8")


# ═══════════════════════════════════════════════════════════════
#  6. DISPLAY FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def render_metric_row(entities: dict) -> None:
    """Render a horizontal row of entity-count metric boxes."""
    categories = [
        ("Names",     "👤", "#c084fc"),
        ("Dates",     "📅", "#34d399"),
        ("Money",     "💰", "#fbbf24"),
        ("Locations", "📍", "#f87171"),
    ]
    boxes_html = '<div class="metric-row">'
    for cat, icon, color in categories:
        count = len(entities.get(cat, []))
        boxes_html += f"""
        <div class="metric-box">
            <div class="m-value" style="color:{color}">{icon} {count}</div>
            <div class="m-label">{cat}</div>
        </div>"""
    boxes_html += "</div>"
    st.markdown(boxes_html, unsafe_allow_html=True)


def render_pill_list(items: list, pill_class: str) -> str:
    """Return an HTML string of entity pill badges."""
    if not items:
        return '<span style="color:#64748b;font-size:0.88rem;">No entities found.</span>'
    return "".join(
        f'<span class="entity-pill {pill_class}">{item}</span>'
        for item in items
    )


def display_entities(entities: dict, rule_entities: dict) -> None:
    """
    Render the full results UI:
      - Metric summary row
      - Tabbed entity sections (Names / Dates / Money / Locations / Rule-Based)
      - displaCy highlighted text
      - Download buttons
    """
    # ── Metric row ──────────────────────────────────────────
    render_metric_row(entities)

    # ── Entity tabs ─────────────────────────────────────────
    tab_names = ["👤 Names", "📅 Dates", "💰 Money", "📍 Locations", "🔍 Rule-Based"]
    tabs = st.tabs(tab_names)

    config = [
        ("Names",     "pill-person"),
        ("Dates",     "pill-date"),
        ("Money",     "pill-money"),
        ("Locations", "pill-loc"),
    ]

    for tab, (cat, pill_cls) in zip(tabs[:4], config):
        with tab:
            items = entities.get(cat, [])
            st.markdown(
                f'<div class="glass-card">{render_pill_list(items, pill_cls)}</div>',
                unsafe_allow_html=True,
            )
            if items:
                st.caption(f"Total: **{len(items)}** unique {cat.lower()} found.")

    with tabs[4]:
        found_any = any(v for v in rule_entities.values())
        if not found_any:
            st.markdown(
                '<div class="glass-card"><span style="color:#64748b;font-size:0.88rem;">'
                "No rule-based clause patterns matched. Try a longer contract snippet.</span></div>",
                unsafe_allow_html=True,
            )
        else:
            html = '<div class="glass-card">'
            clause_colors = {
                "Effective Date": "pill-date",
                "Contract Value": "pill-money",
                "Contract Term":  "pill-rule",
            }
            for clause, items in rule_entities.items():
                if items:
                    html += f'<div class="section-title">{clause}</div>'
                    html += render_pill_list(items, clause_colors.get(clause, "pill-rule"))
            html += "</div>"
            st.markdown(html, unsafe_allow_html=True)

    # ── displaCy highlighted text ────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-title">🎨 Entity Highlights in Original Text</div>',
        unsafe_allow_html=True,
    )
    doc = entities.get("_doc")
    if doc:
        highlighted_html = highlight_text(doc)
        st.markdown(highlighted_html, unsafe_allow_html=True)

    # ── Download buttons ─────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div class="section-title">⬇️ Download Results</div>',
        unsafe_allow_html=True,
    )
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.download_button(
            label="📥 Download JSON",
            data=build_download_json(entities, rule_entities),
            file_name="contract_entities.json",
            mime="application/json",
        )
    with col2:
        st.download_button(
            label="📄 Download TXT",
            data=build_download_txt(entities, rule_entities),
            file_name="contract_entities.txt",
            mime="text/plain",
        )


# ═══════════════════════════════════════════════════════════════
#  7. TEXT EXTRACTION UTILITIES
# ═══════════════════════════════════════════════════════════════

def read_txt_file(uploaded_file) -> str:
    """Decode an uploaded .txt file to string."""
    raw_bytes = uploaded_file.read()
    for enc in ("utf-8", "latin-1", "cp1252"):
        try:
            return raw_bytes.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw_bytes.decode("utf-8", errors="replace")


def read_pdf_file(uploaded_file) -> str:
    """
    Extract text from a PDF using PyMuPDF (fitz).
    Returns empty string if PyMuPDF is not installed.
    """
    try:
        import fitz  # PyMuPDF
        pdf_bytes = uploaded_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        pages_text = []
        for page in doc:
            pages_text.append(page.get_text())
        doc.close()
        return "\n".join(pages_text)
    except ImportError:
        return ""
    except Exception:
        return ""


# ═══════════════════════════════════════════════════════════════
#  8. MAIN APP
# ═══════════════════════════════════════════════════════════════

def main():
    # ── Header ──────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
        <h1>📄 Contract Analysis System</h1>
        <p>Extract structured information from legal and contract text using AI-powered Named Entity Recognition (NER). Fast, lightweight, no internet required.</p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Model ───────────────────────────────────────────
    with st.spinner("Loading NLP model…"):
        nlp = load_model()

    if nlp is None:
        st.error(
            "❌ **spaCy model not found.**\n\n"
            "Run the following commands to install it:\n"
            "```bash\n"
            "pip install spacy\n"
            "python -m spacy download en_core_web_sm\n"
            "```"
        )
        st.stop()

    # ── Sidebar info ────────────────────────────────────────
    with st.sidebar:
        st.markdown("### ℹ️ About")
        st.markdown(
            "**Model:** `en_core_web_sm`  \n"
            "**Framework:** spaCy  \n"
            "**No internet required**  \n\n"
            "**Detected entity types:**"
        )
        categories = {
            "👤 Names":     "PERSON · ORG",
            "📅 Dates":     "DATE · TIME",
            "💰 Money":     "MONEY",
            "📍 Locations": "GPE · LOC · FAC",
            "🔍 Rule-Based":"Effective Date · Contract Value · Term",
        }
        for label, types in categories.items():
            st.markdown(f"- **{label}** — `{types}`")

        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown(
            "- Paste **2–3 paragraphs** minimum for best results\n"
            "- Upload a `.txt` or `.pdf` contract file\n"
            "- Rule-based extraction looks for clause keywords"
        )

    # ── Input Section ────────────────────────────────────────
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    input_tab, upload_tab = st.tabs(["✍️ Paste Contract Text", "📂 Upload File"])

    contract_text = ""

    with input_tab:
        pasted = st.text_area(
            label="Contract / Legal Text",
            placeholder=(
                "Paste your contract text here…\n\n"
                "Example: This Service Agreement (\"Agreement\") is entered into as of January 1, 2024, "
                "between Acme Corporation, a Delaware corporation (\"Company\"), and John Smith "
                "(\"Contractor\"), residing at 123 Main Street, New York, NY 10001…"
            ),
            height=240,
            label_visibility="collapsed",
        )
        if pasted.strip():
            contract_text = pasted.strip()

    with upload_tab:
        uploaded = st.file_uploader(
            "Upload a contract file",
            type=["txt", "pdf"],
            help="Supported formats: .txt, .pdf (PDF requires PyMuPDF)",
            label_visibility="collapsed",
        )
        if uploaded is not None:
            file_ext = uploaded.name.lower().rsplit(".", 1)[-1]
            if file_ext == "txt":
                contract_text = read_txt_file(uploaded)
                st.success(f"✅ Loaded **{uploaded.name}** ({len(contract_text):,} characters)")
            elif file_ext == "pdf":
                extracted = read_pdf_file(uploaded)
                if extracted:
                    contract_text = extracted
                    st.success(f"✅ Loaded **{uploaded.name}** ({len(contract_text):,} characters)")
                else:
                    st.markdown(
                        '<div class="warn-box">⚠️ PDF text extraction failed. '
                        "Install PyMuPDF: <code>pip install pymupdf</code></div>",
                        unsafe_allow_html=True,
                    )

    st.markdown("</div>", unsafe_allow_html=True)

    # Preview extracted file text
    if contract_text and uploaded is not None:
        with st.expander("📋 Preview extracted text", expanded=False):
            st.text(contract_text[:3000] + ("…" if len(contract_text) > 3000 else ""))

    # ── Analyze Button ───────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    analyze_clicked = st.button("🔍 Analyze Contract", use_container_width=True)

    # ── Results ──────────────────────────────────────────────
    if analyze_clicked:
        if not contract_text.strip():
            st.markdown(
                '<div class="warn-box">⚠️ Please paste some contract text or upload a file before analyzing.</div>',
                unsafe_allow_html=True,
            )
            st.stop()

        if len(contract_text.strip()) < 30:
            st.markdown(
                '<div class="warn-box">⚠️ Input is too short. Please provide more contract text for meaningful extraction.</div>',
                unsafe_allow_html=True,
            )
            st.stop()

        with st.spinner("Analyzing contract…"):
            entities     = extract_entities(contract_text, nlp)
            rule_results = rule_based_extraction(contract_text)

        total_ner = sum(
            len(v) for k, v in entities.items() if not k.startswith("_")
        )

        if total_ner == 0 and not any(rule_results.values()):
            st.markdown(
                '<div class="info-box">ℹ️ No entities were detected. '
                "The text may be too abstract or use unusual formatting. "
                "Try pasting actual contract language with names, dates, and amounts.</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="info-box">✅ Analysis complete — '
                f"<strong>{total_ner}</strong> unique NER entities extracted.</div>",
                unsafe_allow_html=True,
            )
            st.markdown("---")
            display_entities(entities, rule_results)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    main()

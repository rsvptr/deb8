# Deb8 - a clickbait detector for news headlines.
# The model and vectorizer are trained in notebooks/07-modeling-interpretation-tuning.ipynb;
# this app only loads them and serves predictions.

import base64
import io
import pickle
import re
import string
import warnings
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image
from scipy import sparse

warnings.filterwarnings('ignore')

BASE_DIR = Path(__file__).parent

# App icon: used as the browser favicon and inlined into the hero.
_logo = Image.open(BASE_DIR / 'docs' / 'logo.png')
_logo.thumbnail((192, 192), Image.LANCZOS)
_buf = io.BytesIO()
_logo.save(_buf, format='PNG')
LOGO_B64 = base64.b64encode(_buf.getvalue()).decode()

# ---------------------------------------------------------------------------
# Page setup and styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Deb8 · Clickbait Detector",
    page_icon=_logo,
    layout="centered",
    menu_items={
        "About": "Deb8 is an NLP and machine learning based clickbait detector. "
                 "Source code: https://github.com/rsvptr/deb8",
    },
)

st.markdown(
    """
    <style>
    /* Layered gradient backdrop (self-contained, no external images).
       Base: vertical navy fade + edge vignette. On top: four soft aurora
       blobs that drift slowly, plus a faint grain layer to prevent banding. */
    .stApp {
        background:
            radial-gradient(130% 100% at 50% 20%, transparent 55%, rgba(5, 7, 12, 0.7) 100%),
            linear-gradient(180deg, #0e1426 0%, #0b0f19 55%, #090c14 100%);
        background-attachment: fixed;
    }
    .stApp::before {
        content: "";
        position: fixed;
        inset: -20%;
        z-index: -1;
        pointer-events: none;
        background:
            radial-gradient(38% 30% at 22% 12%, rgba(240, 176, 41, 0.17), transparent 70%),
            radial-gradient(30% 26% at 82% 8%, rgba(122, 162, 255, 0.15), transparent 70%),
            radial-gradient(36% 32% at 78% 78%, rgba(64, 201, 162, 0.08), transparent 70%),
            radial-gradient(28% 24% at 12% 82%, rgba(154, 106, 255, 0.09), transparent 70%);
    }
    @media (prefers-reduced-motion: no-preference) {
        .stApp::before {
            animation: aurora-drift 70s ease-in-out infinite alternate;
        }
    }
    @keyframes aurora-drift {
        0%   { transform: translate(0, 0) scale(1); }
        50%  { transform: translate(2.5%, -2%) scale(1.06); }
        100% { transform: translate(-2%, 2.5%) scale(1.02); }
    }
    .stApp::after {
        content: "";
        position: fixed;
        inset: 0;
        z-index: -1;
        pointer-events: none;
        opacity: 0.05;
        background-image: url("data:image/svg+xml;utf8,<svg xmlns='http://www.w3.org/2000/svg' width='160' height='160'><filter id='n'><feTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/></filter><rect width='100%25' height='100%25' filter='url(%23n)' opacity='0.6'/></svg>");
    }

    /* Let the gradient show through Streamlit's chrome */
    [data-testid="stHeader"] {background: transparent;}
    #MainMenu, footer {visibility: hidden;}

    /* Glass card around the input form */
    div[data-testid="stForm"] {
        background: rgba(18, 23, 38, 0.55);
        border: 1px solid rgba(122, 162, 255, 0.16);
        border-radius: 1rem;
        padding: 1.35rem 1.35rem 1.05rem;
        box-shadow: 0 18px 50px rgba(0, 0, 0, 0.35);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
    }
    .stTextArea textarea {
        background: rgba(9, 12, 20, 0.65) !important;
        border-radius: 0.6rem !important;
    }

    /* Sidebar and expander pick up the same translucent panel look */
    section[data-testid="stSidebar"] {
        background: rgba(13, 17, 29, 0.88);
        border-right: 1px solid rgba(122, 162, 255, 0.1);
    }
    div[data-testid="stExpander"] > details {
        background: rgba(18, 23, 38, 0.45);
        border: 1px solid rgba(122, 162, 255, 0.12);
        border-radius: 0.75rem;
    }

    /* Hero */
    /* Streamlit renders markdown images at inline-icon size; force real dimensions */
    .hero-logo {
        width: 104px !important;
        height: 104px !important;
        border-radius: 20px;
        box-shadow: 0 14px 44px rgba(240, 176, 41, 0.16), 0 4px 16px rgba(0, 0, 0, 0.4);
        margin-bottom: 1rem;
        display: block;
    }
    .hero-title {
        font-size: 3.2rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        margin-bottom: 0;
        background: linear-gradient(90deg, #f0b429, #f6d365 55%, #7aa2ff);
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-tagline {
        font-size: 1.15rem;
        color: rgba(232, 235, 244, 0.78);
        margin-top: 0.35rem;
        margin-bottom: 1.1rem;
    }
    @media (max-width: 640px) {
        .hero-title {font-size: 2.4rem;}
        .hero-logo {width: 84px !important; height: 84px !important;}
    }

    /* Small pill chips */
    .chip {
        display: inline-block;
        padding: 0.22rem 0.7rem;
        margin: 0 0.4rem 0.4rem 0;
        border: 1px solid rgba(240, 176, 41, 0.35);
        border-radius: 999px;
        font-size: 0.78rem;
        color: rgba(232, 235, 244, 0.85);
        background: rgba(240, 176, 41, 0.07);
        white-space: nowrap;
    }

    /* Verdict cards */
    .verdict {
        border-radius: 0.75rem;
        padding: 1.1rem 1.3rem;
        margin: 1rem 0 0.5rem 0;
        border: 1px solid;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }
    .verdict h3 {
        margin: 0 0 0.35rem 0;
        font-size: 1.25rem;
    }
    .verdict p {
        margin: 0;
        color: rgba(232, 235, 244, 0.82);
        font-size: 0.95rem;
        line-height: 1.5;
    }
    .verdict-clickbait {
        background: rgba(255, 92, 92, 0.09);
        border-color: rgba(255, 92, 92, 0.45);
    }
    .verdict-clickbait h3 {color: #ff8080;}
    .verdict-genuine {
        background: rgba(61, 220, 151, 0.08);
        border-color: rgba(61, 220, 151, 0.4);
    }
    .verdict-genuine h3 {color: #4fe3a3;}

    /* Signal chips inside the details expander */
    .signal {
        display: inline-block;
        padding: 0.3rem 0.75rem;
        margin: 0.15rem 0.35rem 0.15rem 0;
        border-radius: 0.5rem;
        font-size: 0.82rem;
        border: 1px solid rgba(122, 162, 255, 0.35);
        background: rgba(122, 162, 255, 0.08);
        color: rgba(232, 235, 244, 0.9);
    }
    .signal.on {
        border-color: rgba(240, 176, 41, 0.55);
        background: rgba(240, 176, 41, 0.12);
    }

    /* Full-width submit button (Streamlit shrink-wraps button containers by default) */
    div[data-testid="stForm"] div[data-testid="stElementContainer"],
    div[data-testid="stForm"] div[data-testid="stElementContainer"] > div,
    div[data-testid="stFormSubmitButton"] {
        width: 100% !important;
    }
    div[data-testid="stFormSubmitButton"] button {
        width: 100% !important;
        border-radius: 0.6rem;
        font-weight: 600;
    }

    /* Footer note */
    .footnote {
        color: rgba(232, 235, 244, 0.45);
        font-size: 0.8rem;
        text-align: center;
        margin-top: 2.5rem;
    }
    .footnote a {color: rgba(240, 176, 41, 0.75); text-decoration: none;}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Model loading (cached so reruns don't hit the disk again)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading model…")
def load_artifacts():
    with open(BASE_DIR / 'models' / 'naive_bayes_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open(BASE_DIR / 'models' / 'tfidf_vectorizer.pkl', 'rb') as vec_file:
        vectorizer = pickle.load(vec_file)
    return model, vectorizer


model, vectorizer = load_artifacts()

# ---------------------------------------------------------------------------
# Text preprocessing and feature engineering
# (mirrors the training pipeline in notebook 06/07 - do not alter)
# ---------------------------------------------------------------------------

def clean_text_round1(text):
    """
    Cleans the input text by converting it to lowercase, removing URLs, newline
    characters, multiple spaces, punctuation, and certain special characters.
    """
    text = text.lower()
    text = re.sub('\n', ' ', text)
    text = re.sub('  ', ' ', text)
    text = re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', text)
    text = re.sub(r'\[.*?\]', ' ', text)
    text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub('“', '', text)
    text = re.sub('”', '', text)
    text = re.sub('’', '', text)
    text = re.sub('–', '', text)
    text = re.sub('‘', '', text)
    return text


def contains_question(headline):
    """Returns 1 if the headline contains a question mark or starts with common interrogative words; otherwise 0."""
    if "?" in headline or headline.startswith(('who', 'what', 'where', 'why', 'when', 'whose', 'whom', 'would', 'will', 'how', 'which', 'should', 'could', 'did', 'do')):
        return 1
    return 0


def contains_exclamation(headline):
    """Returns 1 if the headline contains an exclamation mark; otherwise 0."""
    return 1 if "!" in headline else 0


def starts_with_num(headline):
    """Returns 1 if the headline starts with a number; otherwise 0."""
    return 1 if headline and headline[0].isdigit() else 0


def classify(headline):
    """Runs a raw headline through the full pipeline and returns the verdict plus the signals used."""
    cleaned = clean_text_round1(headline)
    signals = {
        'question': contains_question(cleaned),
        'exclamation': contains_exclamation(cleaned),
        'starts_with_num': starts_with_num(cleaned),
        'headline_words': len(cleaned.split()),
    }

    vectorized = vectorizer.transform([cleaned])
    numeric_features = np.array([[signals['question'], signals['exclamation'],
                                  signals['starts_with_num'], signals['headline_words']]])
    final_features = sparse.hstack([sparse.csr_matrix(numeric_features), vectorized])

    prediction = model.predict(final_features)[0]
    return prediction, cleaned, signals


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### How it works")
    st.markdown(
        """
        1. Your headline is lowercased and stripped of URLs, punctuation, and stray characters.
        2. Four signals are computed: word count, question phrasing, exclamation marks, and whether it opens with a number.
        3. The cleaned text is TF-IDF vectorized (unigrams and bigrams) and a Multinomial Naive Bayes model returns the verdict.
        """
    )
    st.markdown("### Under the hood")
    st.markdown(
        """
        - Trained on ~52,000 headlines (2007-2020)
        - Clickbait from BuzzFeed, Upworthy, Bored Panda and similar; genuine headlines from Reuters, NYT, The Guardian and others
        - 93% test accuracy, 94% recall on the clickbait class
        """
    )
    st.markdown("### Links")
    st.markdown(
        """
        - [Source code on GitHub](https://github.com/rsvptr/deb8)
        - [Project report (PDF)](https://github.com/rsvptr/deb8/blob/main/docs/deb8-project-report.pdf)
        """
    )
    st.caption(
        "Deb8 judges phrasing, not truth. A calm headline over a false story will pass; "
        "an excitable headline over a real one may not."
    )

# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------

st.markdown(
    f'<img src="data:image/png;base64,{LOGO_B64}" class="hero-logo" alt="Deb8 logo" />'
    '<h1 class="hero-title">Deb8</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="hero-tagline">A clickbait detector for news headlines. '
    'Paste one below and see whether it reads like the real thing or like bait.</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<span class="chip">TF-IDF + n-grams</span>'
    '<span class="chip">Multinomial Naive Bayes</span>'
    '<span class="chip">~52k training headlines</span>',
    unsafe_allow_html=True,
)

with st.form("headline_form"):
    sentence = st.text_area(
        "Headline",
        placeholder='e.g. "17 Things Only People Who Hate Mornings Will Understand"',
        height=100,
    )
    submitted = st.form_submit_button("Analyze headline", type="primary")

if submitted:
    if not sentence.strip():
        st.warning("Enter a headline first - there is nothing to analyze yet.")
    else:
        prediction, cleaned, signals = classify(sentence)

        if prediction == 1:
            st.markdown(
                """
                <div class="verdict verdict-clickbait">
                    <h3>🚨 Clickbait detected</h3>
                    <p>This headline reads like it was written to lure clicks rather than
                    to inform. Expect less substance than it promises.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                """
                <div class="verdict verdict-genuine">
                    <h3>✅ Looks genuine</h3>
                    <p>This headline reads like straightforward news writing - no obvious
                    bait in its phrasing. Happy reading!</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with st.expander("What the model saw"):
            st.markdown("**Cleaned text passed to the vectorizer:**")
            st.code(cleaned if cleaned.strip() else "(empty after cleaning)", language=None)
            st.markdown("**Engineered signals:**")
            st.markdown(
                f'<span class="signal">{signals["headline_words"]} words</span>'
                f'<span class="signal {"on" if signals["question"] else ""}">question: {"yes" if signals["question"] else "no"}</span>'
                f'<span class="signal {"on" if signals["exclamation"] else ""}">exclamation: {"yes" if signals["exclamation"] else "no"}</span>'
                f'<span class="signal {"on" if signals["starts_with_num"] else ""}">starts with a number: {"yes" if signals["starts_with_num"] else "no"}</span>',
                unsafe_allow_html=True,
            )
            st.caption(
                "These four signals are stacked alongside the TF-IDF vector of the cleaned "
                "text, exactly as during training."
            )

st.markdown(
    '<p class="footnote">An undergraduate NLP project · '
    '<a href="https://github.com/rsvptr/deb8">GitHub</a> · MIT License</p>',
    unsafe_allow_html=True,
)

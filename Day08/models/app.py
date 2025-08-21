# app.py
import streamlit as st
import pickle
import docx
import os
import pickle
import gdown
import streamlit as st

import PyPDF2
import re
from io import BytesIO
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from datetime import datetime
import gdown


st.set_page_config(
    page_title="Resume Category Prediction",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_css():
    st.markdown(
        """
        <style>
        /* Global */
        .stApp {
            background: radial-gradient(1200px 600px at 0% -10%, #3a7bd522 10%, transparent 60%),
                        radial-gradient(1200px 600px at 100% 110%, #6ee7b722 10%, transparent 60%),
                        linear-gradient(120deg, #0f172a, #0b1024 55%, #0f172a);
            color: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background: rgba(15, 23, 42, 0.6) !important;
            backdrop-filter: blur(12px);
            border-right: 1px solid rgba(148, 163, 184, 0.2);
        }
        .glass {
            background: rgba(2, 6, 23, 0.5);
            backdrop-filter: blur(14px);
            border: 1px solid rgba(148, 163, 184, 0.18);
            border-radius: 18px;
            padding: 1.1rem 1.2rem;
            box-shadow: 0 10px 30px rgba(2, 8, 23, 0.4);
        }
        .hero {
            border-radius: 22px;
            padding: 2rem 2.2rem;
            background: linear-gradient(135deg, rgba(59,130,246,0.20), rgba(56,189,248,0.12));
            border: 1px solid rgba(148, 163, 184, 0.25);
            box-shadow: 0 30px 60px rgba(30, 64, 175, 0.25), inset 0 0 40px rgba(56,189,248,0.05);
        }
        .headline {
            font-size: 1.9rem;
            font-weight: 800;
            letter-spacing: 0.2px;
            margin: 0;
            color: #e2e8f0;
        }
        .subhead {
            margin-top: 6px;
            color: #cbd5e1;
            font-size: 0.98rem;
        }
        .metric-card {
            border-radius: 18px;
            padding: 1rem 1.2rem;
            background: rgba(2,6,23,0.45);
            border: 1px solid rgba(148, 163, 184, 0.18);
        }
        .metric-title {
            color: #94a3b8;
            font-size: 0.82rem;
            margin-bottom: 4px;
            letter-spacing: 0.3px;
        }
        .metric-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: #e2e8f0;
        }
        .pill {
            display: inline-block;
            padding: 6px 12px;
            border-radius: 999px;
            margin: 4px 6px 0 0;
            border: 1px solid rgba(148, 163, 184, 0.25);
            background: rgba(15, 23, 42, 0.5);
            color: #e5e7eb;
            font-size: 0.86rem;
        }
        .pill.good { border-color: rgba(34,197,94,0.55); background: rgba(34,197,94,0.12); }
        .pill.warn { border-color: rgba(245,158,11,0.55); background: rgba(245,158,11,0.12); }
        .pill.bad  { border-color: rgba(239,68,68,0.55);  background: rgba(239,68,68,0.12); }
        .section-title {
            font-weight: 800;
            font-size: 1.25rem;
            letter-spacing: 0.2px;
            margin-bottom: 0.4rem;
            color: #e2e8f0;
        }
        .divider {
            border-bottom: 1px solid rgba(148, 163, 184, 0.18);
            margin: 0.7rem 0 1.2rem 0;
        }
        .footer {
            text-align: center;
            color: #94a3b8;
            font-size: 0.85rem;
            margin-top: 1.5rem;
        }
        /* Progress Ring holder */
        .ring-wrap {
            display: flex; align-items: center; justify-content: center;
            width: 160px; height: 160px; margin: 0 auto;
        }
        /* File uploader card tweak */
        .uploadedFile { color: #e2e8f0 !important; }
        </style>
        """,
        unsafe_allow_html=True,
    )

inject_css()


MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Google Drive file IDs
files = {
    "clf.pkl": "13ctuS4M_Cdu6maLesyjVJKrjo2Tg7Ojf",
    "tfidf.pkl": "1uc7rAOlmizZS3_V2_CA_0ED-Am8LEbAm",
    "encoder.pkl": "18X1BfAlOOS77V3uVfHttsEHaiDZy7SXT"
}

def download_if_missing(file_name, file_id):
    dest = os.path.join(MODEL_DIR, file_name)
    if os.path.exists(dest):
        # Just skip silently without writing to app
        return dest
    url = f"https://drive.google.com/uc?id={file_id}"
    gdown.download(url, dest, quiet=True)  # quiet=True avoids printing
    if not os.path.exists(dest):
        raise SystemExit(f"Failed to download {file_name}.")
    return dest

# Download required files (only if missing)
clf_path = download_if_missing("clf.pkl", files["clf.pkl"])
tfidf_path = download_if_missing("tfidf.pkl", files["tfidf.pkl"])
encoder_path = download_if_missing("encoder.pkl", files["encoder.pkl"])

# Now load
svc_model = pickle.load(open(clf_path, "rb"))
tfidf = pickle.load(open(tfidf_path, "rb"))
le = pickle.load(open(encoder_path, "rb"))


def cleanResume(txt):
    cleanText = re.sub(r'http\S+\s', ' ', txt)
    cleanText = re.sub(r'RT|cc', ' ', cleanText)
    cleanText = re.sub(r'#\S+\s', ' ', cleanText)
    cleanText = re.sub(r'@\S+', '  ', cleanText)
    cleanText = re.sub("[%s]" % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), " ", cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub(r'\s+', ' ', cleanText)
    return cleanText

def extract_keywords(text, top_n=20):
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    stopwords = set([
        "and", "or", "with", "for", "a", "an", "the", "in", "on", "to",
        "of", "is", "are", "as", "at", "by", "be", "this", "that", "from"
    ])
    filtered_words = [w for w in words if w not in stopwords and len(w) > 2]
    common_words = Counter(filtered_words).most_common(top_n)
    return [w for w, _ in common_words]

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    text = ''
    for page in reader.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted + "\n"
    return text

def extract_text_from_docx(file):
    docx_doc = docx.Document(file)
    return "\n".join([p.text for p in docx_doc.paragraphs])

def extract_text_from_txt(file):
    content = file.read()
    try:
        return content.decode('utf-8')
    except UnicodeDecodeError:
        return content.decode('latin-1')

def handle_file_upload(uploaded_file):
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext == 'pdf':
        return extract_text_from_pdf(uploaded_file)
    elif ext == 'docx':
        return extract_text_from_docx(uploaded_file)
    elif ext == 'txt':
        return extract_text_from_txt(uploaded_file)
    else:
        raise ValueError("Unsupported file type. Please upload a PDF, DOCX, or TXT file.")

def pred(input_resume):
    cleaned_text = cleanResume(input_resume)
    vectorized_text = tfidf.transform([cleaned_text]).toarray()
    predicted_category = svc_model.predict(vectorized_text)
    return le.inverse_transform(predicted_category)[0]

def match_resume_with_jd(resume_text, job_desc):
    resume_clean = cleanResume(resume_text)
    jd_clean = cleanResume(job_desc)
    resume_vec = tfidf.transform([resume_clean])
    jd_vec = tfidf.transform([jd_clean])
    similarity = cosine_similarity(resume_vec, jd_vec)[0][0]
    return round(similarity * 100, 2)


def progress_ring(percent: float):
    """
    SVG circular progress ring rendered via HTML.
    """
    pct = max(0, min(100, percent))
    radius = 60
    circumference = 2 * 3.14159 * radius
    offset = circumference * (1 - pct / 100.0)
    color = "#22c55e" if pct >= 75 else ("#f59e0b" if pct >= 50 else "#ef4444")

    ring_html = f"""
    <div class="ring-wrap glass">
      <svg width="160" height="160">
        <circle cx="80" cy="80" r="{radius}" stroke="rgba(148,163,184,0.25)" stroke-width="12" fill="none" />
        <circle cx="80" cy="80" r="{radius}" stroke="{color}" stroke-width="12" fill="none"
                stroke-linecap="round"
                stroke-dasharray="{circumference}"
                stroke-dashoffset="{offset}"
                transform="rotate(-90 80 80)" />
        <text x="50%" y="52%" dominant-baseline="middle" text-anchor="middle"
              font-size="28" font-weight="700" fill="#e2e8f0">{pct:.0f}%</text>
        <text x="50%" y="69%" dominant-baseline="middle" text-anchor="middle"
              font-size="12" fill="#94a3b8">Match</text>
      </svg>
    </div>
    """
    st.markdown(ring_html, unsafe_allow_html=True)

def pills(items, kind="good"):
    if not items:
        return
    html = "".join([f'<span class="pill {kind}">{st.session_state.get("_esc_", lambda x:x)(w) if False else w}</span>' for w in items])
    st.markdown(html, unsafe_allow_html=True)

def metric_card(title, value):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">{title}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

def make_report(category, match_score, matched, missing):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    lines = [
        f"Resume Category Prediction Report",
        f"Generated: {ts}",
        "",
        f"Predicted Category: {category}",
        f"JD Match Score: {match_score}%",
        "",
        "Matched Keywords:",
        ", ".join(sorted(matched)) if matched else "None",
        "",
        "Missing Keywords:",
        ", ".join(sorted(missing)) if missing else "None",
        "",
        "Suggestions:",
        "• Emphasize projects, responsibilities, and metrics related to missing keywords.",
        "• Mirror JD terminology where truthful to improve ATS matching.",
        "• Add measurable impact (numbers, % improvements) where possible."
    ]
    return "\n".join(lines).encode("utf-8")


with st.sidebar:
    st.markdown(
        """
        <div class="hero">
            <h1 class="headline">📄 Resume Intelligence</h1>
            <p class="subhead">Predict your resume’s best-fit category and measure alignment with any job description. Clean UX, actionable insights.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.markdown("**How it works**")
    st.caption("1) Upload resume (PDF/DOCX/TXT)\n\n2) Get category prediction\n\n3) Paste JD → see match %, matched/missing skills\n\n4) Download quick report")


st.markdown(
    """
    <div class="hero">
        <h2 class="headline">AI-Powered Resume Category & JD Match</h2>
        <p class="subhead">Fast, accurate, and beautifully presented. Built with Scikit-learn + Streamlit.</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("")

col_uploader, col_meta = st.columns([1.2, 0.8], vertical_alignment="center")

with col_uploader:
    uploader_box = st.container()
    with uploader_box:
        st.markdown("<div class='section-title'>📤 Upload your Resume</div>", unsafe_allow_html=True)
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Drop a PDF, DOCX, or TXT file here", type=["pdf", "docx", "txt"], label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

with col_meta:
    st.markdown("<div class='section-title'>🔎 At a Glance</div>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: metric_card("Supported Formats", "PDF • DOCX • TXT")
    with c2: metric_card("Model", "SVC + TF-IDF")

st.write("")

if uploaded_file is not None:
    try:
        with st.spinner("⏳ Extracting text from your resume..."):
            resume_text = handle_file_upload(uploaded_file)

        st.markdown("<div class='section-title'>✅ Extraction</div>", unsafe_allow_html=True)
        st.success("Text extracted successfully from the uploaded resume.")

        with st.expander("👀 View Extracted Resume Text"):
            st.text_area("Extracted Resume Text", resume_text, height=300)

        # ---------------- Prediction Section ----------------
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>🔮 Predicted Category</div>", unsafe_allow_html=True)
        with st.spinner("Analyzing resume and predicting the best-fit category..."):
            category = pred(resume_text)
        st.markdown(f"<div class='glass'><h3 style='margin:0;color:#e2e8f0;'>🎯 {category}</h3></div>", unsafe_allow_html=True)

        # Celebrate just a little
        st.balloons()

        # ---------------- JD Matching ----------------
        st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>📑 Job Description Matching</div>", unsafe_allow_html=True)
        st.markdown("<div class='glass'>", unsafe_allow_html=True)
        job_desc = st.text_area("Paste the Job Description here", height=180, label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)

        if job_desc.strip():
            match_score = match_resume_with_jd(resume_text, job_desc)

            c_left, c_right = st.columns([0.55, 0.45])
            with c_left:
                st.markdown("<div class='section-title'>Match Strength</div>", unsafe_allow_html=True)
                progress_ring(match_score)

                if match_score > 75:
                    st.success("🔥 Strong Match! Your resume fits this JD well.")
                elif match_score > 50:
                    st.warning("🙂 Moderate Match. Tailor your resume to mirror more Job Discription terminology.")
                else:
                    st.error("⚠️ Low Match. Consider adding relevant skills & evidence aligned to this Job Discription.")

            with c_right:
                # Extract keywords & intersections
                resume_keywords = set(extract_keywords(resume_text))
                jd_keywords = set(extract_keywords(job_desc))

                matched = sorted(resume_keywords.intersection(jd_keywords))
                missing = sorted(jd_keywords - resume_keywords)

                st.markdown("<div class='section-title'>✅ Matched Skills / Keywords</div>", unsafe_allow_html=True)
                if matched:
                    pills(matched, "good")
                else:
                    st.write("No strong matches detected yet.")

                st.write("")
                st.markdown("<div class='section-title'>❌ Missing Important Skills</div>", unsafe_allow_html=True)
                if missing:
                    pills(missing, "bad")
                    st.write("")
                    st.markdown("<div class='section-title'>💡 Suggestions</div>", unsafe_allow_html=True)
                    st.markdown(
                        """
                        - Weave missing keywords into **projects**, **work experience**, and **skills** truthfully.  
                        - Use the JD’s wording (where accurate) to improve **ATS** alignment.  
                        - Add measurable impact (e.g., “improved inference latency by 35%”).  
                        """)
                else:
                    st.success("Great news! Your resume covers the major keywords in this Job Dricprion.")

                # Download quick report
                report_bytes = make_report(category, match_score, matched, missing)
                st.download_button(
                    label="⬇️ Download Quick Report",
                    data=report_bytes,
                    file_name="resume_match_report.txt",
                    mime="text/plain",
                    type="primary",
                )

    except Exception as e:
        st.error(f"⚠️ Error processing the file: {str(e)}")

# Footer
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown(
    """
    <div class="footer">
        Built with ❤️ using Streamlit + Scikit-learn • Design: Glassmorphism + Gradients
    </div>
    """,
    unsafe_allow_html=True
)

# --- Imports ---------------------------------------------------------------
import os
import re
import io
import json
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
import PyPDF2
import docx

# --- Secrets ---------------------------------------------------------------
API_KEY = st.secrets.get("GEMINI_API_KEY", "")
if not API_KEY:
    st.warning("⚠️ 'GEMINI_API_KEY' is not set in st.secrets. Add it before deploying.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# --- Identity / System Instructions ----------------------------------------
def load_identity() -> str:
    try:
        with open("identity.txt") as f:
            return f.read()
    except FileNotFoundError:
        return ("You are ReadysetRole, an ATS resume assistant. "
                "Never fabricate. Use headings: Summary, Skills, Professional Experience, Projects, Education, Certifications.")

SYSTEM_INSTRUCTIONS = load_identity()

# --- Helper Functions ------------------------------------------------------
def parse_resume_file(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT file"""
    try:
        if uploaded_file.type == "application/pdf":
            reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for p in reader.pages:
                try:
                    text += p.extract_text() or ""
                except Exception:
                    pass
            return text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            docf = docx.Document(io.BytesIO(uploaded_file.read()))
            return "\n".join(para.text for para in docf.paragraphs)
        else:
            return uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return ""

def call_gemini(prompt: str, temperature: float = 0.5, json_only: bool = False) -> str:
    """Call Gemini API with basic error handling"""
    try:
        cfg = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTIONS,
            temperature=temperature,
            max_output_tokens=8000,
            response_mime_type="application/json" if json_only else None,
        )
        resp = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[types.Content(parts=[types.Part(text=prompt)])],
            config=cfg,
        )
        return resp.text or ""
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return ""

def extract_json(text: str):
    """Extract first valid JSON object/array from text."""
    try:
        # fenced
        m = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        # anywhere
        m = re.search(r'(\{.*?\}|\[.*?\])', text, re.DOTALL)
        if m:
            return json.loads(m.group(1))
        # raw
        return json.loads(text)
    except Exception:
        return None

# --- ATS Prompts -----------------------------------------------------------
SCORE_PROMPT_TMPL = """Return ONLY a JSON object with the fields below (numbers 0–100).

Analyze the resume against the JD.

RESUME:
{resume}

JOB DESCRIPTION:
{jd}

Return:
{{
  "overall_score": 0,
  "skills_fit": 0,
  "experience_fit": 0,
  "education_fit": 0,
  "ats_keywords_coverage": 0
}}
"""

TAILOR_PROMPT_TMPL = """Tailor the resume to the JD using the ATS rules below. 
DO NOT FABRICATE. Use <METRIC_TBD> where impact is missing.

Rules:
- Single column, ATS-safe, headings: Summary, Skills, Professional Experience, Projects, Education, Certifications (omit if N/A)
- Bullet style: Action → What → How/Tools → Impact
- Use JD keywords naturally ONLY if supported by resume evidence
- 1 page if <=8y experience, else max 2 pages
- End output with [END_RESUME]

ORIGINAL RESUME:
{resume}

JOB DESCRIPTION:
{jd}

Generate the tailored resume:
"""

COVER_LETTER_PROMPT_TMPL = """Write a concise cover letter (180–250 words) grounded ONLY in the tailored resume below.
Use the following details if provided.

Company: {company}
Role: {role}
Receiver: {receiver}
UserNotes (optional): {notes}

TAILORED RESUME:
{tailored}

Return plain paragraphs (no greeting macros). End with [END_COVER].
"""

# --- LaTeX Builders --------------------------------------------------------
LATEX_RESUME_SHELL = r"""
\documentclass[10pt]{article}
\usepackage[margin=0.5in]{geometry}
\usepackage[hidelinks]{hyperref}
\usepackage{enumitem}
\usepackage{titlesec}
\setlength{\parindent}{0pt}
\setlength{\parskip}{0pt}
\pagestyle{empty}

\titlespacing*{\section}{0pt}{4pt}{2pt}
\setlist[itemize]{noitemsep, topsep=2pt, leftmargin=1.2em}

\begin{document}

{\Large \bfseries %(name)s} \\[6pt]
%(location)s — %(phone)s — \href{mailto:%(email)s}{%(email)s} — %
\href{%(portfolio_url)s}{%(portfolio_label)s} — %
\href{%(linkedin_url)s}{%(linkedin_label)s}

\vspace{4pt}\hrule\vspace{4pt}

\section*{Summary}
%(summary)s

\vspace{4pt}\hrule\vspace{4pt}

\section*{Education}
%(education)s

\vspace{4pt}\hrule\vspace{4pt}

\section*{Professional Experience}
%(experience)s

\vspace{4pt}\hrule\vspace{4pt}

\section*{Selected Projects}
%(projects)s

\vspace{4pt}\hrule\vspace{4pt}

\section*{Skills}
%(skills)s

%(certs_block)s

\end{document}
"""

LATEX_CERTS_BLOCK = r"""
\vspace{4pt}\hrule\vspace{4pt}
\section*{Certifications}
%s
"""

def split_sections(tailored_text: str) -> dict:
    """
    Parse Gemini output into sections by headings.
    Expected headings: Summary, Skills, Professional Experience, Projects, Education, Certifications
    """
    # normalize
    t = re.sub(r'\r', '', tailored_text)
    # grab until [END_RESUME]
    t = re.split(r'\[END_RESUME\]', t, flags=re.IGNORECASE)[0]

    def grab(label):
        pat = rf'^\s*{label}\s*\n(.*?)(?=^\s*(Summary|Skills|Professional Experience|Projects|Education|Certifications)\s*$|\Z)'
        m = re.search(pat, t, re.IGNORECASE | re.MULTILINE | re.DOTALL)
        return (m.group(1).strip() if m else "")

    return {
        "summary": grab("Summary"),
        "skills": grab("Skills"),
        "experience": grab("Professional\s+Experience"),
        "projects": grab("Projects"),
        "education": grab("Education"),
        "certifications": grab("Certifications"),
    }

def to_latex_itemize(plain: str) -> str:
    """
    Convert plaintext bullets to LaTeX itemize safely.
    Accepts lines with '-' or '•' or '*' or numbered bullets.
    """
    # split blocks into bullet lines
    lines = [ln.strip() for ln in plain.splitlines() if ln.strip()]
    # heuristics: keep roles with inline bullets: handle `-` etc.
    out = []
    buffer_role = []
    for ln in lines:
        if re.match(r'^[A-Za-z].+?\|', ln) or re.match(r'^\*\*.+\*\*', ln):
            # likely a role header, push buffered
            if buffer_role:
                out.append("\\begin{itemize}")
                for b in buffer_role:
                    out.append(f"  \\item {b}")
                out.append("\\end{itemize}\n")
                buffer_role = []
            out.append(ln)
        elif re.match(r'^(\-|\*|•|\d+\.)\s+', ln):
            buffer_role.append(re.sub(r'^(\-|\*|•|\d+\.)\s+', '', ln))
        else:
            out.append(ln)

    if buffer_role:
        out.append("\\begin{itemize}")
        for b in buffer_role:
            out.append(f"  \\item {b}")
        out.append("\\end{itemize}\n")

    return "\n".join(out) if out else plain

def build_resume_latex(tailored: str,
                       name="Your Name",
                       location="City, ST",
                       phone="(000) 000-0000",
                       email="you@example.com",
                       portfolio_url="https://example.com",
                       portfolio_label="example.com",
                       linkedin_url="https://www.linkedin.com/in/your-handle/",
                       linkedin_label="linkedin.com/in/your-handle") -> str:
    sec = split_sections(tailored)

    certs_block = ""
    if sec.get("certifications"):
        certs_block = LATEX_CERTS_BLOCK % (to_latex_itemize(sec["certifications"]))

    payload = {
        "name": name,
        "location": location,
        "phone": phone,
        "email": email,
        "portfolio_url": portfolio_url,
        "portfolio_label": portfolio_label,
        "linkedin_url": linkedin_url,
        "linkedin_label": linkedin_label,
        "summary": sec["summary"],
        "education": to_latex_itemize(sec["education"]),
        "experience": to_latex_itemize(sec["experience"]),
        "projects": to_latex_itemize(sec["projects"]),
        "skills": to_latex_itemize(sec["skills"]),
        "certs_block": certs_block,
    }
    return LATEX_RESUME_SHELL % payload

LATEX_COVER_LETTER_SHELL = r"""
\input{setup/preamble.tex}
\input{setup/macros.tex}

\begin{document}
\name{%s}{%s}
\receiver{%s}

\para{Dear %s,}

\para{%s}

\para{Thank you for considering my application. I would welcome the opportunity to contribute and to discuss how my background aligns with your needs.}

\bottom{%s}{%s}{%s}

\end{document}
"""

# --- Session State ---------------------------------------------------------
if 'master_resume' not in st.session_state:
    st.session_state.master_resume = None
if 'master_resume_name' not in st.session_state:
    st.session_state.master_resume_name = None
if 'current_jd' not in st.session_state:
    st.session_state.current_jd = None
if 'tailored_resume' not in st.session_state:
    st.session_state.tailored_resume = None
if 'scores' not in st.session_state:
    st.session_state.scores = {}

# --- Page Config & Minimal Styling -----------------------------------------
st.set_page_config(page_title="ReadysetRole - Resume Optimizer", page_icon="⚡", layout="wide")

st.markdown("<h1 style='text-align:center;'>⚡ ReadysetRole</h1>", unsafe_allow_html=True)
st.caption("AutoTailor your resume • Evidence-only • Zero fabrication")

# --- Uploads ---------------------------------------------------------------
st.subheader("📤 Upload Your Documents")

c1, c2 = st.columns(2)

with c1:
    up_res = st.file_uploader("Master Resume", type=["pdf","docx","txt"], key="resume_uploader")
    if up_res and st.session_state.master_resume is None:
        txt = parse_resume_file(up_res)
        if txt.strip():
            st.session_state.master_resume = txt
            st.session_state.master_resume_name = up_res.name
            st.success("✅ Resume loaded")

    if st.session_state.master_resume_name:
        st.info(f"📄 {st.session_state.master_resume_name}")

with c2:
    jd_mode = st.radio("Job Description", ["Paste Text", "Upload File"], horizontal=True)
    if jd_mode == "Paste Text":
        jd_txt = st.text_area("Paste JD", height=160, label_visibility="collapsed", key="jd_textarea")
        if st.button("Compute % Match (QuickScore)", use_container_width=True):
            if jd_txt and st.session_state.master_resume:
                st.session_state.current_jd = jd_txt
            else:
                st.warning("Please upload both resume and JD first.")
    else:
        up_jd = st.file_uploader("Upload JD", type=["pdf","docx","txt"], key="jd_uploader")
        if up_jd and st.button("Compute % Match (QuickScore)", use_container_width=True):
            jd_txt = parse_resume_file(up_jd)
            if jd_txt and st.session_state.master_resume:
                st.session_state.current_jd = jd_txt
            else:
                st.warning("Please upload both resume and JD first.")

st.divider()

# --- QuickScore ------------------------------------------------------------
if st.session_state.master_resume and st.session_state.current_jd and not st.session_state.tailored_resume:
    with st.spinner("Scoring..."):
        score_prompt = SCORE_PROMPT_TMPL.format(
            resume=st.session_state.master_resume[:6000],
            jd=st.session_state.current_jd[:6000]
        )
        raw = call_gemini(score_prompt, temperature=0.2, json_only=False)
        obj = extract_json(raw) or {}
        st.session_state.scores = obj

    s = st.session_state.scores or {}
    pre = int(s.get("overall_score", 0))
    colA, colB, colC, colD, colE = st.columns(5)
    colA.metric("Overall Match", f"{pre}%")
    colB.metric("Skills Fit", f"{int(s.get('skills_fit',0))}%")
    colC.metric("Experience Fit", f"{int(s.get('experience_fit',0))}%")
    colD.metric("Education Fit", f"{int(s.get('education_fit',0))}%")
    colE.metric("ATS Keywords", f"{int(s.get('ats_keywords_coverage',0))}%")

    st.success("Next: click **Generate Tailored Resume** to create ATS-safe output and LaTeX.")
    if st.button("🎯 Generate Tailored Resume", type="primary", use_container_width=True):
        with st.spinner("Tailoring..."):
            tailor_prompt = TAILOR_PROMPT_TMPL.format(
                resume=st.session_state.master_resume,
                jd=st.session_state.current_jd
            )
            tailored = call_gemini(tailor_prompt, temperature=0.6)
            st.session_state.tailored_resume = tailored

# --- Tailored Resume + LaTeX -----------------------------------------------
if st.session_state.tailored_resume:
    st.subheader("📄 Tailored Resume (Plain Text)")
    st.code(st.session_state.tailored_resume, language="markdown")

    with st.expander("🧩 Overleaf-ready LaTeX (Resume)"):
        # Minimal header inputs so user can control the LaTeX header
        name = st.text_input("Name", value="Your Name")
        location = st.text_input("Location", value="City, ST")
        phone = st.text_input("Phone", value="(000) 000-0000")
        email = st.text_input("Email", value="you@example.com")
        portfolio_url = st.text_input("Portfolio URL", value="https://example.com")
        portfolio_label = st.text_input("Portfolio Label", value="example.com")
        linkedin_url = st.text_input("LinkedIn URL", value="https://www.linkedin.com/in/your-handle/")
        linkedin_label = st.text_input("LinkedIn Label", value="linkedin.com/in/your-handle")

        if st.button("Build LaTeX Resume", use_container_width=True):
            latex_resume = build_resume_latex(
                st.session_state.tailored_resume,
                name, location, phone, email,
                portfolio_url, portfolio_label,
                linkedin_url, linkedin_label
            )
            st.code(latex_resume, language="latex")

    st.divider()

    # --- Optional Cover Letter --------------------------------------------
    st.subheader("✉️ Optional: Generate Cover Letter (LaTeX)")
    with st.form("cover_form"):
        gen_cover = st.checkbox("Yes, generate a cover letter")
        col1, col2 = st.columns(2)
        with col1:
            company = st.text_input("Company", value="")
            role = st.text_input("Role / Position", value="")
            receiver = st.text_input("Receiver (e.g., Hiring Manager \\\\ Company)", value="Hiring Manager")
            greeting_name = st.text_input("Greeting (Dear ___,)", value="Hiring Manager")
        with col2:
            sender_title = st.text_input("Sender Title (under \\name{})", value="Applicant")
            sender_city = st.text_input("Sender City", value="City, ST")
            sender_phone = st.text_input("Sender Phone", value="(000) 000-0000")
            sender_email = st.text_input("Sender Email", value="you@example.com")
        notes = st.text_area("Optional notes to emphasize (kept factual)", value="", height=100)
        submitted = st.form_submit_button("Generate Cover Letter (LaTeX)")
    if submitted and gen_cover:
        with st.spinner("Drafting cover letter..."):
            cl_prompt = COVER_LETTER_PROMPT_TMPL.format(
                company=company, role=role, receiver=receiver,
                notes=notes, tailored=st.session_state.tailored_resume
            )
            body = call_gemini(cl_prompt, temperature=0.6)
            body = re.split(r'\[END_COVER\]', body)[0].strip()

            latex_letter = LATEX_COVER_LETTER_SHELL % (
                name, sender_title,
                (receiver if receiver else "Hiring Manager"),
                (greeting_name if greeting_name else "Hiring Manager"),
                body,
                sender_city, sender_phone, sender_email
            )
            st.code(latex_letter, language="latex")

st.caption("Built with Streamlit + Gemini 2.0 Flash | ReadysetRole v2 (simple flow)")

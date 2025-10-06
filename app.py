# --- Imports ---------------------------------------------------------------
import os
import json
import re
import streamlit as st
from PIL import Image
from google import genai
from google.genai import types
import PyPDF2
import docx
import io

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
        st.warning("⚠️ 'identity.txt' not found. Using default prompt.")
        return "You are ReadysetRole, a resume optimization assistant."

system_instructions = load_identity()

# --- Helper Functions ------------------------------------------------------
def parse_resume_file(uploaded_file) -> str:
    """Extract text from PDF, DOCX, or TXT file"""
    try:
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(io.BytesIO(uploaded_file.read()))
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            return "\n".join([para.text for para in doc.paragraphs])
        elif uploaded_file.type == "text/plain":
            return uploaded_file.read().decode("utf-8")
        else:
            return uploaded_file.read().decode("utf-8")
    except Exception as e:
        st.error(f"Error parsing file: {e}")
        return ""

def call_gemini(prompt: str, temperature: float = 1.0) -> str:
    """Call Gemini API with error handling"""
    try:
        generation_cfg = types.GenerateContentConfig(
            system_instruction=system_instructions,
            temperature=temperature,
            max_output_tokens=8000,
        )
        
        resp = client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=[types.Content(parts=[types.Part(text=prompt)])],
            config=generation_cfg,
        )
        return resp.text or ""
    except Exception as e:
        st.error(f"Gemini API error: {e}")
        return ""

def extract_json_from_response(text: str):
    """Extract JSON from markdown code blocks or raw text - improved"""
    try:
        # Try to find JSON in code blocks first
        json_match = re.search(r'```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Try to find JSON object/array anywhere in text
        json_match = re.search(r'(\{.*?\}|\[.*?\])', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        
        # Try parsing entire text as JSON
        return json.loads(text)
    except Exception as e:
        # Return empty dict/list if parsing fails
        return {} if '{' in text else []

# --- Session State Init ----------------------------------------------------
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
if 'evidence_map' not in st.session_state:
    st.session_state.evidence_map = []
if 'change_log' not in st.session_state:
    st.session_state.change_log = []
if 'refinement_packs' not in st.session_state:
    st.session_state.refinement_packs = []
if 'current_tool' not in st.session_state:
    st.session_state.current_tool = None
if 'selected_packs' not in st.session_state:
    st.session_state.selected_packs = set()

# --- Page Config -----------------------------------------------------------
st.set_page_config(
    page_title="ReadysetRole - Resume Optimizer",
    page_icon="⚡",
    layout="wide"
)

# --- Custom CSS ------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        font-family: 'Inter', sans-serif;
    }
    
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    .stApp, .stApp * {
        color: #e0e0e0 !important;
    }
    
    .stApp > div > div {
        background: transparent !important;
    }
    
    .stFileUploader {
        background: rgba(30, 30, 46, 0.8) !important;
        border: 2px dashed #7e22ce !important;
        border-radius: 16px !important;
        padding: 2rem !important;
    }
    
    .stFileUploader label {
        color: #a78bfa !important;
        font-weight: 700 !important;
        font-size: 1.2rem !important;
    }
    
    .stTextArea textarea {
        background: #1e1e2e !important;
        color: #e0e0e0 !important;
        border: 2px solid #7e22ce !important;
        border-radius: 12px !important;
    }
    
    .stTextArea label {
        color: #a78bfa !important;
        font-weight: 600 !important;
    }
    
    .stButton button {
        background: linear-gradient(135deg, #7e22ce 0%, #6d28d9 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(126, 34, 206, 0.4) !important;
    }
    
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        font-size: 1.1rem !important;
        padding: 1rem 2rem !important;
    }
    
    .stRadio {
        background: rgba(30, 30, 46, 0.6) !important;
        padding: 1rem !important;
        border-radius: 12px !important;
    }
    
    .score-card {
        background: linear-gradient(135deg, #1e3c72 0%, #7e22ce 100%);
        color: white !important;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }
    
    .score-number {
        font-size: 4rem !important;
        font-weight: 800 !important;
        color: white !important;
    }
    
    .score-label {
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: white !important;
    }
    
    .delta-positive {
        color: #10b981 !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    .stMetric {
        background: rgba(126, 34, 206, 0.2) !important;
        padding: 1rem !important;
        border-radius: 12px !important;
    }
    
    .stMetric label {
        color: #a78bfa !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.8rem !important;
    }
    
    .evidence-row {
        background: rgba(126, 34, 206, 0.15);
        border-left: 4px solid #7e22ce;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }
    
    .jd-anchor {
        font-weight: 600 !important;
        color: #a78bfa !important;
    }
    
    .resume-match {
        color: #d1d5db !important;
        margin-top: 0.5rem;
    }
    
    .badge-verified {
        background: #10b981 !important;
        color: white !important;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-left: 0.5rem;
    }
    
    .badge-pending {
        background: #f59e0b !important;
        color: white !important;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        margin-left: 0.5rem;
    }
    
    .change-item {
        background: rgba(30, 60, 114, 0.2);
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #3b82f6;
    }
    
    .change-before {
        color: #ef4444 !important;
        text-decoration: line-through;
    }
    
    .change-after {
        color: #10b981 !important;
        font-weight: 600 !important;
    }
    
    .change-why {
        color: #9ca3af !important;
        font-style: italic;
    }
    
    .resume-output {
        background: #0f0f1e !important;
        color: #e5e7eb !important;
        padding: 1.5rem;
        border-radius: 16px;
        font-family: 'Courier New', monospace !important;
        font-size: 0.85rem;
        line-height: 1.7;
        max-height: 600px;
        overflow-y: auto;
        white-space: pre-wrap;
        border: 2px solid rgba(126, 34, 206, 0.3);
    }
    
    .stCheckbox label {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    
    .stExpander {
        background: rgba(30, 30, 46, 0.6) !important;
        border: 1px solid rgba(126, 34, 206, 0.3) !important;
        border-radius: 12px !important;
    }
    
    .stDownloadButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    }
    
    .stAlert, .stInfo {
        background: rgba(126, 34, 206, 0.2) !important;
        border: 2px solid #7e22ce !important;
        color: #e0e0e0 !important;
    }
    
    .stSuccess {
        background: rgba(16, 185, 129, 0.2) !important;
        border: 2px solid #10b981 !important;
    }
    
    .no-fabrication {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700 !important;
        margin: 1.5rem 0;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ----------------------------------------------------------------
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3rem; font-weight: 800; margin-bottom: 0.5rem;">
        ⚡ ReadysetRole
    </h1>
    <p style="font-size: 1.1rem; color: #9ca3af;">AutoTailor your resume instantly • Evidence-only • Zero fabrication</p>
</div>
""", unsafe_allow_html=True)

# --- Top: Document Uploads (Full Width) ------------------------------------
st.markdown("### 📤 Upload Your Documents")

col_resume_upload, col_jd_upload = st.columns(2)

with col_resume_upload:
    uploaded_resume = st.file_uploader(
        "Master Resume",
        type=["pdf", "docx", "txt"],
        help="Upload your comprehensive resume",
        key="resume_uploader"
    )
    
    if uploaded_resume and st.session_state.master_resume is None:
        with st.spinner("Processing resume..."):
            resume_text = parse_resume_file(uploaded_resume)
            if resume_text:
                st.session_state.master_resume = resume_text
                st.session_state.master_resume_name = uploaded_resume.name
                st.success(f"✅ Loaded!")
    
    if st.session_state.master_resume:
        st.info(f"📄 {st.session_state.master_resume_name}")

with col_jd_upload:
    jd_input_method = st.radio("Job Description", ["Paste Text", "Upload File"], horizontal=True)
    
    if jd_input_method == "Paste Text":
        jd_text = st.text_area(
            "Paste JD",
            height=120,
            placeholder="Paste job description...",
            key="jd_textarea",
            label_visibility="collapsed"
        )
        if st.button("🎯 AutoTailor Resume", type="primary", use_container_width=True):
            if jd_text and st.session_state.master_resume:
                st.session_state.current_jd = jd_text
                st.session_state.tailored_resume = None
                st.rerun()
            else:
                st.warning("Please upload both resume and JD first")
    else:
        uploaded_jd = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"], key="jd_uploader")
        if uploaded_jd and st.button("🎯 AutoTailor", type="primary", use_container_width=True):
            jd_text = parse_resume_file(uploaded_jd)
            if jd_text and st.session_state.master_resume:
                st.session_state.current_jd = jd_text
                st.session_state.tailored_resume = None
                st.rerun()

st.markdown("---")

# --- Main Editor: Left (Editor) + Right (Tracker) --------------------------
col_editor, col_tracker = st.columns([2.5, 1], gap="large")

with col_editor:
    if st.session_state.current_jd and st.session_state.master_resume and not st.session_state.tailored_resume:
        with st.spinner("🔄 AutoTailoring your resume..."):
            # Step 1: Pre-Score
            score_prompt = f"""You must return ONLY a valid JSON object with scores. No other text.

Analyze this resume against the job description and return scores:

RESUME:
{st.session_state.master_resume[:3000]}

JOB DESCRIPTION:
{st.session_state.current_jd[:3000]}

Return this JSON structure with numbers 0-100:
{{
  "overall_score": 75,
  "skills_fit": 80,
  "experience_fit": 70,
  "education_fit": 85,
  "ats_keywords_coverage": 65
}}"""
            
            pre_response = call_gemini(score_prompt, 0.2)
            pre_scores = extract_json_from_response(pre_response)
            
            # Step 2: Tailor Resume
            tailor_prompt = f"""Tailor this resume to match the job description. Follow these rules:

RULES:
- ATS-safe format: single column, plain text, standard headings
- Use keywords from JD ONLY where resume provides evidence
- Bullet format: Action → What → How/Tools → Impact
- Insert <METRIC_TBD> where metrics are missing
- Never fabricate information
- End with [END_RESUME]

ORIGINAL RESUME:
{st.session_state.master_resume}

JOB DESCRIPTION:
{st.session_state.current_jd}

Generate the tailored resume:"""
            
            st.session_state.tailored_resume = call_gemini(tailor_prompt, 0.7)
            
            # Step 3: Post-Score
            post_response = call_gemini(score_prompt.replace(st.session_state.master_resume[:3000], st.session_state.tailored_resume[:3000]), 0.2)
            post_scores = extract_json_from_response(post_response)
            
            st.session_state.scores = {'pre': pre_scores, 'post': post_scores}
            
            # Step 4: Evidence Map
            evidence_prompt = f"""Return ONLY a JSON array mapping JD requirements to resume evidence.

JOB DESCRIPTION:
{st.session_state.current_jd[:2000]}

TAILORED RESUME:
{st.session_state.tailored_resume[:2000]}

Return this format:
[
  {{"jd_anchor": "5+ years Python", "resume_match": "6 years Python at TechCorp", "verified": true}},
  {{"jd_anchor": "ML deployment", "resume_match": "Deployed 15 models using TensorFlow", "verified": true}}
]"""
            
            evidence_response = call_gemini(evidence_prompt, 0.3)
            evidence_data = extract_json_from_response(evidence_response)
            st.session_state.evidence_map = evidence_data if isinstance(evidence_data, list) else []
            
            # Step 5: Change Log
            changelog_prompt = f"""Return ONLY a JSON array showing key changes.

ORIGINAL:
{st.session_state.master_resume[:2000]}

TAILORED:
{st.session_state.tailored_resume[:2000]}

Return this format:
[
  {{"before": "Built pipelines", "after": "Architected real-time pipelines using Kafka", "why": "JD emphasizes real-time processing"}}
]"""
            
            changelog_response = call_gemini(changelog_prompt, 0.3)
            changelog_data = extract_json_from_response(changelog_response)
            st.session_state.change_log = changelog_data if isinstance(changelog_data, list) else []
            
            # Step 6: Refinement Packs
            packs_prompt = f"""Return ONLY a JSON array with 3 keyword packs.

JOB DESCRIPTION:
{st.session_state.current_jd[:2000]}

RESUME:
{st.session_state.tailored_resume[:2000]}

Return this format:
[
  {{"title": "Pack 1: Tech Stack", "lift": "+5%", "tokens": ["Python", "AWS", "Docker"], "jd_evidence": "requires Python and cloud", "resume_evidence": "has Python, AWS experience"}}
]"""
            
            packs_response = call_gemini(packs_prompt, 0.5)
            packs_data = extract_json_from_response(packs_response)
            st.session_state.refinement_packs = packs_data if isinstance(packs_data, list) else []
            
            st.rerun()
    
    if st.session_state.tailored_resume:
        st.markdown("### 📄 Tailored Resume")
        st.markdown(f'<div class="resume-output">{st.session_state.tailored_resume}</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.download_button("💾 Download", st.session_state.tailored_resume, "resume.txt", use_container_width=True)
        with col2:
            if st.button("📋 Show Code", use_container_width=True):
                st.code(st.session_state.tailored_resume)
        
        if st.session_state.change_log:
            with st.expander("📝 Change Log"):
                for item in st.session_state.change_log:
                    st.markdown(f'<div class="change-item"><div class="change-before">{item.get("before","")}</div><div class="change-after">{item.get("after","")}</div><div class="change-why">{item.get("why","")}</div></div>', unsafe_allow_html=True)
        
        if st.session_state.evidence_map:
            with st.expander("📍 Evidence Map"):
                for item in st.session_state.evidence_map:
                    badge = "verified" if item.get('verified') else "pending"
                    st.markdown(f'<div class="evidence-row"><div class="jd-anchor">{item.get("jd_anchor","")}</div><div class="resume-match">→ {item.get("resume_match","")} <span class="badge-{badge}">{"✅" if badge=="verified" else "◻︎"}</span></div></div>', unsafe_allow_html=True)
        
        if st.session_state.current_tool == "1" and st.session_state.refinement_packs:
            st.markdown("### 🎯 Refinement Packs")
            for idx, pack in enumerate(st.session_state.refinement_packs):
                checked = st.checkbox(f"{pack.get('title','')} {pack.get('lift','')}", key=f"p{idx}", value=idx in st.session_state.selected_packs)
                if checked:
                    st.session_state.selected_packs.add(idx)
                    st.caption("Tokens: " + ", ".join(pack.get('tokens',[])))
                else:
                    st.session_state.selected_packs.discard(idx)
            
            if st.button("Apply Packs", type="primary", use_container_width=True) and st.session_state.selected_packs:
                with st.spinner("Applying packs..."):
                    packs = [st.session_state.refinement_packs[i] for i in st.session_state.selected_packs]
                    apply_prompt = f"""Apply these keyword packs to the resume. Only add where evidence exists.

PACKS TO APPLY:
{json.dumps(packs, indent=2)}

CURRENT RESUME:
{st.session_state.tailored_resume}

JOB DESCRIPTION:
{st.session_state.current_jd}

Return updated resume ending with [END_RESUME]:"""
                    
                    st.session_state.tailored_resume = call_gemini(apply_prompt, 0.7)
                    st.session_state.selected_packs = set()
                    st.session_state.current_tool = None
                    st.rerun()
        
        st.markdown('<div class="no-fabrication">🔒 No fabrication: Resume + JD only</div>', unsafe_allow_html=True)

with col_tracker:
    st.markdown("### 📊 Score Tracker")
    
    if st.session_state.scores:
        pre = st.session_state.scores.get('pre', {})
        post = st.session_state.scores.get('post', {})
        
        pre_overall = int(pre.get('overall_score', 0))
        post_overall = int(post.get('overall_score', 0))
        delta = post_overall - pre_overall
        
        st.markdown(f'<div class="score-card"><div style="display:flex;justify-content:space-around;align-items:center"><div><div class="score-number">{pre_overall}</div><div class="score-label">PRE</div></div><div class="delta-positive">+{delta}</div><div><div class="score-number">{post_overall}</div><div class="score-label">POST</div></div></div></div>', unsafe_allow_html=True)
        
        st.markdown("#### Subscores")
        for key, lbl in [('skills_fit','Skills'),('experience_fit','Experience'),('ats_keywords_coverage','ATS'),('education_fit','Education')]:
            val = int(post.get(key, 0))
            st.metric(lbl, f"{val}%")
    
    if st.session_state.tailored_resume:
        st.markdown("---")
        st.markdown("### 🎯 Actions")
        actions = [("1","Refine","🎯"),("2","Boosters","✨"),("3","Coverage","📊"),("4","Presets","🎭"),("5","A/B","🔀"),("6","Level","⚖️"),("0","Next JD","🔄")]
        
        for num, lbl, ico in actions:
            if st.button(f"{ico} {lbl}", key=f"o{num}", use_container_width=True):
                if num == "0":
                    st.session_state.current_jd = None
                    st.session_state.tailored_resume = None
                    st.session_state.scores = {}
                    st.session_state.evidence_map = []
                    st.session_state.change_log = []
                    st.session_state.refinement_packs = []
                    st.session_state.current_tool = None
                    st.session_state.selected_packs = set()
                    st.rerun()
                else:
                    st.session_state.current_tool = num
                    st.rerun()

st.markdown("---")
st.caption("Built with Streamlit + Gemini 2.0 Flash | ReadysetRole v1.0")

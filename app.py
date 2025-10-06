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

def extract_json_from_response(text: str) -> dict:
    """Extract JSON from markdown code blocks or raw text"""
    try:
        # Try to find JSON in code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
        # Try raw JSON
        return json.loads(text)
    except:
        return {}

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

# --- Custom CSS ------------------------------------------------------------
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    /* Global styles */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Score cards */
    .score-card {
        background: linear-gradient(135deg, #1e3c72 0%, #7e22ce 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.3);
    }
    
    .score-number {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1;
    }
    
    .score-label {
        font-size: 0.9rem;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        margin-top: 0.5rem;
    }
    
    .delta-positive {
        color: #10b981;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Evidence row */
    .evidence-row {
        background: linear-gradient(135deg, #faf5ff 0%, white 100%);
        border-left: 4px solid #7e22ce;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }
    
    .jd-anchor {
        font-weight: 600;
        color: #1e3c72;
        font-size: 0.9rem;
    }
    
    .resume-match {
        color: #555;
        font-size: 0.88rem;
        margin-top: 0.5rem;
    }
    
    .badge-verified {
        background: #d1fae5;
        color: #065f46;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .badge-pending {
        background: #fef3c7;
        color: #92400e;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    /* Change log */
    .change-item {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
        border-left: 4px solid #0ea5e9;
    }
    
    .change-before {
        color: #ef4444;
        text-decoration: line-through;
        font-size: 0.88rem;
    }
    
    .change-after {
        color: #10b981;
        font-weight: 600;
        font-size: 0.88rem;
        margin-top: 0.3rem;
    }
    
    .change-why {
        color: #666;
        font-size: 0.8rem;
        font-style: italic;
        margin-top: 0.3rem;
    }
    
    /* Pack card */
    .pack-card {
        background: white;
        border: 2px solid #e5e7eb;
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
    }
    
    .pack-card.selected {
        border-color: #7e22ce;
        background: linear-gradient(135deg, #f3e8ff 0%, #faf5ff 100%);
    }
    
    .pack-title {
        font-weight: 700;
        color: #1e3c72;
        font-size: 1.1rem;
    }
    
    .lift-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
    }
    
    .token {
        background: #e0f2fe;
        color: #0c4a6e;
        padding: 0.3rem 0.6rem;
        border-radius: 8px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #0ea5e9;
        display: inline-block;
        margin: 0.2rem;
    }
    
    /* Resume output */
    .resume-output {
        background: #1a1a2e;
        color: #eee;
        padding: 1.5rem;
        border-radius: 16px;
        font-family: 'Courier New', monospace;
        font-size: 0.85rem;
        line-height: 1.7;
        max-height: 600px;
        overflow-y: auto;
        white-space: pre-wrap;
    }
    
    .end-marker {
        color: #10b981;
        font-weight: 700;
        text-align: center;
        margin-top: 1rem;
    }
    
    /* No fabrication badge */
    .no-fabrication {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700;
        margin: 1.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# --- Page Config -----------------------------------------------------------
st.set_page_config(
    page_title="ReadysetRole - Resume Optimizer",
    page_icon="⚡",
    layout="wide"
)

# --- Header ----------------------------------------------------------------
st.markdown("""
<div style="text-align: center; padding: 2rem 0;">
    <h1 style="font-size: 3rem; font-weight: 800; background: linear-gradient(135deg, #1e3c72 0%, #7e22ce 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0.5rem;">
        ⚡ ReadysetRole
    </h1>
    <p style="font-size: 1.1rem; color: #666;">AutoTailor your resume instantly • Evidence-only • Zero fabrication</p>
</div>
""", unsafe_allow_html=True)

# --- Main Layout -----------------------------------------------------------
col_main, col_sidebar = st.columns([2, 1])

with col_main:
    # Upload Section
    st.markdown("### 📤 Upload Your Documents")
    
    col_resume, col_jd = st.columns(2)
    
    with col_resume:
        uploaded_resume = st.file_uploader(
            "Master Resume",
            type=["pdf", "docx", "txt"],
            help="Upload your comprehensive resume (PDF, DOCX, or TXT)"
        )
        
        if uploaded_resume and st.session_state.master_resume is None:
            with st.spinner("Processing resume..."):
                resume_text = parse_resume_file(uploaded_resume)
                if resume_text:
                    st.session_state.master_resume = resume_text
                    st.session_state.master_resume_name = uploaded_resume.name
                    st.success(f"✅ {uploaded_resume.name} loaded!")
        
        if st.session_state.master_resume:
            st.info(f"📄 **Loaded:** {st.session_state.master_resume_name}")
    
    with col_jd:
        jd_input_method = st.radio("Job Description Input", ["Paste Text", "Upload File"], horizontal=True)
        
        if jd_input_method == "Paste Text":
            jd_text = st.text_area(
                "Paste Job Description",
                height=150,
                placeholder="Paste the complete job description here..."
            )
            if st.button("🎯 AutoTailor Resume", type="primary", use_container_width=True):
                if jd_text and st.session_state.master_resume:
                    st.session_state.current_jd = jd_text
                    st.session_state.current_tool = None
                    st.rerun()
        else:
            uploaded_jd = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"])
            if uploaded_jd:
                jd_text = parse_resume_file(uploaded_jd)
                if jd_text and st.button("🎯 AutoTailor Resume", type="primary", use_container_width=True):
                    st.session_state.current_jd = jd_text
                    st.session_state.current_tool = None
                    st.rerun()
    
    # AutoTailor Process
    if st.session_state.current_jd and st.session_state.master_resume and not st.session_state.tailored_resume:
        with st.spinner("🔄 AutoTailoring your resume..."):
            # Step 1: Compute Pre-Score
            score_prompt = f"""Analyze the fit between this resume and job description.

RESUME:
{st.session_state.master_resume}

JOB DESCRIPTION:
{st.session_state.current_jd}

Return ONLY valid JSON with this exact structure:
{{
  "overall_score": <number 0-100>,
  "skills_fit": <number 0-100>,
  "experience_fit": <number 0-100>,
  "education_fit": <number 0-100>,
  "ats_keywords_coverage": <number 0-100>
}}"""
            
            score_response = call_gemini(score_prompt, temperature=0.3)
            pre_scores = extract_json_from_response(score_response)
            
            # Step 2: AutoTailor Resume
            tailor_prompt = f"""Tailor this resume to the job description following these rules:

RULES:
- ATS-safe: single column, plain text, standard headings (SUMMARY, SKILLS, EXPERIENCE, PROJECTS, EDUCATION, CERTIFICATIONS)
- Rewrite bullets using JD keywords ONLY if evidenced in original resume
- Bullet format: Action → What → How/Tools → Impact
- Insert <METRIC_TBD> where impact metrics are missing
- Never fabricate titles, employers, tools, dates, or certifications
- End output with [END_RESUME]

ORIGINAL RESUME:
{st.session_state.master_resume}

JOB DESCRIPTION:
{st.session_state.current_jd}

Generate the tailored resume now:"""
            
            tailored_response = call_gemini(tailor_prompt, temperature=0.7)
            st.session_state.tailored_resume = tailored_response
            
            # Step 3: Compute Post-Score
            post_score_response = call_gemini(score_prompt.replace(st.session_state.master_resume, tailored_response), temperature=0.3)
            post_scores = extract_json_from_response(post_score_response)
            
            st.session_state.scores = {
                'pre': pre_scores,
                'post': post_scores
            }
            
            # Step 4: Generate Evidence Map
            evidence_prompt = f"""Map JD requirements to resume evidence.

JOB DESCRIPTION:
{st.session_state.current_jd}

TAILORED RESUME:
{tailored_response}

Return ONLY valid JSON array:
[
  {{
    "jd_anchor": "brief JD requirement",
    "resume_match": "matching resume evidence",
    "verified": true or false
  }}
]

Generate 4-6 key mappings."""
            
            evidence_response = call_gemini(evidence_prompt, temperature=0.3)
            evidence_data = extract_json_from_response(evidence_response)
            if isinstance(evidence_data, list):
                st.session_state.evidence_map = evidence_data
            
            # Step 5: Generate Change Log
            changelog_prompt = f"""Show key changes from original to tailored resume.

ORIGINAL RESUME:
{st.session_state.master_resume}

TAILORED RESUME:
{tailored_response}

JOB DESCRIPTION:
{st.session_state.current_jd}

Return ONLY valid JSON array:
[
  {{
    "before": "original bullet",
    "after": "tailored bullet",
    "why": "reason tied to JD"
  }}
]

Generate 3-4 key changes."""
            
            changelog_response = call_gemini(changelog_prompt, temperature=0.3)
            changelog_data = extract_json_from_response(changelog_response)
            if isinstance(changelog_data, list):
                st.session_state.change_log = changelog_data
            
            # Step 6: Generate Refinement Packs
            packs_prompt = f"""Suggest 3 targeted keyword refinement packs.

JOB DESCRIPTION:
{st.session_state.current_jd}

TAILORED RESUME:
{tailored_response}

Return ONLY valid JSON array:
[
  {{
    "title": "Pack name",
    "lift": "+X%",
    "tokens": ["keyword1", "keyword2", "keyword3"],
    "jd_evidence": "relevant JD quote",
    "resume_evidence": "supporting resume content"
  }}
]"""
            
            packs_response = call_gemini(packs_prompt, temperature=0.5)
            packs_data = extract_json_from_response(packs_response)
            if isinstance(packs_data, list):
                st.session_state.refinement_packs = packs_data
            
            st.rerun()
    
    # Display Tailored Resume
    if st.session_state.tailored_resume:
        st.markdown("### 📄 Tailored Resume")
        st.markdown(f'<div class="resume-output">{st.session_state.tailored_resume}</div>', unsafe_allow_html=True)
        
        col_copy, col_export = st.columns(2)
        with col_copy:
            if st.button("📋 Copy to Clipboard", use_container_width=True):
                st.code(st.session_state.tailored_resume, language=None)
        with col_export:
            st.download_button(
                "💾 Download .txt",
                data=st.session_state.tailored_resume,
                file_name="tailored_resume.txt",
                mime="text/plain",
                use_container_width=True
            )
        
        # Evidence Map
        if st.session_state.evidence_map:
            st.markdown("### 📍 Evidence Map")
            for item in st.session_state.evidence_map:
                badge = "verified" if item.get('verified', False) else "pending"
                badge_text = "✅ VERIFIED" if badge == "verified" else "◻︎ METRIC_TBD"
                st.markdown(f"""
                <div class="evidence-row">
                    <div class="jd-anchor">{item.get('jd_anchor', '')}</div>
                    <div class="resume-match">
                        → {item.get('resume_match', '')}
                        <span class="badge-{badge}">{badge_text}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        # Change Log
        if st.session_state.change_log:
            st.markdown("### 📝 Change Log")
            for item in st.session_state.change_log:
                st.markdown(f"""
                <div class="change-item">
                    <div class="change-before">{item.get('before', '')}</div>
                    <div class="change-after">{item.get('after', '')}</div>
                    <div class="change-why">{item.get('why', '')}</div>
                </div>
                """, unsafe_allow_html=True)
        
        # No Fabrication Oath
        st.markdown('<div class="no-fabrication">🔒 No-Fabrication Oath: We only used your resume + JD. Nothing else.</div>', unsafe_allow_html=True)

with col_sidebar:
    # Score Dashboard
    if st.session_state.scores:
        pre = st.session_state.scores.get('pre', {})
        post = st.session_state.scores.get('post', {})
        
        pre_overall = pre.get('overall_score', 0)
        post_overall = post.get('overall_score', 0)
        delta = post_overall - pre_overall
        
        st.markdown(f"""
        <div class="score-card">
            <div style="display: flex; justify-content: space-around; align-items: center;">
                <div>
                    <div class="score-number">{int(pre_overall)}</div>
                    <div class="score-label">Pre-Score</div>
                </div>
                <div class="delta-positive">+{int(delta)}</div>
                <div>
                    <div class="score-number">{int(post_overall)}</div>
                    <div class="score-label">Post-Score</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        # Subscores
        st.markdown("#### Subscores")
        for key, label in [
            ('skills_fit', 'Skills Fit'),
            ('experience_fit', 'Experience Fit'),
            ('ats_keywords_coverage', 'ATS Keywords'),
            ('education_fit', 'Education Match')
        ]:
            value = post.get(key, 0)
            st.metric(label, f"{int(value)}%")
    
    # Next Options
    if st.session_state.tailored_resume:
        st.markdown("### 🎯 Next Options")
        
        options = [
            ("1", "Refine Packs"),
            ("2", "Minor Boosters"),
            ("3", "Coverage Board"),
            ("4", "Narrative Presets"),
            ("5", "A/B Bullet"),
            ("6", "Level Calibrator"),
            ("7", "Export .txt"),
            ("0", "Next JD")
        ]
        
        for num, label in options:
            if st.button(f"[{num}] {label}", key=f"opt_{num}", use_container_width=True):
                if num == "0":
                    # Reset for next JD
                    st.session_state.current_jd = None
                    st.session_state.tailored_resume = None
                    st.session_state.scores = {}
                    st.session_state.evidence_map = []
                    st.session_state.change_log = []
                    st.session_state.refinement_packs = []
                    st.session_state.current_tool = None
                    st.rerun()
                elif num == "7":
                    # Already handled above
                    pass
                else:
                    st.session_state.current_tool = num
                    st.rerun()
        
        # Refinement Packs
        if st.session_state.refinement_packs and st.session_state.current_tool == "1":
            st.markdown("### 🎯 Targeted Packs")
            st.caption("Evidence-backed additions")
            
            for idx, pack in enumerate(st.session_state.refinement_packs):
                selected = idx in st.session_state.selected_packs
                card_class = "selected" if selected else ""
                
                if st.checkbox(f"**{pack.get('title', '')}** {pack.get('lift', '')}", key=f"pack_{idx}"):
                    st.session_state.selected_packs.add(idx)
                else:
                    st.session_state.selected_packs.discard(idx)
                
                with st.expander("Details"):
                    st.write("**Tokens:**", ", ".join(pack.get('tokens', [])))
                    st.write("**JD:**", pack.get('jd_evidence', ''))
                    st.write("**Resume:**", pack.get('resume_evidence', ''))
            
            if st.button("Apply Selected Packs", type="primary", use_container_width=True):
                if st.session_state.selected_packs:
                    selected_pack_data = [st.session_state.refinement_packs[i] for i in st.session_state.selected_packs]
                    
                    with st.spinner("Applying packs..."):
                        apply_prompt = f"""Apply these keyword packs to the resume:

PACKS:
{json.dumps(selected_pack_data, indent=2)}

CURRENT RESUME:
{st.session_state.tailored_resume}

JOB DESCRIPTION:
{st.session_state.current_jd}

RULES:
- Only add keywords where evidence exists
- Maintain ATS-safe format
- End with [END_RESUME]

Generate updated resume:"""
                        
                        updated_resume = call_gemini(apply_prompt, temperature=0.7)
                        st.session_state.tailored_resume = updated_resume
                        st.session_state.selected_packs = set()
                        st.session_state.current_tool = None
                        st.rerun()

st.markdown("---")
st.caption("Built with Streamlit + Gemini 2.0 Flash | ReadysetRole v1.0")

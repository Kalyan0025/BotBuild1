# --- Main Layout -----------------------------------------------------------
# Top: Document Uploads (Full Width)
st.markdown("### 📤 Upload Your Documents")

col_resume_upload, col_jd_upload = st.columns(2)

with col_resume_upload:
    uploaded_resume = st.file_uploader(
        "Master Resume",
        type=["pdf", "docx", "txt"],
        help="Upload your comprehensive resume (PDF, DOCX, or TXT)",
        key="resume_uploader"
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

with col_jd_upload:
    jd_input_method = st.radio("Job Description Input", ["Paste Text", "Upload File"], horizontal=True, key="jd_method")
    
    if jd_input_method == "Paste Text":
        jd_text = st.text_area(
            "Paste Job Description",
            height=120,
            placeholder="Paste the complete job description here...",
            key="jd_textarea"
        )
        if st.button("🎯 AutoTailor Resume", type="primary", use_container_width=True):
            if jd_text and st.session_state.master_resume:
                st.session_state.current_jd = jd_text
                st.session_state.current_tool = None
                st.rerun()
    else:
        uploaded_jd = st.file_uploader("Upload JD", type=["pdf", "docx", "txt"], key="jd_uploader")
        if uploaded_jd:
            jd_text = parse_resume_file(uploaded_jd)
            if jd_text and st.button("🎯 AutoTailor Resume", type="primary", use_container_width=True):
                st.session_state.current_jd = jd_text
                st.session_state.current_tool = None
                st.rerun()

st.markdown("---")

# Main Editor Layout: Left (Editor) + Right (Score Tracker)
col_editor, col_tracker = st.columns([2.5, 1], gap="large")

with col_editor:
    # AutoTailor Process (same as before)
    if st.session_state.current_jd and st.session_state.master_resume and not st.session_state.tailored_resume:
        with st.spinner("🔄 AutoTailoring your resume..."):
            # [Keep all the AutoTailor logic here - same as before]
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
        
        st.markdown("---")
        
        # Change Log in Editor
        if st.session_state.change_log:
            with st.expander("📝 Change Log - See What Changed", expanded=False):
                for item in st.session_state.change_log:
                    st.markdown(f"""
                    <div class="change-item">
                        <div class="change-before">{item.get('before', '')}</div>
                        <div class="change-after">{item.get('after', '')}</div>
                        <div class="change-why">{item.get('why', '')}</div>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Evidence Map in Editor
        if st.session_state.evidence_map:
            with st.expander("📍 Evidence Map - How We Matched", expanded=False):
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
        
        # Refinement Tools
        if st.session_state.current_tool == "1" and st.session_state.refinement_packs:
            st.markdown("### 🎯 Targeted Refinement Packs")
            st.caption("Select packs to boost your score")
            
            for idx, pack in enumerate(st.session_state.refinement_packs):
                selected = idx in st.session_state.selected_packs
                
                if st.checkbox(f"**{pack.get('title', '')}** {pack.get('lift', '')}", key=f"pack_{idx}", value=selected):
                    st.session_state.selected_packs.add(idx)
                else:
                    st.session_state.selected_packs.discard(idx)
                
                if selected:
                    st.write("**Tokens:**", ", ".join(pack.get('tokens', [])))
                    st.caption(f"**JD:** {pack.get('jd_evidence', '')}")
                    st.caption(f"**Resume:** {pack.get('resume_evidence', '')}")
            
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
        
        # No Fabrication Oath
        st.markdown('<div class="no-fabrication">🔒 No-Fabrication Oath: We only used your resume + JD. Nothing else.</div>', unsafe_allow_html=True)

with col_tracker:
    st.markdown("### 📊 Score Tracker")
    
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
        st.markdown("---")
        st.markdown("### 🎯 Next Actions")
        
        options = [
            ("1", "Refine Packs", "🎯"),
            ("2", "Minor Boosters", "✨"),
            ("3", "Coverage Board", "📊"),
            ("4", "Narrative Presets", "🎭"),
            ("5", "A/B Bullet", "🔀"),
            ("6", "Level Calibrator", "⚖️"),
            ("7", "Export .txt", "💾"),
            ("0", "Next JD", "🔄")
        ]
        
        for num, label, icon in options:
            if st.button(f"{icon} {label}", key=f"opt_{num}", use_container_width=True):
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
                    # Already handled in editor
                    pass
                else:
                    st.session_state.current_tool = num
                    st.rerun()

st.markdown("---")
st.caption("Built with Streamlit + Gemini 2.0 Flash | ReadysetRole v1.0")

# --- Custom CSS ------------------------------------------------------------
st.markdown("""
<style>
    /* Import Inter font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');
    
    /* FORCE DARK MODE - Override Streamlit defaults */
    .stApp {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}
    
    /* Dark theme for all text */
    .stApp, .stApp * {
        color: #e0e0e0 !important;
    }
    
    /* Override Streamlit's white backgrounds */
    .stApp > div > div {
        background: transparent !important;
    }
    
    /* File uploader dark styling */
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
    
    .stFileUploader > div {
        background: transparent !important;
    }
    
    /* Text area dark styling */
    .stTextArea textarea {
        background: #1e1e2e !important;
        color: #e0e0e0 !important;
        border: 2px solid #7e22ce !important;
        border-radius: 12px !important;
        font-family: 'Inter', sans-serif !important;
    }
    
    .stTextArea label {
        color: #a78bfa !important;
        font-weight: 600 !important;
    }
    
    /* Buttons */
    .stButton button {
        background: linear-gradient(135deg, #7e22ce 0%, #6d28d9 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 8px 25px rgba(126, 34, 206, 0.4) !important;
    }
    
    /* Primary button (AutoTailor) */
    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        font-size: 1.1rem !important;
        padding: 1rem 2rem !important;
    }
    
    /* Radio buttons */
    .stRadio {
        background: rgba(30, 30, 46, 0.6) !important;
        padding: 1rem !important;
        border-radius: 12px !important;
    }
    
    .stRadio label {
        color: #a78bfa !important;
    }
    
    /* Score card styling */
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
        line-height: 1 !important;
        color: white !important;
    }
    
    .score-label {
        font-size: 0.9rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        opacity: 0.9;
        margin-top: 0.5rem;
        color: white !important;
    }
    
    .delta-positive {
        color: #10b981 !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    /* Metrics (subscores) */
    .stMetric {
        background: rgba(126, 34, 206, 0.2) !important;
        padding: 1rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(126, 34, 206, 0.3) !important;
    }
    
    .stMetric label {
        color: #a78bfa !important;
        font-weight: 600 !important;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: white !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    
    /* Evidence row */
    .evidence-row {
        background: linear-gradient(135deg, rgba(126, 34, 206, 0.15) 0%, rgba(30, 60, 114, 0.15) 100%);
        border-left: 4px solid #7e22ce;
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 0.8rem;
    }
    
    .jd-anchor {
        font-weight: 600 !important;
        color: #a78bfa !important;
        font-size: 0.95rem !important;
    }
    
    .resume-match {
        color: #d1d5db !important;
        font-size: 0.88rem !important;
        margin-top: 0.5rem;
    }
    
    .badge-verified {
        background: #10b981 !important;
        color: white !important;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-left: 0.5rem;
    }
    
    .badge-pending {
        background: #f59e0b !important;
        color: white !important;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-left: 0.5rem;
    }
    
    /* Change log */
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
        font-size: 0.88rem;
    }
    
    .change-after {
        color: #10b981 !important;
        font-weight: 600 !important;
        font-size: 0.88rem !important;
        margin-top: 0.3rem;
    }
    
    .change-why {
        color: #9ca3af !important;
        font-size: 0.8rem !important;
        font-style: italic;
        margin-top: 0.3rem;
    }
    
    /* Resume output */
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
    
    .end-marker {
        color: #10b981 !important;
        font-weight: 700 !important;
        text-align: center;
        margin-top: 1rem;
    }
    
    /* Checkboxes */
    .stCheckbox {
        background: rgba(30, 30, 46, 0.6) !important;
        padding: 0.5rem !important;
        border-radius: 8px !important;
    }
    
    .stCheckbox label {
        color: #e0e0e0 !important;
        font-weight: 600 !important;
    }
    
    /* Expander */
    .stExpander {
        background: rgba(30, 30, 46, 0.6) !important;
        border: 1px solid rgba(126, 34, 206, 0.3) !important;
        border-radius: 12px !important;
    }
    
    .stExpander label {
        color: #a78bfa !important;
    }
    
    /* Download button */
    .stDownloadButton button {
        background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
        color: white !important;
    }
    
    /* Info box */
    .stAlert {
        background: rgba(126, 34, 206, 0.2) !important;
        border: 2px solid #7e22ce !important;
        border-radius: 12px !important;
        color: #e0e0e0 !important;
    }
    
    /* Success box */
    .stSuccess {
        background: rgba(16, 185, 129, 0.2) !important;
        border: 2px solid #10b981 !important;
        color: #e0e0e0 !important;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-top-color: #7e22ce !important;
    }
    
    /* No fabrication badge */
    .no-fabrication {
        background: linear-gradient(135deg, #059669 0%, #10b981 100%) !important;
        color: white !important;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        text-align: center;
        font-weight: 700 !important;
        margin: 1.5rem 0;
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
    }
    
    /* Column styling */
    .stColumn {
        background: transparent !important;
    }
    
    /* Section headers */
    h1, h2, h3, h4, h5, h6 {
        color: white !important;
        font-weight: 700 !important;
    }
    
    /* Caption text */
    .stCaption {
        color: #9ca3af !important;
    }
    
    /* Code blocks */
    .stCode {
        background: #0f0f1e !important;
        border: 1px solid rgba(126, 34, 206, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

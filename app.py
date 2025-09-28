# ReadySetRole — Simple Gemini Chatbot (Streamlit)
# Fixes: repetition, hallucination guard, instruction upload, small+clean UI
# Author credit retained per original template license.
# "This code uses portions of code developed by Ronald A. Beghetto for a course taught at Arizona State University."

import io
import time
import re
import mimetypes
from difflib import SequenceMatcher
from typing import List, Dict, Any, Optional

import streamlit as st
from PIL import Image

from google import genai
from google.genai import types

# ---------------------------
# Page + Header
# ---------------------------
st.set_page_config(
    page_title="ReadySetRole Bot",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Centered header image (optional)
try:
    col1, col2, col3 = st.columns([1, 6, 1])
    with col2:
        st.image(Image.open("Bot1.png"), caption="ReadySetRole (2025)", width=340)
except Exception:
    pass

st.markdown("<h1 style='text-align:center'>ReadySetRole</h1>", unsafe_allow_html=True)
sub = (
    "I can make mistakes — please verify important information. "
    "No fabrication: I use only your resume, the job description, and what you explicitly approve."
)
st.markdown(f"<div style='text-align:center;color:gray;font-size:12px'>{sub}</div>", unsafe_allow_html=True)

# ---------------------------
# Helpers
# ---------------------------

def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def load_default_identity() -> str:
    return (
        "You are ReadysetRole — a resume optimization assistant that turns a user's master resume "
        "and a job description (JD) into an ATS-safe tailored resume and a concise, evidence-based cover letter.\n\n"
        "Rules:\n"
        "- Never fabricate titles, employers, tools, certs, or metrics.\n"
        "- Use only what the user provides or has explicitly verified.\n"
        "- Keep outputs plain-text, single column, ATS-safe.\n"
        "- If impact numbers are missing, insert <METRIC_TBD> and ask a precise follow-up.\n"
        "- Be concise, confident, and non-repetitive.\n"
        "- If you are about to repeat the same guidance as the previous turn, instead summarize in one line and ask what to refine.\n"
        "- Persist the user's master resume for the session; whenever a new JD is provided, immediately tailor the resume to it.\n"
        "- Always return a non-empty response — never output 'None'.\n"
    )


def parse_xmlish_instr(txt: str) -> str:
    """Extracts <Role>, <Goal>, <Rules>, <Knowledge>, <SpecializedActions>, <Guidelines> into one system instruction string."""
    import re as _re
    sections = ["Role", "Goal", "Rules", "Knowledge", "SpecializedActions", "Guidelines"]
    chunks = []
    for tag in sections:
        m = _re.search(fr"<{tag}>(.*?)</{tag}>", txt, flags=_re.DOTALL | _re.IGNORECASE)
        if m:
            chunks.append(f"{tag}:\n{m.group(1).strip()}\n\n")
    return "".join(chunks).strip() or load_default_identity()


def ensure_active_files(client: genai.Client, files_meta: List[Dict[str, Any]], max_wait_s: float = 12.0):
    deadline = time.time() + max_wait_s
    any_processing = True
    while any_processing and time.time() < deadline:
        any_processing = False
        for i, meta in enumerate(files_meta):
            fobj = meta["file"]
            if getattr(fobj, "state", "") != "ACTIVE":
                any_processing = True
                try:
                    files_meta[i]["file"] = client.files.get(name=fobj.name)
                except Exception:
                    pass
        if any_processing:
            time.sleep(0.5)


def too_similar(a: str, b: str, threshold: float = 0.90) -> bool:
    if not a or not b:
        return False
    ratio = SequenceMatcher(None, a.strip(), b.strip()).ratio()
    return ratio >= threshold


def extract_text_from_response(resp) -> str:
    """Robustly extract text across SDK variants; avoid printing raw Candidate objects."""
    try:
        # Newer SDKs
        t = getattr(resp, "text", None)
        if t:
            return t
        # Candidates/parts path
        out = []
        candidates = getattr(resp, "candidates", None) or []
        for c in candidates:
            content = getattr(c, "content", None)
            parts = getattr(content, "parts", None) or []
            for p in parts:
                pt = getattr(p, "text", None)
                if pt:
                    out.append(pt)
        if out:
            return "
".join(out).strip()
    except Exception:
        pass
    return ""

    st.markdown("### Model")
    model_name = st.selectbox(
        "Choose a model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
        index=1,
    )

    st.markdown("### Generation Settings")
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
    top_p = st.slider("top_p", 0.0, 1.0, 0.9, 0.05)
    top_k = st.slider("top_k", 1, 100, 40, 1)
    max_tokens = st.number_input("max_output_tokens", min_value=256, max_value=4096, value=3072, step=64)
    concise_mode = st.toggle("Concise mode (short answers)", value=True, help="Adds extra brevity hints to the system instruction")

    st.divider()

    st.markdown("### System Instructions")
    instr_src = st.radio("Load instructions from:", ["identity.txt", "Paste inline"], index=1, horizontal=True)
    pasted = ""
    identity_text = load_default_identity()
    if instr_src == "identity.txt":
        try:
            with open("identity.txt", "r") as f:
                raw = f.read()
            identity_text = parse_xmlish_instr(raw) if "<Role>" in raw else raw
        except FileNotFoundError:
            st.warning("identity.txt not found — using default identity.")
    else:
        pasted = st.text_area(
            "Paste your <Role>…<Guidelines> block (optional)",
            height=220,
            placeholder="<Role>…</Role>\n<Goal>…</Goal>\n<Rules>…</Rules>\n…",
        )
        if pasted.strip():
            identity_text = parse_xmlish_instr(pasted)

    if concise_mode:
        identity_text += "\n\nStyle: Prefer concise bullets and avoid repetition across turns."

    st.caption("Effective system prompt in use:")
    with st.expander("Show system prompt"):
        st.code(identity_text)

# ---------------------------
# Client + Session setup
# ---------------------------
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("Failed to init Gemini client. Set GEMINI_API_KEY in Streamlit secrets.\n" + str(e))
    st.stop()

st.session_state.setdefault("chat_history", [])  # our UI echo of messages
st.session_state.setdefault("last_assistant_text", "")

# Persisted resume for the session
st.session_state.setdefault("resume_meta", None)    # {name,size,mime,file}
st.session_state.setdefault("resume_text", "")      # optional pasted text

# JD text cache (not persisted as file; uploaded JD triggers tailoring immediately)
st.session_state.setdefault("jd_text_temp", "")

# Create/refresh chat with current config
search_tool = types.Tool(google_search=types.GoogleSearch())

generation_cfg = types.GenerateContentConfig(
    system_instruction=identity_text,
    tools=[search_tool],
    temperature=float(temperature),
    top_p=float(top_p),
    top_k=int(top_k),
    max_output_tokens=int(max_tokens),
    response_mime_type="text/plain",
)

if "chat" not in st.session_state:
    st.session_state.chat = client.chats.create(model=model_name, config=generation_cfg)
else:
    if getattr(st.session_state.chat, "model", None) != model_name:
        st.session_state.chat = client.chats.create(model=model_name, config=generation_cfg)
    else:
        try:
            st.session_state.chat.update(config=generation_cfg)
        except Exception:
            st.session_state.chat = client.chats.create(model=model_name, config=generation_cfg)

# ---------------------------
# Main Inputs (CENTER) — Resume first, then JD each time
# ---------------------------
resume_container = st.container()
with resume_container:
    if not (st.session_state["resume_meta"] or st.session_state["resume_text"].strip()):
        st.info("**Step 1 — Upload your Master Resume** (stored for this session only)")
        col_a, col_b = st.columns(2)
        with col_a:
            up = st.file_uploader("Upload Resume (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"], accept_multiple_files=False, key="resume_uploader_center")
            if up is not None:
                try:
                    mime = up.type or (mimetypes.guess_type(up.name)[0] or "application/octet-stream")
                    gfile = client.files.upload(file=io.BytesIO(up.getvalue()), config=types.UploadFileConfig(mime_type=mime))
                    st.session_state["resume_meta"] = {"name": up.name, "size": up.size, "mime": mime, "file": gfile}
                    st.toast(f"Resume on file: {up.name}")
                except Exception as e:
                    st.error(f"Upload failed for {up.name}: {e}")
        with col_b:
            st.session_state["resume_text"] = st.text_area("Or paste Resume text", value=st.session_state["resume_text"], height=140)
            if st.button("Use Pasted Resume", type="primary"):
                if st.session_state["resume_text"].strip():
                    st.toast("Resume text stored for session")
                else:
                    st.warning("Paste some resume text first.")
    else:
        # Show resume on file + option to replace
        meta = st.session_state.get("resume_meta")
        msg = "**Resume on file:** "
        if meta:
            msg += f"{meta['name']} ({human_size(meta['size'])})"
        else:
            msg += "(pasted text)"
        st.success(msg)
        if st.button("Replace Resume"):
            st.session_state["resume_meta"] = None
            st.session_state["resume_text"] = ""
            st.rerun()

# JD input appears only after resume is present
jd_container = st.container()
with jd_container:
    if st.session_state["resume_meta"] or st.session_state["resume_text"].strip():
        st.info("**Step 2 — Provide a Job Description (JD)**. Each time you add a JD, I will tailor your resume.")
        col1, col2 = st.columns(2)
        with col1:
            jd_up = st.file_uploader("Upload JD (PDF/TXT/DOCX)", type=["pdf", "txt", "docx"], accept_multiple_files=False, key="jd_uploader_center")
        with col2:
            st.session_state["jd_text_temp"] = st.text_area("Or paste JD text", value=st.session_state["jd_text_temp"], height=140)
            jd_paste_click = st.button("Use This JD")

        # Auto-run when JD file is uploaded, or when paste button clicked
        def process_with_current_resume_and_jd(jd_meta: Optional[Dict[str,Any]] = None, jd_text: str = ""):
            # Compose an explicit, deterministic instruction to kick off the pipeline
            command = (
                """
Run QuickScore on the provided RESUME and JD, then continue as follows:
1) Output PRE-SCORE with subscores (skills_fit, experience_fit, education_fit, ats_keywords_coverage) in a clear block.
2) List up to 4 KEYWORD PACKS (numbered 1..N) with short labels and predicted score lift.
3) End with: 'Reply with 1,2,3 or 0 for all to apply.'
Constraints: Keep total output under 500 words. No fabrication; use only resume + JD evidence. ATS-safe formatting.
                """
            )
            parts = [types.Part.from_text(text=command)]
            if st.session_state.get("resume_text"):
                parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
            if st.session_state.get("resume_meta"):
                parts.append(st.session_state["resume_meta"]["file"])
            if jd_text.strip():
                parts.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
            if jd_meta and jd_meta.get("file"):
                parts.append(jd_meta["file"])

            with st.chat_message("user", avatar="👤"):
                st.markdown("JD provided — starting QuickScore and suggestions…")
            st.session_state.chat_history.append({"role": "user", "parts": "JD provided — starting QuickScore and suggestions…"})

            try:
                with st.chat_message("assistant", avatar=":material/robot_2:"):
                    with st.spinner("Scoring resume ↔ JD and preparing keyword packs…"):
                        response = st.session_state.chat.send_message(parts)
                    full_response = extract_text_from_response(response) or "[No content returned — try increasing max_output_tokens or switching to gemini-2.5-pro]"
                    if too_similar(st.session_state.last_assistant_text, full_response):
                        full_response = (
                            "I've already proposed packs above. Reply with numbers (e.g., 1,3) or 0 to apply all."
                        )
                    st.markdown(full_response)
                st.session_state.chat_history.append({"role": "assistant", "parts": full_response})
                st.session_state.last_assistant_text = full_response
            except Exception as e:
                st.error(f"❌ Gemini error: {e}")

        # Handle JD file upload
        if jd_up is not None:
            try:
                mime = jd_up.type or (mimetypes.guess_type(jd_up.name)[0] or "application/octet-stream")
                jd_gfile = client.files.upload(file=io.BytesIO(jd_up.getvalue()), config=types.UploadFileConfig(mime_type=mime))
                jd_meta = {"name": jd_up.name, "size": jd_up.size, "mime": mime, "file": jd_gfile}
                ensure_active_files(client, [jd_meta])
                st.toast(f"JD received: {jd_up.name}")
                process_with_current_resume_and_jd(jd_meta=jd_meta, jd_text="")
            except Exception as e:
                st.error(f"Upload failed for {jd_up.name}: {e}")

        # Handle pasted JD
        if jd_paste_click and st.session_state["jd_text_temp"].strip():
            process_with_current_resume_and_jd(jd_meta=None, jd_text=st.session_state["jd_text_temp"]) 

# ---------------------------
# Render prior messages
# ---------------------------
for msg in st.session_state.chat_history:
    avatar = "👤" if msg["role"] == "user" else ":material/robot_2:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["parts"])  # plain text

# ---------------------------
# Open chat input for follow-ups
# ---------------------------
user_prompt = st.chat_input("Reply with packs to apply (e.g., 1,3 or 0 for all), or ask follow‑ups…")
if user_prompt:
    st.session_state.chat_history.append({"role": "user", "parts": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    # If no resume yet, nudge
    need_resume = not (st.session_state.get("resume_meta") or st.session_state.get("resume_text"))
    if need_resume:
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            st.markdown("Please upload your **Master Resume** first above. Then add a JD; I’ll score and propose packs.")
        st.stop()

    # Detect a pack selection like '0' or '1,3,4'
    pack_match = re.fullmatch(r"\s*0\s*|\s*(\d+\s*(,\s*\d+\s*)*)\s*", user_prompt)

    try:
        if pack_match:
            # Tell the model to apply packs and finish the flow
            selection = user_prompt.strip()
            apply_cmd = (
                "ApplySuggestions with the selected packs: '" + selection + "'. "
                "Then generate: (a) the tailored ATS-safe resume, (b) Post-Score with delta vs Pre-Score, "
                "(c) a concise evidence-based cover letter (180–250 words), and (d) a short Change Log (before→after bullets). "
                "If metrics are missing, insert <METRIC_TBD> and list 3 micro-questions via AskForMetrics at the end."
            )
            parts = [types.Part.from_text(text=apply_cmd)]
            if st.session_state.get("resume_text"):
                parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
            if st.session_state.get("resume_meta"):
                parts.append(st.session_state["resume_meta"]["file"])  # resume file
            if st.session_state.get("jd_text_temp"):
                parts.append(types.Part.from_text(text="[JD TEXT]\n" + st.session_state["jd_text_temp"]))

            with st.chat_message("assistant", avatar=":material/robot_2:"):
                with st.spinner("Applying your selections and generating outputs…"):
                    response = st.session_state.chat.send_message(parts)
                full_response = extract_text_from_response(response) or "[No content returned — try increasing max_output_tokens or switching to gemini-2.5-pro]"
                if too_similar(st.session_state.last_assistant_text, full_response):
                    full_response = "Outputs generated above. Want me to export a plain-text file?"
                st.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "parts": full_response})
            st.session_state.last_assistant_text = full_response
        else:
            # General follow-up: include resume + current JD text context
            parts = [types.Part.from_text(text=user_prompt)]
            if st.session_state.get("resume_text"):
                parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
            if st.session_state.get("resume_meta"):
                parts.append(st.session_state["resume_meta"]["file"])  # resume file
            if st.session_state.get("jd_text_temp"):
                parts.append(types.Part.from_text(text="[JD TEXT]\n" + st.session_state["jd_text_temp"]))
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                with st.spinner("Thinking…"):
                    response = st.session_state.chat.send_message(parts)
                full_response = extract_text_from_response(response) or "[No content returned — try increasing max_output_tokens or switching to gemini-2.5-pro]"
                if too_similar(st.session_state.last_assistant_text, full_response):
                    full_response = "I've covered that above. Want me to export, or propose boosters?"
                st.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "parts": full_response})
            st.session_state.last_assistant_text = full_response
    except Exception as e:
        st.error(f"❌ Gemini error: {e}")

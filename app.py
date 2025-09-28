# ReadySetRole — Simple Gemini Chatbot (Streamlit)
# Deterministic flow with stage tracking; robust response parsing; resume persisted for session
# Two-step "Apply" to avoid truncation: (A) Resume only → (B) Post-Score + Letter + Change Log + Options
# License note per template:
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
st.set_page_config(page_title="ReadySetRole Bot", layout="centered", initial_sidebar_state="expanded")

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
        "- Persist the user's master resume for the session; whenever a new JD is provided, immediately tailor the resume to it.\n"
        "- Always return a non-empty response — never output 'None'.\n"
    )


def parse_xmlish_instr(txt: str) -> str:
    """Extract <Role>, <Goal>, <Rules>, <Knowledge>, <SpecializedActions>, <Guidelines> into one system string."""
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
        t = getattr(resp, "text", None)
        if t:
            return t
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
            return "\n".join(out).strip()
    except Exception:
        pass
    return ""

# ---------------------------
# Sidebar (settings only — no uploads)
# ---------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.caption("Gemini settings and system instructions")

    st.markdown("### Model")
    model_name = st.selectbox(
        "Choose a model",
        ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
        index=0,
    )

    st.markdown("### Generation Settings")
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
    top_p = st.slider("top_p", 0.0, 1.0, 0.9, 0.05)
    top_k = st.slider("top_k", 1, 100, 40, 1)
    max_tokens = st.number_input("max_output_tokens", min_value=512, max_value=4096, value=3584, step=64)
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

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("last_assistant_text", "")

# Persisted resume for the session
st.session_state.setdefault("resume_meta", None)    # {name,size,mime,file}
st.session_state.setdefault("resume_text", "")      # optional pasted text

# JD + flow state
st.session_state.setdefault("jd_text_temp", "")
st.session_state.setdefault("last_jd_meta", None)   # remember most recent JD file
st.session_state.setdefault("awaiting_pack_selection", False)
st.session_state.setdefault("awaiting_next_options", False)
st.session_state.setdefault("last_apply_output", "")

# Chat config
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
# Main Inputs (center) — resume first, then JD
# ---------------------------
resume_container = st.container()
with resume_container:
    if not (st.session_state["resume_meta"] or st.session_state["resume_text"].strip()):
        st.info("**Step 1 — Upload your Master Resume** (stored for this session only)")
        col_a, col_b = st.columns(2)
        with col_a:
            up = st.file_uploader("Upload Resume (PDF/TXT/DOCX)", ["pdf", "txt", "docx"], False, key="resume_uploader_center")
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
        meta = st.session_state.get("resume_meta")
        msg = "**Resume on file:** "
        msg += f"{meta['name']} ({human_size(meta['size'])})" if meta else "(pasted text)"
        st.success(msg)
        if st.button("Replace Resume"):
            st.session_state["resume_meta"] = None
            st.session_state["resume_text"] = ""
            st.session_state["awaiting_pack_selection"] = False
            st.session_state["awaiting_next_options"] = False
            st.session_state["last_apply_output"] = ""
            st.rerun()

jd_container = st.container()
with jd_container:
    if st.session_state["resume_meta"] or st.session_state["resume_text"].strip():
        st.info("**Step 2 — Provide a Job Description (JD)**. Each time you add a JD, I will tailor your resume.")
        col1, col2 = st.columns(2)
        with col1:
            jd_up = st.file_uploader("Upload JD (PDF/TXT/DOCX)", ["pdf", "txt", "docx"], False, key="jd_uploader_center")
        with col2:
            st.session_state["jd_text_temp"] = st.text_area("Or paste JD text", value=st.session_state["jd_text_temp"], height=140)
            jd_paste_click = st.button("Use This JD")

        def process_with_current_resume_and_jd(jd_meta: Optional[Dict[str, Any]] = None, jd_text: str = ""):
            # Kickoff: Pre-Score + up to 4 Packs (≤ ~500 words)
            command = (
                "Run QuickScore on the provided RESUME and JD, then continue as follows:\n"
                "1) Output PRE-SCORE with subscores (skills_fit, experience_fit, education_fit, ats_keywords_coverage) in a clear block.\n"
                "2) List up to 4 KEYWORD PACKS (numbered 1..N) with short labels and predicted score lift.\n"
                "3) End with: 'Reply with 1,2,3 or 0 for all to apply.'\n"
                "Constraints: Keep total output under 500 words. No fabrication; use only resume + JD evidence. ATS-safe formatting."
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
                        full_response = "I've already proposed packs above. Reply with numbers (e.g., 1,3) or 0 to apply all."
                    st.markdown(full_response)
                st.session_state.chat_history.append({"role": "assistant", "parts": full_response})
                st.session_state.last_assistant_text = full_response
                # Expect pack selection next
                st.session_state["awaiting_pack_selection"] = True
                st.session_state["awaiting_next_options"] = False
            except Exception as e:
                st.error(f"❌ Gemini error: {e}")

        # JD file upload
        if jd_up is not None:
            try:
                mime = jd_up.type or (mimetypes.guess_type(jd_up.name)[0] or "application/octet-stream")
                jd_gfile = client.files.upload(file=io.BytesIO(jd_up.getvalue()), config=types.UploadFileConfig(mime_type=mime))
                jd_meta = {"name": jd_up.name, "size": jd_up.size, "mime": mime, "file": jd_gfile}
                ensure_active_files(client, [jd_meta])
                st.session_state["last_jd_meta"] = jd_meta  # remember file for apply stage
                st.toast(f"JD received: {jd_up.name}")
                process_with_current_resume_and_jd(jd_meta=jd_meta, jd_text="")
            except Exception as e:
                st.error(f"Upload failed for {jd_up.name}: {e}")

        # JD pasted
        if jd_paste_click and st.session_state["jd_text_temp"].strip():
            st.session_state["last_jd_meta"] = None  # this run uses pasted text
            process_with_current_resume_and_jd(jd_meta=None, jd_text=st.session_state["jd_text_temp"]) 

# ---------------------------
# Render prior messages
# ---------------------------
for msg in st.session_state.chat_history:
    avatar = "👤" if msg["role"] == "user" else ":material/robot_2:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["parts"])  # plain text

# ---------------------------
# Chat input — packs → apply → next options
# ---------------------------
user_prompt = st.chat_input("Reply with packs (e.g., 1,3 or 0 for all). After apply: 1=boosters, 2=next JD, 3=export.")
if user_prompt:
    st.session_state.chat_history.append({"role": "user", "parts": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    need_resume = not (st.session_state.get("resume_meta") or st.session_state.get("resume_text"))
    if need_resume:
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            st.markdown("Please upload your **Master Resume** first above. Then add a JD; I’ll score and propose packs.")
        st.stop()

    pack_match = re.fullmatch(r"\s*0\s*|\s*(\d+\s*(,\s*\d+\s*)*)\s*", user_prompt)
    option_match = re.fullmatch(r"\s*[123]\s*", user_prompt)

    try:
        # A) APPLY PACKS (only if we are expecting them) — TWO-STEP TO AVOID TRUNCATION
        if st.session_state.get("awaiting_pack_selection") and pack_match:
            selection = user_prompt.strip()

            # --- STEP A: Apply + Tailored Resume ONLY ---
            apply_cmd_a = (
                "ApplySuggestions with the selected packs: '" + selection + "'. "
                "Output ONLY the Tailored ATS-safe resume (plain text). "
                "Do NOT include Post-Score, percentages, cover letter, change log, or extra commentary. "
                "End with a line exactly: [END_RESUME]"
            )
            parts_a = [types.Part.from_text(text=apply_cmd_a)]
            if st.session_state.get("resume_text"):
                parts_a.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
            if st.session_state.get("resume_meta"):
                parts_a.append(st.session_state["resume_meta"]["file"])
            if st.session_state.get("jd_text_temp"):
                parts_a.append(types.Part.from_text(text="[JD TEXT]\n" + st.session_state["jd_text_temp"]))
            if st.session_state.get("last_jd_meta"):
                parts_a.append(st.session_state["last_jd_meta"]["file"])

            with st.chat_message("assistant", avatar=":material/robot_2:"):
                with st.spinner("Applying packs — generating tailored resume…"):
                    resp_a = st.session_state.chat.send_message(parts_a)
                resume_only = extract_text_from_response(resp_a) or "[Resume content missing]"
                st.markdown(resume_only)

            # --- STEP B: Post-Score + Cover Letter + Change Log + Options ---
            apply_cmd_b = (
                "Now output the remaining items for the same tailoring you just performed: "
                "2) POST-SCORE (overall 0–100) and Δ vs PRE-SCORE. "
                "3) Concise evidence-based cover letter (180–250 words). "
                "4) Short Change Log (before→after bullets). "
                "At the very end, print on separate lines: NEW MATCH %: <overall>\n"
                "NEXT OPTIONS: [1] Suggest minor boosters (no new packs), [2] Accept & move to next JD, [3] Export .txt. "
                "Do NOT repeat the resume and do NOT propose new keyword packs."
            )
            parts_b = [types.Part.from_text(text=apply_cmd_b)]
            if st.session_state.get("resume_text"):
                parts_b.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
            if st.session_state.get("resume_meta"):
                parts_b.append(st.session_state["resume_meta"]["file"])
            if st.session_state.get("jd_text_temp"):
                parts_b.append(types.Part.from_text(text="[JD TEXT]\n" + st.session_state["jd_text_temp"]))
            if st.session_state.get("last_jd_meta"):
                parts_b.append(st.session_state["last_jd_meta"]["file"])

            with st.chat_message("assistant", avatar=":material/robot_2:"):
                with st.spinner("Finishing — Post-Score, cover letter, and change log…"):
                    resp_b = st.session_state.chat.send_message(parts_b)
                rest = extract_text_from_response(resp_b) or "[Details missing]"
                st.markdown(rest)

            # Save combined output for export
            st.session_state["last_apply_output"] = (resume_only.rstrip() + "\n\n" + rest.lstrip()).strip()

            # Move to next-options phase
            st.session_state["awaiting_pack_selection"] = False
            st.session_state["awaiting_next_options"] = True
            st.session_state["last_assistant_text"] = st.session_state["last_apply_output"]
            st.session_state.chat_history.append({"role": "assistant", "parts": st.session_state["last_apply_output"]})

        # B) NEXT OPTIONS (after apply): 1/2/3
        elif st.session_state.get("awaiting_next_options") and option_match:
            choice = option_match.group(0).strip()
            if choice == "1":
                # Minor boosters — no new packs
                boost_cmd = (
                    "BoostScore(): Suggest 3–6 minor booster edits to the tailored resume above without proposing new keyword packs. "
                    "Focus on phrasing, ordering, quantification (<METRIC_TBD>), and alignment tweaks. "
                    "End with exactly three micro-questions asking for missing metric values."
                )
                parts = [types.Part.from_text(text=boost_cmd)]
                if st.session_state.get("resume_text"):
                    parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
                if st.session_state.get("resume_meta"):
                    parts.append(st.session_state["resume_meta"]["file"])  # resume file
                if st.session_state.get("jd_text_temp"):
                    parts.append(types.Part.from_text(text="[JD TEXT]\n" + st.session_state["jd_text_temp"]))
                if st.session_state.get("last_jd_meta"):
                    parts.append(st.session_state["last_jd_meta"]["file"])  # JD file
                with st.chat_message("assistant", avatar=":material/robot_2:"):
                    with st.spinner("Preparing targeted boosters…"):
                        response = st.session_state.chat.send_message(parts)
                    full_response = extract_text_from_response(response) or "[No content returned — try increasing max_output_tokens or switching to gemini-2.5-pro]"
                    st.markdown(full_response)
                st.session_state.chat_history.append({"role": "assistant", "parts": full_response})
                st.session_state.last_assistant_text = full_response
                st.session_state["awaiting_next_options"] = True

            elif choice == "2":
                # Accept & move to next JD
                with st.chat_message("assistant", avatar=":material/robot_2:"):
                    st.markdown("Great — upload/paste the **next JD** and I’ll reuse your master resume on file.")
                st.session_state.chat_history.append({"role": "assistant", "parts": "Proceed with next JD."})
                st.session_state.last_assistant_text = "Proceed with next JD."
                st.session_state["awaiting_next_options"] = False
                st.session_state["jd_text_temp"] = ""
                st.session_state["last_jd_meta"] = None

            elif choice == "3":
                with st.chat_message("assistant", avatar=":material/robot_2:"):
                    if st.session_state.get("last_apply_output"):
                        st.download_button(
                            "Download tailored resume + cover letter (.txt)",
                            data=st.session_state["last_apply_output"].encode("utf-8"),
                            file_name="readysetrole_results.txt",
                            mime="text/plain",
                            use_container_width=True,
                        )
                    else:
                        st.markdown("No recent outputs to export. Apply packs first.")
                st.session_state.chat_history.append({"role": "assistant", "parts": "Export option shown."})
                st.session_state.last_assistant_text = "Export option shown."
                st.session_state["awaiting_next_options"] = True

        # C) General follow-up
        else:
            parts = [types.Part.from_text(text=user_prompt)]
            if st.session_state.get("resume_text"):
                parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
            if st.session_state.get("resume_meta"):
                parts.append(st.session_state["resume_meta"]["file"])  # resume file
            if st.session_state.get("jd_text_temp"):
                parts.append(types.Part.from_text(text="[JD TEXT]\n" + st.session_state["jd_text_temp"]))
            if st.session_state.get("last_jd_meta"):
                parts.append(st.session_state["last_jd_meta"]["file"])  # JD file
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                with st.spinner("Thinking…"):
                    response = st.session_state.chat.send_message(parts)
                full_response = extract_text_from_response(response) or "[No content returned — try increasing max_output_tokens or switching to gemini-2.5-pro]"
                if too_similar(st.session_state.last_assistant_text, full_response):
                    full_response = "I've covered that above. Want boosters (1), accept & next JD (2), or export (3)?"
                st.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "parts": full_response})
            st.session_state.last_assistant_text = full_response

    except Exception as e:
        st.error(f"❌ Gemini error: {e}")

# ReadySetRole — AutoTailor Flow with Action Buttons (Streamlit + Gemini)
# Core UX: AutoTailor → PostScore + Evidence Map + Change Log → Action Bar (buttons)
# On-demand tools: 3 targeted packs, Minor boosters, Coverage Board, Narrative presets,
# A/B Bullet rewriter, Level calibrator, Export, Next JD. Resume is persisted for the session.
# License note per template:
# "This code uses portions of code developed by Ronald A. Beghetto for a course taught at Arizona State University."

import io
import time
import re
import mimetypes
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
    return """You are ReadysetRole — a resume optimization assistant that turns a user's master resume and a job description (JD) into an ATS-safe tailored resume and a concise, evidence-based cover letter.

Rules:
- Never fabricate titles, employers, tools, certs, or metrics.
- Use only what the user provides or has explicitly verified.
- Keep outputs plain-text, single column, ATS-safe.
- If impact numbers are missing, insert <METRIC_TBD> and ask a precise follow-up.
- Be concise, confident, and non-repetitive.
- Persist the user's master resume for the session; whenever a new JD is provided, immediately tailor the resume to it.
- Always return a non-empty response — never output 'None'.
"""


def parse_xmlish_instr(txt: str) -> str:
    """Extracts <Role>, <Goal>, <Rules>, <Knowledge>, <SpecializedActions>, <Guidelines>
    into one system string. Avoids f-strings entirely.
    """
    import re as _re
    sections = ["Role", "Goal", "Rules", "Knowledge", "SpecializedActions", "Guidelines"]
    chunks: List[str] = []
    for tag in sections:
        pattern = "<" + tag + ">(.*?)</" + tag + ">"
        m = _re.search(pattern, txt, flags=_re.DOTALL | _re.IGNORECASE)
        if m:
            chunks.append(tag + ":\n" + m.group(1).strip() + "\n\n")
    out = "".join(chunks).strip()
    return out if out else load_default_identity()


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


def ends_with_end_resume(text: str) -> bool:
    return text.strip().endswith("[END_RESUME]")


def trim_after_end_resume(text: str) -> str:
    idx = text.find("[END_RESUME]")
    if idx == -1:
        return text
    return text[:idx + len("[END_RESUME]")]

# ---------------------------
# Sidebar (settings only — no uploads here)
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
    concise_mode = st.toggle("Concise mode (short answers)", value=True, help="Adds brevity hints to the system instruction")

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

st.session_state.setdefault("chat_history", [])  # **store only USER messages to prevent duplicate renders**

# Persisted resume for the session
st.session_state.setdefault("resume_meta", None)    # {name,size,mime,file}
st.session_state.setdefault("resume_text", "")      # optional pasted text

# JD + flow state
st.session_state.setdefault("jd_text_temp", "")
st.session_state.setdefault("last_jd_meta", None)

# Stage flags
st.session_state.setdefault("stage", "idle")  # idle | options | awaiting_packs | awaiting_ab_choice | awaiting_level
st.session_state.setdefault("last_outputs", {})  # {"resume": str, "post": str}

# On-demand tool sub-states
st.session_state.setdefault("awaiting_ab_bullet", False)
st.session_state.setdefault("awaiting_ab_choice", False)
st.session_state.setdefault("awaiting_level_choice", False)
st.session_state.setdefault("custom_refine_text", "")

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
# Core prompt senders
# ---------------------------

def send(parts: List[Any]) -> str:
    # filter out Nones to avoid SDK issues
    parts = [p for p in parts if p is not None]
    resp = st.session_state.chat.send_message(parts)
    return extract_text_from_response(resp) or ""


def autotailor_resume_only(jd_meta: Optional[Dict[str, Any]], jd_text: str) -> str:
    cmd = """AutoTailorNow:
- Use the provided RESUME + JD only.
- Do NOT fabricate employers, titles, tools, or metrics.
- Convert implied matches into exact JD phrasing only when evidence exists.
- Reorder bullets so JD-critical items appear first; compress low-relevance content.
- Keep ATS-safe plain text, single column; insert <METRIC_TBD> where numbers are missing.
- Output ONLY the tailored resume (plain text). Do not include explanations.
- End with exactly: [END_RESUME]"""
    parts: List[Any] = [types.Part.from_text(text=cmd)]
    if st.session_state.get("resume_text"):
        parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
    if st.session_state.get("resume_meta"):
        parts.append(st.session_state["resume_meta"]["file"])
    if jd_text.strip():
        parts.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
    if jd_meta and jd_meta.get("file"):
        parts.append(jd_meta["file"])
    return send(parts)


def poststep_scores_and_proofs(jd_meta: Optional[Dict[str, Any]], jd_text: str) -> str:
    cmd = """PostStep:
- Compute PRE-SCORE (overall + subscores) and POST-SCORE (overall + subscores); show Δ.
- Evidence Map (mini): 4–6 lines mapping {JD anchor → matching resume bullet fragment}. Use short quotes only.
- Change Log (max 6 bullets): before→after with a one-clause 'why' tied to the JD.
- At the end, print on separate lines:
NEW MATCH %: <overall-post>
NEXT OPTIONS: [1] Improve further (3 targeted packs), [2] Minor boosters, [3] Coverage Board, [4] Narrative presets, [5] A/B Bullet, [6] Level calibrator, [7] Export .txt, [0] Next JD"""
    parts: List[Any] = [types.Part.from_text(text=cmd)]
    if st.session_state.get("resume_text"):
        parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
    if st.session_state.get("resume_meta"):
        parts.append(st.session_state["resume_meta"]["file"])
    if jd_text.strip():
        parts.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
    if jd_meta and jd_meta.get("file"):
        parts.append(jd_meta["file"])
    return send(parts)


def suggest_three_packs(jd_meta: Optional[Dict[str, Any]], jd_text: str) -> str:
    cmd = """SuggestRefinementPacks (after AutoTailor):
- Propose exactly 3 narrowly scoped packs (2–5 tokens/phrases each).
- Each pack must be fully evidenced by the tailored resume and JD; exclude anything without proof.
- For each pack, include: short label, tokens/phrases, JD anchor (short quote), resume anchor (short quote), predicted lift (+2–6%).
- Keep under 180 words total.
- End with: Reply with 1,2 or 0 to apply all."""
    parts: List[Any] = [types.Part.from_text(text=cmd)]
    if st.session_state.get("resume_text"):
        parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
    if st.session_state.get("resume_meta"):
        parts.append(st.session_state["resume_meta"]["file"])
    if jd_text.strip():
        parts.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
    if jd_meta and jd_meta.get("file"):
        parts.append(jd_meta["file"])
    return send(parts)


def apply_selected_packs(selection: str, jd_meta: Optional[Dict[str, Any]], jd_text: str) -> str:
    cmd_a = (
        "ApplySuggestions with the selected refinement packs: '" + selection + "'. "
        "Output ONLY the updated ATS-safe resume (plain text) and end with: [END_RESUME]"
    )
    parts_a: List[Any] = [types.Part.from_text(text=cmd_a)]
    if st.session_state.get("resume_text"):
        parts_a.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
    if st.session_state.get("resume_meta"):
        parts_a.append(st.session_state["resume_meta"]["file"])
    if jd_text.strip():
        parts_a.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
    if jd_meta and jd_meta.get("file"):
        parts_a.append(jd_meta["file"])
    return send(parts_a)


def post_after_apply(jd_meta: Optional[Dict[str, Any]], jd_text: str) -> str:
    cmd_b = """Now output only: POST-SCORE (overall 0–100) and Δ vs previous; a very short Change Log (max 4 bullets).
At the end print:
NEW MATCH %: <overall>
NEXT OPTIONS: [1] Improve further (3 targeted packs), [2] Minor boosters, [3] Coverage Board, [4] Narrative presets, [5] A/B Bullet, [6] Level calibrator, [7] Export .txt, [0] Next JD"""
    parts_b: List[Any] = [types.Part.from_text(text=cmd_b)]
    if st.session_state.get("resume_text"):
        parts_b.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
    if st.session_state.get("resume_meta"):
        parts_b.append(st.session_state["resume_meta"]["file"])
    if jd_text.strip():
        parts_b.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
    if jd_meta and jd_meta.get("file"):
        parts_b.append(jd_meta["file"])
    return send(parts_b)


def minor_boosters(jd_meta: Optional[Dict[str, Any]], jd_text: str, user_brief: str) -> str:
    extra = (" Also incorporate the user's brief (only if evidenced): '''" + user_brief + "'''.") if user_brief.strip() else ""
    cmd = (
        "BoostScore(): Suggest 3–6 minor booster edits to the tailored resume above without proposing new keywords. "
        "Focus on phrasing, ordering, and quantification (<METRIC_TBD>), alignment tweaks." + extra +
        " End with exactly three micro-questions asking for missing metric values."
    )
    parts: List[Any] = [types.Part.from_text(text=cmd)]
    if st.session_state.get("resume_text"):
        parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
    if st.session_state.get("resume_meta"):
        parts.append(st.session_state["resume_meta"]["file"])
    if jd_text.strip():
        parts.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
    if jd_meta and jd_meta.get("file"):
        parts.append(jd_meta["file"])
    return send(parts)


def coverage_board(jd_meta: Optional[Dict[str, Any]], jd_text: str) -> str:
    cmd = """Coverage Board:
- Group into Must-Have (role-critical), Nice-to-Have (differentiators), and Avoid/Exposure (no evidence).
- For each item: JD anchor (short quote), resume anchor (short quote), placement (Summary/Skills/Role bullet), risk flag, predicted lift (+%).
- Keep total under 220 words."""
    parts: List[Any] = [types.Part.from_text(text=cmd)]
    if st.session_state.get("resume_text"):
        parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
    if st.session_state.get("resume_meta"):
        parts.append(st.session_state["resume_meta"]["file"])
    if jd_text.strip():
        parts.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
    if jd_meta and jd_meta.get("file"):
        parts.append(jd_meta["file"])
    return send(parts)


def narrative_presets(jd_meta: Optional[Dict[str, Any]], jd_text: str) -> str:
    cmd = """NarrativePresetGuide:
- Present 3–4 presets (Impact-First / Research-Forward / Systems & Accessibility / Product Partnering).
- For each: 1–2 tone rules and a single sample bullet rewrite drawn from the tailored resume.
- No new tools/titles; keep ATS-safe. Max 180 words."""
    parts: List[Any] = [types.Part.from_text(text=cmd)]
    if st.session_state.get("resume_text"):
        parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
    if st.session_state.get("resume_meta"):
        parts.append(st.session_state["resume_meta"]["file"])
    if jd_text.strip():
        parts.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
    if jd_meta and jd_meta.get("file"):
        parts.append(jd_meta["file"])
    return send(parts)


def ab_bullet_variants(jd_meta: Optional[Dict[str, Any]], jd_text: str, bullet_text: str) -> str:
    cmd = (
        "ABullet:\n"
        "- Rewrite this single bullet in three variants (A/B/C), each in a different preset (Impact-First, Research-Forward, Systems/Accessibility): '''"
        + bullet_text + "'''.\n"
        "- Keep facts/tools/titles as-is; ATS-safe; no fabrication. Return clearly labeled A, B, C."
    )
    parts: List[Any] = [types.Part.from_text(text=cmd)]
    if st.session_state.get("resume_text"):
        parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
    if st.session_state.get("resume_meta"):
        parts.append(st.session_state["resume_meta"]["file"])
    if jd_text.strip():
        parts.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
    if jd_meta and jd_meta.get("file"):
        parts.append(jd_meta["file"])
    return send(parts)


def apply_ab_choice(jd_meta: Optional[Dict[str, Any]], jd_text: str, chosen_text: str) -> str:
    cmd_a = (
        "Integrate the chosen rewritten bullet into the tailored resume above, replacing the most relevant original bullet only. "
        "Keep everything ATS-safe; do not fabricate. Output ONLY the updated resume and end with: [END_RESUME]\n"
        "Chosen bullet: '''" + chosen_text + "'''"
    )
    parts_a: List[Any] = [types.Part.from_text(text=cmd_a)]
    if st.session_state.get("resume_text"):
        parts_a.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
    if st.session_state.get("resume_meta"):
        parts_a.append(st.session_state["resume_meta"]["file"])
    if jd_text.strip():
        parts_a.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
    if jd_meta and jd_meta.get("file"):
        parts_a.append(jd_meta["file"])
    return send(parts_a)


def level_calibrator_prompt(jd_meta: Optional[Dict[str, Any]], jd_text: str, level: str) -> str:
    level = level.lower().strip()
    cmd_a = (
        "LevelCalibrator:\n"
        "- Reframe scope language for a " + level + " lens (ownership verbs, impact scale, collaboration framing) without adding new facts.\n"
        "- Keep ATS-safe and evidence-aligned. Output ONLY the updated resume; end with: [END_RESUME]"
    )
    parts_a: List[Any] = [types.Part.from_text(text=cmd_a)]
    if st.session_state.get("resume_text"):
        parts_a.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
    if st.session_state.get("resume_meta"):
        parts_a.append(st.session_state["resume_meta"]["file"])
    if jd_text.strip():
        parts_a.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
    if jd_meta and jd_meta.get("file"):
        parts_a.append(jd_meta["file"])
    return send(parts_a)

# ---------------------------
# AutoTailor kickoff (single place). No assistant messages are stored in chat_history → prevents duplicates.
# ---------------------------

def run_autotailor_flow(jd_meta: Optional[Dict[str, Any]], jd_text: str):
    with st.chat_message("user", avatar="👤"):
        st.markdown("JD provided — AutoTailor in progress…")

    # Step A — resume only (with auto-continue safety)
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        with st.spinner("Tailoring resume to JD…"):
            resume_part = autotailor_resume_only(jd_meta, jd_text)
        combined = resume_part
        attempts = 0
        while not ends_with_end_resume(combined) and attempts < 3:
            attempts += 1
            cont = send([
                types.Part.from_text(text="Continue the tailored resume exactly where it stopped. Do not repeat previous lines. Output ONLY the remaining resume text. End with: [END_RESUME]"),
                types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state.get("resume_text", "")),
                st.session_state["resume_meta"]["file"] if st.session_state.get("resume_meta") else None,
                types.Part.from_text(text="[JD TEXT]\n" + jd_text) if jd_text.strip() else None,
                jd_meta["file"] if jd_meta else None
            ])
            if cont.strip():
                combined = (combined.rstrip() + "\n" + cont.lstrip()).strip()
            else:
                break
        combined = trim_after_end_resume(combined)
        st.markdown(combined)

    # Step B — scores + proofs + options
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        with st.spinner("Scoring and preparing evidence…"):
            rest = poststep_scores_and_proofs(jd_meta, jd_text)
        st.markdown(rest)

    st.session_state["last_outputs"] = {"resume": combined, "post": rest}
    st.session_state["stage"] = "options"

# ---------------------------
# Main Inputs (center)
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
            st.session_state["stage"] = "idle"
            st.session_state["last_outputs"] = {}
            st.rerun()

jd_container = st.container()
with jd_container:
    if st.session_state["resume_meta"] or st.session_state["resume_text"].strip():
        st.info("**Step 2 — Provide a Job Description (JD)**. On upload, I’ll AutoTailor your resume.")
        col1, col2 = st.columns(2)
        with col1:
            jd_up = st.file_uploader("Upload JD (PDF/TXT/DOCX)", ["pdf", "txt", "docx"], False, key="jd_uploader_center")
        with col2:
            st.session_state["jd_text_temp"] = st.text_area("Or paste JD text", value=st.session_state["jd_text_temp"], height=140)
            jd_paste_click = st.button("Use This JD")

        if jd_up is not None:
            try:
                mime = jd_up.type or (mimetypes.guess_type(jd_up.name)[0] or "application/octet-stream")
                jd_gfile = client.files.upload(file=io.BytesIO(jd_up.getvalue()), config=types.UploadFileConfig(mime_type=mime))
                jd_meta = {"name": jd_up.name, "size": jd_up.size, "mime": mime, "file": jd_gfile}
                ensure_active_files(client, [jd_meta])
                st.session_state["last_jd_meta"] = jd_meta
                st.toast(f"JD received: {jd_up.name}")
                run_autotailor_flow(jd_meta, "")
            except Exception as e:
                st.error(f"Upload failed for {jd_up.name}: {e}")

        if jd_paste_click and st.session_state["jd_text_temp"].strip():
            st.session_state["last_jd_meta"] = None
            run_autotailor_flow(None, st.session_state["jd_text_temp"])  # AutoTailor now

# ---------------------------
# Render prior USER messages only (prevents assistant duplicates)
# ---------------------------
for msg in st.session_state.chat_history:
    if msg["role"] == "user":
        with st.chat_message("user", avatar="👤"):
            st.markdown(msg["parts"])

# ---------------------------
# ACTION BAR (buttons) — shown after AutoTailor PostStep
# ---------------------------
if st.session_state.get("stage") == "options":
    st.markdown("---")
    st.subheader("Next actions")

    # Optional refine brief (shown above the action buttons)
    st.session_state["custom_refine_text"] = st.text_area(
        "Optional refine brief (used by Minor boosters)",
        value=st.session_state.get("custom_refine_text", ""),
        height=90,
        placeholder="e.g., tighten summary to 45 words, surface 'Figma' in Skills, quantify Mayo with users impacted",
    )

    # Buttons row 1
    c1, c2, c3, c4 = st.columns(4)
    btn_improve = c1.button("Improve (3 packs)")
    btn_boost = c2.button("Minor boosters")
    btn_coverage = c3.button("Coverage Board")
    btn_narr = c4.button("Narrative presets")

    # Buttons row 2
    d1, d2, d3, d4 = st.columns(4)
    btn_ab = d1.button("A/B Bullet")
    btn_level = d2.button("Level: junior/mid/senior")
    btn_export = d3.button("Export .txt")
    btn_nextjd = d4.button("Next JD")

    # --- Handlers
    if btn_improve:
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            with st.spinner("Preparing three targeted, evidence-backed packs…"):
                out = suggest_three_packs(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp", ""))
            st.markdown(out)
            st.info("Apply selection:")
            e1, e2, e3, e4 = st.columns([1, 1, 1, 2])
            if e1.button("Apply 1"): st.session_state["_apply_selection"] = "1"
            if e2.button("Apply 2"): st.session_state["_apply_selection"] = "2"
            if e3.button("Apply 3"): st.session_state["_apply_selection"] = "3"
            if e4.button("Apply all (0)"): st.session_state["_apply_selection"] = "0"
        st.session_state["stage"] = "awaiting_packs"

    if btn_boost:
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            with st.spinner("Suggesting minor boosters…"):
                out = minor_boosters(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp", ""), st.session_state.get("custom_refine_text", ""))
            st.markdown(out)

    if btn_coverage:
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            with st.spinner("Building Coverage Board…"):
                st.markdown(coverage_board(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp", "")))

    if btn_narr:
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            with st.spinner("Showing narrative presets…"):
                st.markdown(narrative_presets(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp", "")))

    if btn_ab:
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            st.markdown("Paste the **single bullet** below, then click *Generate A/B/C*.")
            bullet_text = st.text_area("Bullet to rewrite", height=80, key="ab_bullet_text")
            if st.button("Generate A/B/C") and bullet_text.strip():
                with st.spinner("Creating variants…"):
                    out = ab_bullet_variants(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp", ""), bullet_text)
                st.markdown(out)
                g1, g2, g3 = st.columns(3)
                if g1.button("Use A"): st.session_state["_ab_choice"] = "A"
                if g2.button("Use B"): st.session_state["_ab_choice"] = "B"
                if g3.button("Use C"): st.session_state["_ab_choice"] = "C"
                st.session_state["stage"] = "awaiting_ab_choice"

    if btn_level:
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            st.markdown("Choose a level to reframe scope safely:")
            l1, l2, l3 = st.columns(3)
            if l1.button("junior"): st.session_state["_level_choice"] = "junior"
            if l2.button("mid"): st.session_state["_level_choice"] = "mid"
            if l3.button("senior"): st.session_state["_level_choice"] = "senior"
            st.session_state["stage"] = "awaiting_level"

    if btn_export:
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            if st.session_state.get("last_outputs"):
                bundle = (st.session_state["last_outputs"].get("resume", "") + "\n\n" + st.session_state["last_outputs"].get("post", "")).strip()
                st.download_button(
                    "Download tailored resume + evidence (.txt)",
                    data=bundle.encode("utf-8"),
                    file_name="readysetrole_results.txt",
                    mime="text/plain",
                    use_container_width=True,
                )
            else:
                st.markdown("No recent outputs to export. Run AutoTailor first.")

    if btn_nextjd:
        st.session_state["stage"] = "options"  # keep options visible
        st.session_state["jd_text_temp"] = ""
        st.session_state["last_jd_meta"] = None
        st.session_state["show_next_jd_panel"] = True
        st.toast("Scroll down to '+ Add Next JD' to submit another role.")

# ---------------------------
# Handle follow-up sub-stages from buttons
# ---------------------------
if st.session_state.get("stage") == "awaiting_packs" and st.session_state.get("_apply_selection"):
    sel = st.session_state.pop("_apply_selection")
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        with st.spinner("Applying selected packs — updating resume…"):
            resume_only = apply_selected_packs(sel, st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp", ""))
        # ensure full resume
        combined = resume_only
        tries = 0
        while not ends_with_end_resume(combined) and tries < 3:
            tries += 1
            cont = send([
                types.Part.from_text(text="Continue the tailored resume exactly where it stopped. Do not repeat previous lines. Output ONLY the remaining resume text. End with: [END_RESUME]"),
                types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state.get("resume_text", "")),
                st.session_state["resume_meta"]["file"] if st.session_state.get("resume_meta") else None,
                types.Part.from_text(text="[JD TEXT]\n" + st.session_state.get("jd_text_temp", "")) if st.session_state.get("jd_text_temp", "").strip() else None,
                st.session_state["last_jd_meta"]["file"] if st.session_state.get("last_jd_meta") else None
            ])
            if cont.strip():
                combined = (combined.rstrip() + "\n" + cont.lstrip()).strip()
            else:
                break
        combined = trim_after_end_resume(combined)
        st.markdown(combined)
        with st.spinner("Re-scoring and summarizing changes…"):
            rest = post_after_apply(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp", ""))
        st.markdown(rest)
    st.session_state["last_outputs"] = {"resume": combined, "post": rest}
    st.session_state["stage"] = "options"

if st.session_state.get("stage") == "awaiting_ab_choice" and st.session_state.get("_ab_choice"):
    choice = st.session_state.pop("_ab_choice")
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        with st.spinner(f"Applying variant {choice} and updating resume…"):
            updated = apply_ab_choice(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp", ""), f"Choice {choice}")
        st.markdown(updated)
        with st.spinner("Re-scoring and summarizing changes…"):
            post = post_after_apply(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp", ""))
        st.markdown(post)
    st.session_state["last_outputs"] = {"resume": updated, "post": post}
    st.session_state["stage"] = "options"

if st.session_state.get("stage") == "awaiting_level" and st.session_state.get("_level_choice"):
    level = st.session_state.pop("_level_choice")
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        with st.spinner(f"Reframing to {level} lens…"):
            updated = level_calibrator_prompt(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp", ""), level)
        st.markdown(updated)
        with st.spinner("Re-scoring and summarizing changes…"):
            post = post_after_apply(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp", ""))
        st.markdown(post)
    st.session_state["last_outputs"] = {"resume": updated, "post": post}
    st.session_state["stage"] = "options"

# ---------------------------
# Bottom “Next JD” panel — always visible once options are shown
# ---------------------------
if st.session_state.get("stage") in ("options", "awaiting_packs", "awaiting_ab_choice", "awaiting_level") or st.session_state.get("show_next_jd_panel"):
    st.markdown("---")
    st.subheader("➕ Add Next JD")
    nj_col1, nj_col2 = st.columns(2)
    with nj_col1:
        next_jd_up = st.file_uploader("Upload next JD (PDF/TXT/DOCX)", ["pdf", "txt", "docx"], False, key="next_jd_up")
    with nj_col2:
        next_jd_text = st.text_area("Or paste next JD text", value="", height=120, key="next_jd_text")
        next_jd_btn = st.button("Use This Next JD", type="primary", key="use_next_jd")

    if next_jd_up is not None:
        try:
            mime = next_jd_up.type or (mimetypes.guess_type(next_jd_up.name)[0] or "application/octet-stream")
            next_jd_gfile = client.files.upload(file=io.BytesIO(next_jd_up.getvalue()), config=types.UploadFileConfig(mime_type=mime))
            jd_meta2 = {"name": next_jd_up.name, "size": next_jd_up.size, "mime": mime, "file": next_jd_gfile}
            ensure_active_files(client, [jd_meta2])
            st.session_state["last_jd_meta"] = jd_meta2
            st.session_state["jd_text_temp"] = ""
            st.toast(f"JD received: {next_jd_up.name}")
            run_autotailor_flow(jd_meta2, "")
        except Exception as e:
            st.error(f"Upload failed for {next_jd_up.name}: {e}")

    if next_jd_btn and next_jd_text.strip():
        st.session_state["last_jd_meta"] = None
        st.session_state["jd_text_temp"] = next_jd_text
        run_autotailor_flow(None, next_jd_text)

# ---------------------------
# Minimal chat input — **disabled** during options (we drive via buttons)
# ---------------------------
if st.session_state.get("stage") in ("idle",):
    user_prompt = st.chat_input("Optional: ask a question about the process.")
    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "parts": user_prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_prompt)
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            out = send([types.Part.from_text(text=user_prompt)])
            st.markdown(out or "Got it.")

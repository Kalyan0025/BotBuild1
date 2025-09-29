# ReadySetRole — AutoTailor Flow (Streamlit + Gemini)
# Core: AutoTailor → PostScore (with Evidence Map + Change Log) → Options
# On-demand tools: Targeted packs, Minor boosters, Coverage Board, Narrative presets,
# A/B Bullet rewriter, Level calibrator, Export, Next JD
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
# Sidebar (settings only)
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
    max_tokens = st.number_input("max_output_tokens", 512, 4096, 3584, 64)
    concise_mode = st.toggle("Concise mode (short answers)", value=True)

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
st.session_state.setdefault("resume_text", "")      # pasted text

# JD + flow state
st.session_state.setdefault("jd_text_temp", "")
st.session_state.setdefault("last_jd_meta", None)
st.session_state.setdefault("awaiting_next_options", False)
st.session_state.setdefault("show_next_jd_panel", False)
st.session_state.setdefault("last_apply_output", "")

# On-demand tool states
st.session_state.setdefault("awaiting_refine_pack_selection", False)  # for option [1]
st.session_state.setdefault("awaiting_ab_bullet", False)              # for option [5]
st.session_state.setdefault("awaiting_ab_choice", False)              # choice A/B/C
st.session_state.setdefault("ab_variants", {})                        # store A/B/C text
st.session_state.setdefault("awaiting_level_choice", False)           # for option [6]
st.session_state.setdefault("custom_refine_text", "")                 # optional brief for boosters

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
# Core prompts as functions
# ---------------------------
def send(parts: List[Any]) -> str:
    resp = st.session_state.chat.send_message(parts)
    return extract_text_from_response(resp) or ""

def autotailor_resume_only(jd_meta: Optional[Dict[str, Any]], jd_text: str) -> str:
    cmd = (
        "AutoTailorNow:\n"
        "- Use the provided RESUME + JD only.\n"
        "- Do NOT fabricate employers, titles, tools, or metrics.\n"
        "- Convert implied matches into exact JD phrasing only when evidence exists.\n"
        "- Reorder bullets so JD-critical items appear first; compress low-relevance content.\n"
        "- Keep ATS-safe plain text, single column; insert <METRIC_TBD> where numbers are missing.\n"
        "- Output ONLY the tailored resume (plain text). Do not include explanations.\n"
        "- End with exactly: [END_RESUME]"
    )
    parts = [types.Part.from_text(text=cmd)]
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
    cmd = (
        "PostStep:\n"
        "- Compute PRE-SCORE (overall + subscores) and POST-SCORE (overall + subscores); show Δ.\n"
        "- Evidence Map (mini): 4–6 lines mapping {JD anchor → matching resume bullet fragment}. Use short quotes only.\n"
        "- Change Log (max 6 bullets): before→after with a one-clause 'why' tied to the JD.\n"
        "- At the end, print on separate lines:\n"
        "NEW MATCH %: <overall-post>\n"
        "NEXT OPTIONS: [1] Improve further (3 targeted packs), [2] Minor boosters, [3] Coverage Board, [4] Narrative presets, [5] A/B Bullet, [6] Level calibrator, [7] Export .txt, [0] Next JD"
    )
    parts = [types.Part.from_text(text=cmd)]
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
    cmd = (
        "SuggestRefinementPacks (after AutoTailor):\n"
        "- Propose exactly 3 narrowly scoped packs (2–5 tokens/phrases each).\n"
        "- Each pack must be fully evidenced by the tailored resume and JD; exclude anything without proof.\n"
        "- For each pack, include: short label, tokens/phrases, JD anchor (short quote), resume anchor (short quote), predicted lift (+2–6%).\n"
        "- Keep under 180 words total.\n"
        "- End with: Reply with 1,2 or 0 to apply all."
    )
    parts = [types.Part.from_text(text=cmd)]
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
    parts_a = [types.Part.from_text(text=cmd_a)]
    if st.session_state.get("resume_text"):
        parts_a.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
    if st.session_state.get("resume_meta"):
        parts_a.append(st.session_state["resume_meta"]["file"])
    if jd_text.strip():
        parts_a.append(types.Part.from_text(text="[JD TEXT]\n" + jd_text.strip()))
    if jd_meta and jd_meta.get("file"):
        parts_a.append(jd_meta["file"])
    resp_a = send(parts_a)
    return resp_a

def post_after_apply(jd_meta: Optional[Dict[str, Any]], jd_text: str, micropacks: bool = True) -> str:
    cmd_b = (
        "Now output only: POST-SCORE (overall 0–100) and Δ vs previous; a very short Change Log (max 4 bullets). "
        "At the end print:\n"
        "NEW MATCH %: <overall>\n"
        "NEXT OPTIONS: [1] Improve further (3 targeted packs), [2] Minor boosters, [3] Coverage Board, [4] Narrative presets, [5] A/B Bullet, [6] Level calibrator, [7] Export .txt, [0] Next JD"
    )
    parts_b = [types.Part.from_text(text=cmd_b)]
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
    extra = f" Also incorporate the user's brief (only if evidenced): '''{user_brief}'''." if user_brief.strip() else ""
    cmd = (
        "BoostScore(): Suggest 3–6 minor booster edits to the tailored resume above without proposing new keywords. "
        "Focus on phrasing, ordering, and quantification (<METRIC_TBD>), alignment tweaks." + extra +
        " End with exactly three micro-questions asking for missing metric values."
    )
    parts = [types.Part.from_text(text=cmd)]
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
    cmd = (
        "Coverage Board:\n"
        "- Group into Must-Have (role-critical), Nice-to-Have (differentiators), and Avoid/Exposure (no evidence).\n"
        "- For each item: JD anchor (short quote), resume anchor (short quote), placement (Summary/Skills/Role bullet), risk flag, predicted lift (+%).\n"
        "- Keep total under 220 words."
    )
    parts = [types.Part.from_text(text=cmd)]
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
    cmd = (
        "NarrativePresetGuide:\n"
        "- Present 3–4 presets (Impact-First / Research-Forward / Systems & Accessibility / Product Partnering).\n"
        "- For each: 1–2 tone rules and a single sample bullet rewrite drawn from the tailored resume.\n"
        "- No new tools/titles; keep ATS-safe. Max 180 words."
    )
    parts = [types.Part.from_text(text=cmd)]
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
        f"- Rewrite this single bullet in three variants (A/B/C), each in a different preset (Impact-First, Research-Forward, Systems/Accessibility): '''{bullet_text}'''.\n"
        "- Keep facts/tools/titles as-is; ATS-safe; no fabrication. Return clearly labeled A, B, C."
    )
    parts = [types.Part.from_text(text=cmd)]
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
        f"Chosen bullet: '''{chosen_text}'''"
    )
    parts_a = [types.Part.from_text(text=cmd_a)]
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
        f"- Reframe scope language for a {level} lens (ownership verbs, impact scale, collaboration framing) without adding new facts.\n"
        "- Keep ATS-safe and evidence-aligned. Output ONLY the updated resume; end with: [END_RESUME]"
    )
    parts_a = [types.Part.from_text(text=cmd_a)]
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
# AutoTailor kickoff
# ---------------------------
def run_autotailor_flow(jd_meta: Optional[Dict[str, Any]], jd_text: str):
    with st.chat_message("user", avatar="👤"):
        st.markdown("JD provided — AutoTailor in progress…")

    # Step A — resume only (with auto-continue)
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        with st.spinner("Tailoring resume to JD…"):
            resume_part = autotailor_resume_only(jd_meta, jd_text)
        # auto-continue up to 3 times
        combined = resume_part
        attempts = 0
        while not ends_with_end_resume(combined) and attempts < 3:
            attempts += 1
            cont = send([
                types.Part.from_text(text="Continue the tailored resume exactly where it stopped. "
                                           "Do not repeat previous lines. Output ONLY the remaining resume text. "
                                           "End with: [END_RESUME]"),
                types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state.get("resume_text","")),
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

    st.session_state["last_apply_output"] = (combined.rstrip() + "\n\n" + rest.lstrip()).strip()
    st.session_state["awaiting_next_options"] = True
    st.session_state["show_next_jd_panel"] = True
    st.session_state.chat_history.append({"role": "assistant", "parts": st.session_state["last_apply_output"]})

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
            st.session_state["awaiting_next_options"] = False
            st.session_state["show_next_jd_panel"] = False
            st.session_state["last_apply_output"] = ""
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
            run_autotailor_flow(None, st.session_state["jd_text_temp"])

# ---------------------------
# Render prior messages
# ---------------------------
for msg in st.session_state.chat_history:
    avatar = "👤" if msg["role"] == "user" else ":material/robot_2:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["parts"])

# ---------------------------
# Bottom “Next JD” panel (shows after NEXT OPTIONS)
# ---------------------------
if st.session_state.get("awaiting_next_options") or st.session_state.get("show_next_jd_panel"):
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
# NEXT OPTIONS — chat input
# ---------------------------
placeholder = ("After AutoTailor, choose: "
               "[1] Improve (3 packs) • [2] Minor boosters • [3] Coverage Board • [4] Narrative presets • "
               "[5] A/B Bullet • [6] Level calibrator • [7] Export • [0] Next JD.\n"
               "If [1], reply with packs later as '1,2' or '0'. If [5], I’ll ask for a bullet text.")
user_prompt = st.chat_input(placeholder)

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "parts": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    need_resume = not (st.session_state.get("resume_meta") or st.session_state.get("resume_text"))
    if need_resume:
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            st.markdown("Please upload your **Master Resume** first above.")
        st.stop()

    # Sub-step waits
    if st.session_state.get("awaiting_ab_bullet"):
        bullet = user_prompt.strip()
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            with st.spinner("Creating A/B/C variants…"):
                out = ab_bullet_variants(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp",""), bullet)
            st.markdown(out)
            st.markdown("_Reply with **A**, **B**, or **C** to apply your chosen variant._")
        st.session_state["awaiting_ab_bullet"] = False
        st.session_state["awaiting_ab_choice"] = True
        st.stop()

    if st.session_state.get("awaiting_ab_choice"):
        choice = user_prompt.strip().upper()
        if choice not in ("A", "B", "C"):
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                st.markdown("Please reply with **A**, **B**, or **C**.")
            st.stop()
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            with st.spinner(f"Applying variant {choice} and updating resume…"):
                updated = apply_ab_choice(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp",""), f"Choice {choice}")
            st.markdown(updated)
            with st.spinner("Re-scoring and summarizing changes…"):
                post = post_after_apply(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp",""), micropacks=False)
            st.markdown(post)
        st.session_state["last_apply_output"] = (updated.rstrip() + "\n\n" + post.lstrip()).strip()
        st.session_state["awaiting_ab_choice"] = False
        st.session_state["awaiting_next_options"] = True
        st.session_state["show_next_jd_panel"] = True
        st.stop()

    if st.session_state.get("awaiting_level_choice"):
        level = user_prompt.strip().lower()
        if level not in ("junior", "mid", "senior"):
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                st.markdown("Please reply with **junior**, **mid**, or **senior**.")
            st.stop()
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            with st.spinner(f"Reframing to {level} lens…"):
                updated = level_calibrator_prompt(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp",""), level)
            st.markdown(updated)
            with st.spinner("Re-scoring and summarizing changes…"):
                post = post_after_apply(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp",""), micropacks=False)
            st.markdown(post)
        st.session_state["last_apply_output"] = (updated.rstrip() + "\n\n" + post.lstrip()).strip()
        st.session_state["awaiting_level_choice"] = False
        st.session_state["awaiting_next_options"] = True
        st.session_state["show_next_jd_panel"] = True
        st.stop()

    # Pack selection after option [1]
    pack_match = re.fullmatch(r"\s*0\s*|\s*(\d+\s*(,\s*\d+\s*)*)\s*", user_prompt)
    if st.session_state.get("awaiting_refine_pack_selection") and pack_match:
        selection = user_prompt.strip()
        with st.chat_message("assistant", avatar=":material/robot_2:"):
            with st.spinner("Applying selected packs — updating resume…"):
                resume_only = apply_selected_packs(selection, st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp",""))
            combined = resume_only
            tries = 0
            while not ends_with_end_resume(combined) and tries < 3:
                tries += 1
                cont = send([
                    types.Part.from_text(text="Continue the tailored resume exactly where it stopped. "
                                               "Do not repeat previous lines. Output ONLY the remaining resume text. "
                                               "End with: [END_RESUME]"),
                    types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state.get("resume_text","")),
                    st.session_state["resume_meta"]["file"] if st.session_state.get("resume_meta") else None,
                    types.Part.from_text(text="[JD TEXT]\n" + st.session_state.get("jd_text_temp","")) if st.session_state.get("jd_text_temp","").strip() else None,
                    st.session_state["last_jd_meta"]["file"] if st.session_state.get("last_jd_meta") else None
                ])
                if cont.strip():
                    combined = (combined.rstrip() + "\n" + cont.lstrip()).strip()
                else:
                    break
            combined = trim_after_end_resume(combined)
            st.markdown(combined)

            with st.spinner("Re-scoring and summarizing changes…"):
                rest = post_after_apply(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp",""))
            st.markdown(rest)
        st.session_state["last_apply_output"] = (combined.rstrip() + "\n\n" + rest.lstrip()).strip()
        st.session_state["awaiting_refine_pack_selection"] = False
        st.session_state["awaiting_next_options"] = True
        st.session_state["show_next_jd_panel"] = True
        st.stop()

    # Otherwise, treat input as an option
    opt = user_prompt.strip().lower()

    try:
        if opt in ("1", "improve", "packs"):
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                with st.spinner("Preparing three targeted, evidence-backed packs…"):
                    out = suggest_three_packs(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp",""))
                st.markdown(out)
                st.markdown("_Reply with **1,2** or **0** to apply all._")
            st.session_state["awaiting_refine_pack_selection"] = True
            st.session_state["awaiting_next_options"] = False

        elif opt in ("2", "booster", "boosters"):
            st.session_state["custom_refine_text"] = st.session_state.get("custom_refine_text","")
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                with st.spinner("Suggesting minor boosters…"):
                    out = minor_boosters(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp",""), st.session_state["custom_refine_text"])
                st.markdown(out)
            st.session_state["awaiting_next_options"] = True
            st.session_state["show_next_jd_panel"] = True

        elif opt in ("3", "coverage", "board"):
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                with st.spinner("Building Coverage Board…"):
                    out = coverage_board(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp",""))
                st.markdown(out)
            st.session_state["awaiting_next_options"] = True
            st.session_state["show_next_jd_panel"] = True

        elif opt in ("4", "narrative", "presets"):
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                with st.spinner("Showing narrative presets…"):
                    out = narrative_presets(st.session_state.get("last_jd_meta"), st.session_state.get("jd_text_temp",""))
                st.markdown(out)
            st.session_state["awaiting_next_options"] = True
            st.session_state["show_next_jd_panel"] = True

        elif opt in ("5", "ab", "a/b", "bullet"):
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                st.markdown("Paste the **single bullet** you want rewritten (A/B/C).")
            st.session_state["awaiting_ab_bullet"] = True
            st.session_state["awaiting_next_options"] = False

        elif opt in ("6", "level", "calibrator"):
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                st.markdown("Reply with **junior**, **mid**, or **senior** to reframe scope safely.")
            st.session_state["awaiting_level_choice"] = True
            st.session_state["awaiting_next_options"] = False

        elif opt in ("7", "export", "txt", "download"):
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                if st.session_state.get("last_apply_output"):
                    st.download_button(
                        "Download tailored resume + evidence (.txt)",
                        data=st.session_state["last_apply_output"].encode("utf-8"),
                        file_name="readysetrole_results.txt",
                        mime="text/plain",
                        use_container_width=True,
                    )
                else:
                    st.markdown("No recent outputs to export. Run AutoTailor first.")
            st.session_state["awaiting_next_options"] = True
            st.session_state["show_next_jd_panel"] = True

        elif opt in ("0", "next", "next jd"):
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                st.markdown("Upload/paste the **next JD** — I’ll reuse your master resume on file.")
            st.session_state["jd_text_temp"] = ""
            st.session_state["last_jd_meta"] = None
            st.session_state["awaiting_next_options"] = True
            st.session_state["show_next_jd_panel"] = True
            st.rerun()

        else:
            parts = [types.Part.from_text(text=user_prompt)]
            if st.session_state.get("resume_text"):
                parts.append(types.Part.from_text(text="[RESUME TEXT]\n" + st.session_state["resume_text"]))
            if st.session_state.get("resume_meta"):
                parts.append(st.session_state["resume_meta"]["file"])
            if st.session_state.get("jd_text_temp"):
                parts.append(types.Part.from_text(text="[JD TEXT]\n" + st.session_state["jd_text_temp"]))
            if st.session_state.get("last_jd_meta"):
                parts.append(st.session_state["last_jd_meta"]["file"])
            with st.chat_message("assistant", avatar=":material/robot_2:"):
                with st.spinner("Thinking…"):
                    out = send(parts)
                out = out or "I’m ready. Choose an option: 1/2/3/4/5/6/7 or 0 for the next JD."
                st.markdown(out)
            st.session_state["awaiting_next_options"] = True

    except Exception as e:
        st.error(f"❌ Gemini error: {e}")

# ------------- Optional: small refine brief UI when at options -------------
if st.session_state.get("awaiting_next_options"):
    st.markdown("---")
    st.markdown("**Optional: add a custom refine brief** (used by option [2] Minor boosters)")
    st.session_state["custom_refine_text"] = st.text_area(
        "e.g., tighten summary to 45 words, surface 'Figma' in Skills, quantify Mayo with users impacted",
        value=st.session_state.get("custom_refine_text",""),
        height=90
    )

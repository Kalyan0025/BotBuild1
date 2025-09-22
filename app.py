# (same license header...)

import streamlit as st
from PIL import Image
import io, time, mimetypes
from pathlib import Path

from google import genai
from google.genai import types

# ==== ReadySetRole config ====
BOT_NAME = "ReadySetRole"
CREATOR = "Kalyan Kadavanti Sudhakar"
PROMPT_PATH = "identity.txt"             # <- uses your existing file
DEFAULT_MODEL = "gemini-2.5-pro"         # pro tracks longer prompts better
# =============================

st.set_page_config(page_title=BOT_NAME, layout="centered", initial_sidebar_state="expanded")

try:
    st.image(Image.open("Bot.png"), caption=f"{BOT_NAME} by {CREATOR} (2025)", use_container_width=True)
except Exception:
    pass

st.markdown(f"<h1 style='text-align:center;'>{BOT_NAME}</h1>", unsafe_allow_html=True)

# ---------- helpers ----------
def load_developer_prompt() -> str:
    try:
        return Path(PROMPT_PATH).read_text(encoding="utf-8")
    except FileNotFoundError:
        st.warning(f"⚠️ '{PROMPT_PATH}' not found. Using compact fallback.")
        return (
            "<Role>ReadySetRole — resume & cover letter tailoring bot.</Role>\n"
            "<Goal>Deliver ATS-safe tailored resume+letter from Master Resume + JD, with Pre-Score, Packs, Post-Score, Evidence Map.</Goal>\n"
            "<Rules>Be truthful, ATS-safe; never fabricate; show numbered options; return all outputs together.</Rules>"
        )

def human_size(n: int) -> str:
    for unit in ["B","KB","MB","GB","TB"]:
        if n < 1024.0: return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"

def upload_to_files_api(u, client):
    mime = u.type or (mimetypes.guess_type(u.name)[0] or "application/octet-stream")
    data = u.getvalue()
    gfile = client.files.upload(file=io.BytesIO(data), config=types.UploadFileConfig(mime_type=mime))
    return {"name": u.name, "size": len(data), "mime": mime, "file": gfile}

def ensure_files_active(client, files, max_wait_s: float = 12.0):
    deadline = time.time() + max_wait_s
    any_processing = True
    while any_processing and time.time() < deadline:
        any_processing = False
        for i, meta in enumerate(files):
            fobj = meta["file"]
            if getattr(fobj, "state", "") != "ACTIVE":
                any_processing = True
                try:
                    files[i]["file"] = client.files.get(name=fobj.name)
                except Exception:
                    pass
        if any_processing: time.sleep(0.6)

# ---------- Gemini client ----------
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    system_instructions = load_developer_prompt()

    # Optional tool (remove if you don't need search grounding)
    search_tool = types.Tool(google_search=types.GoogleSearch())

    try:
        think_cfg = types.ThinkingConfig(thinking_budget=-1)
    except Exception:
        think_cfg = None

    generation_cfg = types.GenerateContentConfig(
        system_instruction=system_instructions,
        tools=[search_tool],
        thinking_config=think_cfg,
        temperature=0.4,
        max_output_tokens=4096,
        response_mime_type="text/plain",
    )
except Exception as e:
    st.error("Gemini init failed. Check GEMINI_API_KEY in Streamlit → Settings → Secrets.\n\n" + str(e))
    st.stop()

# ---------- state ----------
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("uploaded_files", [])
st.session_state.setdefault("bootstrapped", False)
st.session_state.setdefault("master_resume_text", "")
st.session_state.setdefault("master_resume_file", None)
st.session_state.setdefault("jd_text", "")
st.session_state.setdefault("chat", None)

# ---------- sidebar ----------
with st.sidebar:
    st.title("⚙️ Controls")
    st.caption("Tailor a resume & cover letter to each JD. ATS-safe. No fabrication.")

    with st.expander(":material/text_fields_alt: Model", expanded=True):
        options = ["gemini-2.5-pro","gemini-2.5-flash","gemini-2.5-flash-lite"]
        default_idx = options.index(DEFAULT_MODEL) if DEFAULT_MODEL in options else 0
        selected_model = st.selectbox("Choose a model:", options=options, index=default_idx, help="Pro follows longer prompts best.")
        st.caption(f"Selected: **{selected_model}**")
        if "chat" not in st.session_state or getattr(st.session_state.chat, "model", None) != selected_model:
            st.session_state.chat = client.chats.create(model=selected_model, config=generation_cfg)

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.chat_history.clear()
        st.session_state.chat = client.chats.create(model=selected_model, config=generation_cfg)
        st.toast("Chat cleared.")
        st.rerun()

    with st.expander(":material/badge: Master Resume (remembered)", expanded=True):
        st.caption("Paste Master Resume text OR upload a file once; it will be reused.")
        master_text = st.text_area("Paste Master Resume (optional)", value=st.session_state.master_resume_text, height=160)
        master_file = st.file_uploader("Upload Master Resume file (PDF/TXT/DOCX)", type=["pdf","txt","docx"], accept_multiple_files=False)
        c1,c2 = st.columns(2)
        with c1:
            if st.button("💾 Save Master Resume", use_container_width=True):
                st.session_state.master_resume_text = master_text.strip()
                if master_file is not None:
                    try:
                        meta = upload_to_files_api(master_file, client)
                        st.session_state.master_resume_file = meta
                        st.success(f"Saved file: {meta['name']}")
                    except Exception as e:
                        st.error(f"Upload failed: {e}")
                else:
                    st.session_state.master_resume_file = None
                st.toast("Master Resume remembered.")
        with c2:
            if st.button("🗑 Forget Master Resume", use_container_width=True):
                mf = st.session_state.master_resume_file
                if mf:
                    try:
                        client.files.delete(name=mf["file"].name)
                    except Exception:
                        pass
                st.session_state.master_resume_text = ""
                st.session_state.master_resume_file = None
                st.toast("Forgotten.")
                st.rerun()

    with st.expander(":material/attach_file: Extra Files (optional)", expanded=False):
        uploads = st.file_uploader("Upload files", type=["pdf","txt","docx"], accept_multiple_files=True, label_visibility="collapsed")
        if uploads:
            slots_left = max(0, 5 - len(st.session_state.uploaded_files))
            added = []
            for u in uploads[:slots_left]:
                if any((u.name == f["name"] and u.size == f["size"]) for f in st.session_state.uploaded_files):
                    continue
                try:
                    meta = upload_to_files_api(u, client)
                    st.session_state.uploaded_files.append(meta); added.append(meta["name"])
                except Exception as e:
                    st.error(f"Upload failed for **{u.name}**: {e}")
            if added: st.toast("Uploaded: " + ", ".join(added))
        st.markdown("**Attached files**")
        if st.session_state.uploaded_files:
            for idx, meta in enumerate(st.session_state.uploaded_files):
                left,right = st.columns([0.88,0.12])
                with left:
                    st.write(f"• {meta['name']}  <small>{human_size(meta['size'])} · {meta['mime']}</small>", unsafe_allow_html=True)
                with right:
                    if st.button("✖", key=f"rm_{idx}"):
                        try: client.files.delete(name=meta["file"].name)
                        except Exception: pass
                        st.session_state.uploaded_files.pop(idx); st.rerun()
            st.caption(f"{5 - len(st.session_state.uploaded_files)} slots remaining.")
        else:
            st.caption("No files attached.")

    with st.expander("🛠 Prompt debug", expanded=False):
        st.caption(f"Loaded prompt chars: **{len(system_instructions)}** from **{PROMPT_PATH}**")

# ---------- first assistant message ----------
if not st.session_state.bootstrapped:
    msg = (
        f"Hi! I’m **{BOT_NAME}** by **{CREATOR}**.\n\n"
        "Upload/paste your **Master Resume** once (I’ll remember it), and paste the **Job Description (JD)**. "
        "Click **Tailor Now** to get Pre-Score → Packs → tailored resume + cover letter + evidence."
    )
    st.session_state.chat_history.append({"role":"assistant","parts":msg})
    st.session_state.bootstrapped = True

# ---------- JD input + Tailor Now ----------
with st.container(border=True):
    st.subheader("Start here")
    jd_text = st.text_area("Paste the Job Description (JD):", value=st.session_state.jd_text, height=180, placeholder="Paste the full JD here…")
    c1,c2 = st.columns([0.4,0.6])
    with c1:
        tailor_now = st.button("🚀 Tailor Now", use_container_width=True, type="primary")
    with c2:
        st.caption("I’ll compute a Pre-Score, suggest Packs, and return tailored resume + cover letter with evidence.")

# ---------- replay history ----------
with st.container():
    for msg in st.session_state.chat_history:
        avatar = "👤" if msg["role"]=="user" else ":material/robot_2:"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["parts"])

# ---------- tailor flow ----------
if tailor_now:
    st.session_state.jd_text = jd_text.strip()
    parts = []
    scaffold = (
        "FOLLOW READYSETROLE FLOW STRICTLY.\n"
        "Inputs:\n"
        f"<<MASTER_RESUME_TEXT>>\n{st.session_state.master_resume_text}\n"
        "<<JOB_DESCRIPTION>>\n"
        f"{st.session_state.jd_text}\n"
        "If a Master Resume file is present, read it for evidence too.\n\n"
        "Tasks:\n"
        "1) QuickScore → Pre-Score (overall + sub-scores + top missing keywords + ≤60-word explanation).\n"
        "2) SuggestPacks → grouped keyword Packs with predicted Δ; show numbered options.\n"
        "3) If no user selection yet, propose 0 = Apply All (safe).\n"
        "4) ApplySuggestions + FitToLength → ATS-safe resume (1 page default) + GenerateCoverLetter (~200 words).\n"
        "5) Return Post-Score + Evidence Map + ATS Preview + Change Log + Metric Badges + Confidence Receipt.\n"
        "6) Offer BoostScore (optional, 2–3 taps).\n"
        "Rules: never fabricate; use only resume + JD; unproven → Exposure/Learning; minimal numbered UI."
    )
    parts.append(types.Part.from_text(scaffold))

    if st.session_state.master_resume_file:
        ensure_files_active(client, [st.session_state.master_resume_file])
        parts.append(st.session_state.master_resume_file["file"])

    if st.session_state.uploaded_files:
        ensure_files_active(client, st.session_state.uploaded_files)
        parts.extend([meta["file"] for meta in st.session_state.uploaded_files])

    with st.chat_message("assistant", avatar=":material/robot_2:"):
        try:
            with st.spinner("🔍 Scoring your fit and preparing Packs…"):
                resp = st.session_state.chat.send_message(parts)
            text = resp.text if hasattr(resp, "text") else str(resp)
            st.markdown(text)
            st.session_state.chat_history.append({"role":"assistant","parts":text})
        except Exception as e:
            err = f"❌ Error from Gemini: {e}"
            st.error(err)
            st.session_state.chat_history.append({"role":"assistant","parts":err})

# ---------- chat input (packs selection, etc.) ----------
placeholder = "Reply with options (e.g., 1,3 or 0) or paste a new JD…"
if user_prompt := st.chat_input(placeholder):
    st.session_state.chat_history.append({"role":"user","parts":user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=":material/robot_2:"):
        try:
            ctx = (
                "Context (do not echo):\n"
                f"- Master Resume text length: {len(st.session_state.master_resume_text)}\n"
                f"- JD length: {len(st.session_state.jd_text)}\n"
                "If user reply is numeric like '1,3' or '0', treat as Pack selections. "
                "Apply suggestions → return Post-Score + tailored resume + cover letter + evidence. "
                "Never fabricate; keep ATS-safe; minimal numbered UI."
            )
            parts = [types.Part.from_text(ctx), types.Part.from_text(user_prompt)]

            files_for_turn = []
            if st.session_state.master_resume_file: files_for_turn.append(st.session_state.master_resume_file)
            files_for_turn.extend(st.session_state.uploaded_files)
            if files_for_turn:
                ensure_files_active(client, files_for_turn)
                parts.extend([m["file"] for m in files_for_turn])

            with st.spinner("✍️ Applying your selection…"):
                resp = st.session_state.chat.send_message(parts)
            text = resp.text if hasattr(resp, "text") else str(resp)
            st.markdown(text)
            st.session_state.chat_history.append({"role":"assistant","parts":text})
        except Exception as e:
            err = f"❌ Error from Gemini: {e}"
            st.error(err)
            st.session_state.chat_history.append({"role":"assistant","parts":err})

# ---------- footer ----------
st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px;'>"
    "I can make mistakes—please verify important information."
    "</div>",
    unsafe_allow_html=True,
)

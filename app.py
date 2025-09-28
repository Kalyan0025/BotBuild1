# ReadySetRole — Simple Gemini Chatbot (Streamlit)
# Fixes: repetition, hallucination guard, instruction upload, small+clean UI
# Author credit retained per original template license.
# "This code uses portions of code developed by Ronald A. Beghetto for a course taught at Arizona State University."

import io
import time
import mimetypes
from difflib import SequenceMatcher
from typing import List, Dict, Any

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
    )


def parse_xmlish_instr(txt: str) -> str:
    """Very light parser: extracts content inside <Role>, <Goal>, <Rules>, <Knowledge>, <SpecializedActions>, <Guidelines>.
    Concatenates into a single system instruction string. Safe even if tags are missing.
    """
    import re
    sections = ["Role", "Goal", "Rules", "Knowledge", "SpecializedActions", "Guidelines"]
    chunks = []
    for tag in sections:
        m = re.search(fr"<{tag}>(.*?)</{tag}>", txt, flags=re.DOTALL | re.IGNORECASE)
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


# ---------------------------
# Sidebar Controls
# ---------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.caption("Simple, non-repeating Gemini bot for resume tailoring.")

    st.markdown("### Model")
    model_name = st.selectbox(
        "Choose a model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.5-flash-lite"],
        index=0,
    )

    st.markdown("### Generation Settings")
    temperature = st.slider("temperature", 0.0, 1.0, 0.2, 0.05)
    top_p = st.slider("top_p", 0.0, 1.0, 0.9, 0.05)
    top_k = st.slider("top_k", 1, 100, 40, 1)
    max_tokens = st.number_input("max_output_tokens", min_value=256, max_value=4096, value=1536, step=64)
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

    st.divider()

    # File uploads reused across turns
    st.markdown("### Files (PDF/TXT/DOCX)")
    st.caption("Attach up to 5 files. Uploaded once, reused across turns. Stored temporarily (~48h).")
    uploads = st.file_uploader(
        "Upload files",
        type=["pdf", "txt", "docx"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

# ---------------------------
# Client + Chat setup
# ---------------------------
try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error("Failed to init Gemini client. Set GEMINI_API_KEY in Streamlit secrets.\n" + str(e))
    st.stop()

# Session State
st.session_state.setdefault("chat_history", [])  # our UI echo of messages
st.session_state.setdefault("uploaded_files", [])
st.session_state.setdefault("last_assistant_text", "")

# Upload/track files
if uploads:
    # Limit to 5 unique files by name+size
    slots_left = max(0, 5 - len(st.session_state.uploaded_files))
    for u in uploads[:slots_left]:
        already = any((u.name == f["name"] and u.size == f["size"]) for f in st.session_state.uploaded_files)
        if already:
            continue
        try:
            mime = u.type or (mimetypes.guess_type(u.name)[0] or "application/octet-stream")
            gfile = client.files.upload(file=io.BytesIO(u.getvalue()), config=types.UploadFileConfig(mime_type=mime))
            st.session_state.uploaded_files.append({"name": u.name, "size": u.size, "mime": mime, "file": gfile})
            st.toast(f"Uploaded: {u.name}")
        except Exception as e:
            st.error(f"Upload failed for {u.name}: {e}")

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
    # If model changes, create a fresh chat; otherwise update config in-place
    if getattr(st.session_state.chat, "model", None) != model_name:
        st.session_state.chat = client.chats.create(model=model_name, config=generation_cfg)
    else:
        try:
            st.session_state.chat.update(config=generation_cfg)
        except Exception:
            # some SDK versions may not support update; recreate chat
            st.session_state.chat = client.chats.create(model=model_name, config=generation_cfg)

# Clear chat button
with st.sidebar:
    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.chat_history.clear()
        st.session_state.last_assistant_text = ""
        st.session_state.chat = client.chats.create(model=model_name, config=generation_cfg)
        st.toast("Chat cleared")
        st.rerun()

# Render prior messages
for msg in st.session_state.chat_history:
    avatar = "👤" if msg["role"] == "user" else ":material/robot_2:"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["parts"])  # already plain text

# ---------------------------
# Chat input
# ---------------------------
user_prompt = st.chat_input("Upload your master resume + JD, then ask anything…")

if user_prompt:
    st.session_state.chat_history.append({"role": "user", "parts": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=":material/robot_2:"):
        try:
            contents_to_send = None
            if st.session_state.uploaded_files:
                ensure_active_files(client, st.session_state.uploaded_files)
                contents_to_send = [types.Part.from_text(text=user_prompt)] + [m["file"] for m in st.session_state.uploaded_files]

            with st.spinner("Thinking…"):
                if contents_to_send is None:
                    response = st.session_state.chat.send_message(user_prompt)
                else:
                    response = st.session_state.chat.send_message(contents_to_send)

            full_response = response.text if hasattr(response, "text") else str(response)

            # Repetition guard — if too similar to last assistant text, summarize+ask next step
            if too_similar(st.session_state.last_assistant_text, full_response):
                full_response = (
                    "I've already covered most of that. Would you like me to (1) propose keyword Packs, "
                    "(2) tailor your resume now, or (3) generate a concise cover letter?"
                )

            st.markdown(full_response)
            st.session_state.chat_history.append({"role": "assistant", "parts": full_response})
            st.session_state.last_assistant_text = full_response

        except Exception as e:
            st.error(f"❌ Gemini error: {e}")

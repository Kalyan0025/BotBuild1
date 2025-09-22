# CUSTOM BOT TEMPLATE
# Copyright (c) 2025 Ronald A. Beghetto
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this code and associated files, to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the code, and to permit
# persons to whom the code is furnished to do so, subject to the
# following conditions:
#
# An acknowledgement of the original template author must be made in any use,
# in whole or part, of this code. The following notice shall be included:
# "This code uses portions of code developed by Ronald A. Beghetto for a
# course taught at Arizona State University."
#
# THE CODE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# IMPORT Code Packages
import streamlit as st  # <- streamlit
from PIL import Image   # <- Python code to display images
import io
import time
import mimetypes
import uuid

# --- Google GenAI Models import ---------------------------
from google import genai
from google.genai import types   # <--Allows for tool use
# ----------------------------------------------------

# Streamlit page setup
st.set_page_config(
    page_title="My Bot",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Load and display a custom image for your bot
try:
    st.image(
        Image.open("Bot.png"),
        caption="Bot Created by YOUR NAME (2025)",
        use_container_width=True
    )
except Exception as e:
    st.error(f"Error loading image: {e}")

# Bot Title
st.markdown("<h1 style='text-align: center;'>YOUR BOT'S NAME</h1>", unsafe_allow_html=True)

# --- Helper -----------------------------------------
def load_developer_prompt() -> str:
    try:
        with open("identity.txt") as f:  # rename here if your file is "instruction.txt"
            return f.read()
    except FileNotFoundError:
        st.warning("⚠️ 'identity.txt' not found. Using default prompt.")
        return ("You are a helpful assistant. "
                "Be friendly, engaging, and provide clear, concise responses.")

def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"

# --- Gemini configuration (hardened) ---------------------------
api_key = st.secrets.get("GEMINI_API_KEY")
if not api_key:
    st.error("Missing GEMINI_API_KEY in Streamlit secrets. Add it in Settings → Secrets.")
    st.stop()

try:
    # Activate Gemini GenAI client
    client = genai.Client(api_key=api_key)

    # System instructions
    system_instructions = load_developer_prompt()

    # Disable optional search tool to reduce variability
    tools = None  # keep None for simplest, most compatible config

    # Avoid unstable thinking_config; use conservative settings
    generation_cfg = types.GenerateContentConfig(
        system_instruction=system_instructions,
        tools=tools,
        temperature=0.7,
        max_output_tokens=2048,
    )

except Exception as e:
    st.error(
        "Error initialising the Gemini client. "
        "Check your GEMINI_API_KEY and package versions."
        f" Details: {e}"
    )
    st.stop()

# Ensure chat history and files state stores exist
st.session_state.setdefault("chat_history", [])
# Each entry: {"name": str, "size": int, "mime": str, "file": google.genai.types.File}
st.session_state.setdefault("uploaded_files", [])
# Idempotency guards for repeated replies on reruns
st.session_state.setdefault("last_processed_key", None)
st.session_state.setdefault("pending_prompt", None)
st.session_state.setdefault("turn_key", None)

# --- Sidebar ----------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("### About: Briefly describe your bot here for users.")

    # Model Selection Expander
    with st.expander(":material/text_fields_alt: Model Selection", expanded=True):
        selected_model = st.selectbox(
            "Choose a model:",
            options=[
                "gemini-2.5-flash",
                "gemini-2.5-pro",
                "gemini-2.5-flash-lite"
            ],
            index=0,
            label_visibility="visible",
            help="Response Per Day Limits: Pro = 100, Flash = 250, Flash-lite = 1000)"
        )
        st.caption(f"Selected: **{selected_model}**")

        def _create_chat(model_name: str):
            try:
                return client.chats.create(model=model_name, config=generation_cfg)
            except Exception as e:
                st.error(f"Failed to create chat for model '{model_name}': {e}")
                st.stop()

        # Create chat now (post-selection), or re-create if the model changed
        if "chat" not in st.session_state:
            st.session_state.chat = _create_chat(selected_model)
        elif getattr(st.session_state.chat, "model", None) != selected_model:
            st.session_state.chat = _create_chat(selected_model)

    # ---- Clear Chat button ----
    if st.button("🧹 Clear chat", use_container_width=True, help="Clear messages and reset chat context"):
        st.session_state.chat_history.clear()
        st.session_state.pending_prompt = None
        st.session_state.turn_key = None
        st.session_state.last_processed_key = None
        st.session_state.chat = client.chats.create(model=selected_model, config=generation_cfg)
        st.toast("Chat cleared.")
        st.rerun()

    # ---- File Upload (Files API) ----
    with st.expander(":material/attach_file: Files (PDF/TXT/DOCX)", expanded=True):
        st.caption(
            "Attach up to **5** files. They’ll be uploaded once and reused across turns. "
            "Files are stored temporarily (≈48 hours) in Google’s File store and count toward "
            "your 20 GB storage cap until deleted (clicking ✖) or expired."
        )
        uploads = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        # Helper: Upload one file to Gemini Files API
        def _upload_to_gemini(u):
            mime = u.type or (mimetypes.guess_type(u.name)[0] or "application/octet-stream")
            data = u.getvalue()
            gfile = client.files.upload(
                file=io.BytesIO(data),
                config=types.UploadFileConfig(mime_type=mime)
            )
            return {
                "name": u.name,
                "size": len(data),
                "mime": mime,
                "file": gfile,
            }

        # Add newly selected files (respect cap of 5)
        if uploads:
            slots_left = max(0, 5 - len(st.session_state.uploaded_files))
            newly_added = []
            for u in uploads[:slots_left]:
                already = any((u.name == f["name"] and u.size == f["size"]) for f in st.session_state.uploaded_files)
                if already:
                    continue
                try:
                    meta = _upload_to_gemini(u)
                    st.session_state.uploaded_files.append(meta)
                    newly_added.append(meta["name"])
                except Exception as e:
                    st.error(f"File upload failed for **{u.name}**: {e}")
            if newly_added:
                st.toast(f"Uploaded: {', '.join(newly_added)}")

        # Show current file list with remove buttons
        st.markdown("**Attached files**")
        if st.session_state.uploaded_files:
            for idx, meta in enumerate(st.session_state.uploaded_files):
                left, right = st.columns([0.88, 0.12])
                with left:
                    st.write(
                        f"• {meta['name']}"
                        f"<small>{human_size(meta['size'])} · {meta['mime']}</small>",
                        unsafe_allow_html=True
                    )
                with right:
                    if st.button("✖", key=f"remove_{idx}", help="Remove this file"):
                        try:
                            client.files.delete(name=meta['file'].name)
                        except Exception:
                            pass
                        st.session_state.uploaded_files.pop(idx)
                        st.rerun()
            st.caption(f"{5 - len(st.session_state.uploaded_files)} slots remaining.")
        else:
            st.caption("No files attached.")

    # Show Stored files on Google (server side)
    with st.expander("🛠️ Developer: See and Delete all files stored on Google server", expanded=False):
        try:
            files_list = client.files.list()
            files_iter = files_list if isinstance(files_list, list) else list(files_list) if files_list else []
            if not files_iter:
                st.caption("No active files on server.")
            else:
                for f in files_iter:
                    exp = getattr(f, "expiration_time", None)
                    size = getattr(f, "size_bytes", None)
                    size_str = f"{(size or 0)/1024:.1f} KB"
                    st.write(
                        f"• **{getattr(f, 'name', '?')}**  "
                        f"({getattr(f, 'mime_type', '?')}, {size_str})  "
                        f"Expires: {exp or '?'}"
                    )
                if st.button("🗑️ Delete all files", use_container_width=True):
                    failed = []
                    for f in files_iter:
                        try:
                            client.files.delete(name=getattr(f, "name", None))
                        except Exception:
                            failed.append(getattr(f, "name", "?"))
                    if failed:
                        st.error(f"Failed to delete: {', '.join(failed)}")
                    else:
                        st.success("All files deleted from server.")
                        st.rerun()
        except Exception as e:
            st.error(f"Could not fetch files list: {e}")

#######################################
# Enable chat container and chat set-up
#######################################
with st.container():
    # Replay chat history
    for msg in st.session_state.chat_history:
        avatar = "👤" if msg["role"] == "user" else ":material/robot_2:"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["parts"])

def _ensure_files_active(files, max_wait_s: float = 6.0):
    """Poll the Files API for PROCESSING files until ACTIVE or timeout."""
    deadline = time.time() + max_wait_s
    while time.time() < deadline:
        any_processing = False
        for i, meta in enumerate(files):
            fobj = meta.get("file")
            if not fobj:
                continue
            state = getattr(fobj, "state", None)
            if state not in ("ACTIVE", "active", "READY"):
                any_processing = True
                try:
                    name = getattr(fobj, "name", None) or getattr(fobj, "id", None)
                    if name:
                        files[i]["file"] = client.files.get(name=name)
                except Exception:
                    pass
        if not any_processing:
            break
        time.sleep(0.6)

# Capture input without immediately processing (prevents repeated replies on reruns)
if user_prompt := st.chat_input("Message 'your bot name'…"):
    st.session_state.turn_key = str(uuid.uuid4())
    st.session_state.pending_prompt = user_prompt

# Process exactly once per unique turn_key
if st.session_state.get("pending_prompt") and st.session_state.get("turn_key") != st.session_state.get("last_processed_key"):
    user_prompt = st.session_state.pending_prompt
    st.session_state.chat_history.append({"role": "user", "parts": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=":material/robot_2:"):
        try:
            contents_to_send = None
            if st.session_state.uploaded_files:
                _ensure_files_active(st.session_state.uploaded_files, max_wait_s=6.0)
                contents_to_send = [types.Part.from_text(text=user_prompt)]
                contents_to_send += [
                    meta["file"] for meta in st.session_state.uploaded_files if meta.get("file")
                ]

            with st.spinner("🔍 Thinking about what I know about this ..."):
                try:
                    if contents_to_send is None:
                        response = st.session_state.chat.send_message(user_prompt)
                    else:
                        response = st.session_state.chat.send_message(contents_to_send)
                except Exception as e:
                    st.warning(f"Retrying without files due to: {e}")
                    response = st.session_state.chat.send_message(user_prompt)

            full_response = response.text if hasattr(response, "text") else str(response)
            st.markdown(full_response)

        except Exception as e:
            full_response = f"❌ Error from Gemini: {e}"
            st.error(full_response)

    # Record assistant reply and mark this turn processed
    st.session_state.chat_history.append({"role": "assistant", "parts": full_response})
    st.session_state.last_processed_key = st.session_state.turn_key
    st.session_state.pending_prompt = None

# Footer
st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px;'>"
    "I can make mistakes—please verify important information."
    "</div>",
    unsafe_allow_html=True,
)

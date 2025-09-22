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

# --- Google GenAI Models import ---------------------------
from google import genai
from google.genai import types   # <--Allows for tool use, like Google Search
# ----------------------------------------------------

# ===== ReadySetRole config (edit these) ==================
BOT_NAME = "ReadySetRole"
CREATOR = "Kalyan Kadavanti Sudhakar"
PROMPT_PATH = "identity.txt"            # keep using your existing file
DEFAULT_MODEL = "gemini-2.5-pro"        # pro follows longer prompts best
# ========================================================

# Streamlit page setup <--this should be the first streamlit command after imports
st.set_page_config(page_title=BOT_NAME,  # name on the browser tab
                   layout="centered",
                   initial_sidebar_state="expanded")

# Load and display a custom image for your bot
try:
    st.image(Image.open("Bot.png"),
             caption=f"{BOT_NAME} by {CREATOR} (2025)",
             use_container_width=True)
except Exception as e:
    st.error(f"Error loading image: {e}")

# Bot Title
st.markdown(f"<h1 style='text-align: center;'>{BOT_NAME}</h1>", unsafe_allow_html=True)

# --- Helper -----------------------------------------
def load_developer_prompt() -> str:
    try:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        st.warning(f"⚠️ '{PROMPT_PATH}' not found. Using default prompt.")
        return (f"You are a helpful assistant named {BOT_NAME}, created by {CREATOR}. "
                "Be friendly, engaging, and provide clear, concise responses.")

def human_size(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"

# --- Gemini configuration ---------------------------
try:
    # Activate Gemini GenAI model and access your API key in streamlit secrets
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])  # make sure GEMINI_API_KEY is set in Streamlit secrets

    # System instructions
    system_instructions = load_developer_prompt()

    # Enable Google Search Tool (optional)
    search_tool = types.Tool(google_search=types.GoogleSearch())

    # Generation configuration for every turn
    generation_cfg = types.GenerateContentConfig(
        system_instruction=system_instructions,
        tools=[search_tool],
        thinking_config=types.ThinkingConfig(thinking_budget=-1),  # dynamic thinking
        temperature=1.0,
        max_output_tokens=2048,
    )

except Exception as e:
    st.error(
        "Error initialising the Gemini client. "
        "Check your `GEMINI_API_KEY` in Streamlit → Settings → Secrets."
        f" Details: {e}"
    )
    st.stop()

# Ensure chat history and files state stores exist
st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("uploaded_files", [])
st.session_state.setdefault("bootstrapped", False)  # <- to show the first assistant message once

# ---------- First assistant message (bot makes the first move) ----------
if not st.session_state.bootstrapped:
    first_msg = (
        f"Hi! I’m **{BOT_NAME}** by **{CREATOR}**.\n\n"
        "Upload/paste your **Master Resume** (I’ll remember it in this session), "
        "and paste the **Job Description (JD)**. Then send both here to get "
        "**Pre-Score → tailored resume + cover letter → Post-Score**."
    )
    st.session_state.chat_history.append({"role": "assistant", "parts": first_msg})
    st.session_state.bootstrapped = True

# --- Sidebar ----------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("**About:** Tailor a resume & cover letter to each JD. ATS-safe. No fabrication.")

    # Model Selection Expander (testing different models)
    with st.expander(":material/text_fields_alt: Model Selection", expanded=True):
        selected_model = st.selectbox(
            "Choose a model:",
            options=[
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite"
            ],
            index=0,  # Default to gemini-2.5-pro
            label_visibility="visible",
            help="Pro follows longer prompts best; Flash/Flash-lite are faster."
        )
        st.caption(f"Selected: **{selected_model}**")

        # Create chat now (post-selection), or re-create if the model changed
        if "chat" not in st.session_state:
            st.session_state.chat = client.chats.create(model=selected_model, config=generation_cfg)
        elif getattr(st.session_state.chat, "model", None) != selected_model:
            st.session_state.chat = client.chats.create(model=selected_model, config=generation_cfg)

    # ---- Clear Chat button ----
    if st.button("🧹 Clear chat", use_container_width=True, help="Clear messages and reset chat context"):
        st.session_state.chat_history.clear()
        st.session_state.bootstrapped = False
        # Recreate a fresh chat session (resets server-side history)
        st.session_state.chat = client.chats.create(model=selected_model, config=generation_cfg)
        st.toast("Chat cleared.")
        st.rerun()

    # ---- File Upload (Files API) ----
    with st.expander(":material/attach_file: Files (PDF/TXT/DOCX)", expanded=True):
        st.caption(
            "Attach up to **5** files. They’ll be uploaded once and reused across turns. "
            "Files are stored temporarily (≈48 hours) in Google’s File store."
        )
        uploads = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

        # Helper: Upload one file to Gemini Files API
        def _upload_to_gemini(u):
            # Infer MIME type
            mime = u.type or (mimetypes.guess_type(u.name)[0] or "application/octet-stream")
            data = u.getvalue()
            # Upload with bytes buffer; SDK infers metadata, we provide mime
            gfile = client.files.upload(
                file=io.BytesIO(data),
                config=types.UploadFileConfig(mime_type=mime)
            )
            # Persist minimal metadata (avoid keeping the raw bytes in memory)
            return {
                "name": u.name,
                "size": len(data),
                "mime": mime,
                "file": gfile,          # has .name, .uri, .mime_type, .state, .expiration_time
            }

        # Add newly selected files (respect cap of 5)
        if uploads:
            slots_left = max(0, 5 - len(st.session_state.uploaded_files))
            newly_added = []
            for u in uploads[:slots_left]:
                # Skip duplicates by (name, size)
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
                        f"• {meta['name']} "
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

    #show Stored files on Google (server side) --
    with st.expander("🛠️ Developer: See and Delete all files stored on Google server", expanded=False):
        try:
            files_list = client.files.list()
            if not files_list:
                st.caption("No active files on server.")
            else:
                for f in files_list:
                    exp = getattr(f, "expiration_time", None)
                    exp_str = exp if exp else "?"
                    size = getattr(f, "size_bytes", None)
                    size_str = f"{size/1024:.1f} KB" if size else "?"
                    st.write(
                        f"• **{f.name}**  "
                        f"({f.mime_type}, {size_str})  "
                        f"Expires: {exp_str}"
                    )
                if st.button("🗑️ Delete all files", use_container_width=True):
                    failed = []
                    for f in files_list:
                        try:
                            client.files.delete(name=f.name)
                        except Exception as e:
                            failed.append(f.name)
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
        avatar = "👤" if msg["role"] == "user" else ":material/robot_2:"  # <-- These emoji's can be changed
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["parts"])

def _ensure_files_active(files, max_wait_s: float = 12.0):
    """Poll the Files API for PROCESSING files until ACTIVE or timeout."""
    deadline = time.time() + max_wait_s
    any_processing = True
    while any_processing and time.time() < deadline:
        any_processing = False
        for i, meta in enumerate(files):
            fobj = meta["file"]
            if getattr(fobj, "state", "") not in ("ACTIVE",):
                any_processing = True
                try:
                    updated = client.files.get(name=fobj.name)
                    files[i]["file"] = updated
                except Exception:
                    pass
        if any_processing:
            time.sleep(0.6)

# ---------- Chat input ----------
placeholder = "Paste the JD and (optionally) attach your Master Resume to tailor now"
if user_prompt := st.chat_input(placeholder):
    # Record & show user message
    st.session_state.chat_history.append({"role": "user", "parts": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    # Send message and display full response (no streaming)
    with st.chat_message("assistant", avatar=":material/robot_2:"):
        try:
            # If files are attached, ensure they're ready and include them in this turn
            if st.session_state.uploaded_files:
                _ensure_files_active(st.session_state.uploaded_files)
                # IMPORTANT: send plain strings + file objects (no Part.from_text)
                contents_to_send = [user_prompt] + [meta["file"] for meta in st.session_state.uploaded_files]
                response = st.session_state.chat.send_message(contents_to_send)
            else:
                response = st.session_state.chat.send_message(user_prompt)

            # Extract the full response text
            full_response = response.text if hasattr(response, "text") else str(response)

            # Display the full response
            st.markdown(full_response)

        except Exception as e:
            full_response = f"❌ Error from Gemini: {e}"
            st.error(full_response)

        # Record assistant reply
        st.session_state.chat_history.append({"role": "assistant", "parts": full_response})

# Footer
st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px;'>"
    "I can make mistakes—please verify important information."
    "</div>",
    unsafe_allow_html=True,
)

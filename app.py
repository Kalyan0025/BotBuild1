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

# Streamlit page setup <--this should be the first streamlit command after imports
st.set_page_config(page_title="My Bot",  # <-- Change this also but always keep " " this will be the name on the browser tag
                   layout="centered",    # <--- options are "centered", "wide", or nothing for default
                   initial_sidebar_state="expanded")  # <-- will expand the sidebar automatically

# Load and display a custom image for your bot
try:
    st.image(Image.open("Bot.png"),  # <-- make sure your image is called this or change it to be the same
             caption="Bot Created by YOUR NAME (2025)",  # <-- change with your bot name and your own name
             use_container_width=True)
except Exception as e:
    st.error(f"Error loading image: {e}")

# Bot Title
st.markdown("<h1 style='text-align: center;'>YOUR BOT'S NAME</h1>", unsafe_allow_html=True)

# --- Helper -----------------------------------------
def load_developer_prompt() -> str:
    try:
        with open("identity.txt") as f:  # <-- Make sure your rules.text name matches this exactly
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

# --- Gemini configuration ---------------------------
try:
    # Activate Gemini GenAI model and access your API key in streamlit secrets
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])  # <-- make sure you have your google API key (from Google AI Studio) and put it in streamlit secrets as GEMINI_API_KEY = "yourapikey" use " "

    # System instructions
    system_instructions = load_developer_prompt()

    # Enable Google Search Tool
    search_tool = types.Tool(google_search=types.GoogleSearch())  # <-- optional Google Search tool

    # Generation configuration for every turn
    generation_cfg = types.GenerateContentConfig(
        system_instruction=system_instructions,
        tools=[search_tool],
        thinking_config=types.ThinkingConfig(thinking_budget=-1), # <--- set to dynamic thinking (model decides whether to use thinking based on context)
        temperature=1.0,
        max_output_tokens=2048,
    )
    
except Exception as e:
    st.error(
        "Error initialising the Gemini client. "
        "Check your `GEMINI_API_KEY` in Streamlit → Settings → Secrets."
        f"Details: {e}"
    )
    st.stop()

# Ensure chat history and files state stores exist
st.session_state.setdefault("chat_history", [])
# Each entry: {"name": str, "size": int, "mime": str, "file": google.genai.types.File}
st.session_state.setdefault("uploaded_files", [])

# --- Sidebar ----------------------------------------
with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("### About: Briefly describe your bot here for users.")

    # Model Selection Expander (testing different models)
    with st.expander(":material/text_fields_alt: Model Selection", expanded=True):
        selected_model = st.selectbox(
            "Choose a model:",
            options=[
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite"
            ],
            index=2,  # Default to gemini-2.5-flash-lite
            label_visibility="visible",
            help="Response Per Day Limits: Pro = 100, Flash = 250, Flash-lite = 1000)"
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
        # Recreate a fresh chat session (resets server-side history)
        st.session_state.chat = client.chats.create(model=selected_model, config=generation_cfg)
        st.toast("Chat cleared.")
        st.rerun()

    # ---- File Upload (Files API) ----
    with st.expander(":material/attach_file: Files (PDF/TXT/DOCX)", expanded=True):
        st.caption(
            "Attach up to **5** files. They’ll be uploaded once and reused across turns.  "
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
          
if user_prompt := st.chat_input("Message 'your bot name'…"):
    # Record & show user message
    st.session_state.chat_history.append({"role": "user", "parts": user_prompt})
    with st.chat_message("user", avatar="👤"):  # <-- This emoji can be changed
        st.markdown(user_prompt)

    # Send message and display full response (no streaming)
    with st.chat_message("assistant", avatar=":material/robot_2:"):  # <-- This bot image can be replaced with an emoji
        try:
            # If files are attached, ensure they're ready and include them in this turn
            contents_to_send = None
            if st.session_state.uploaded_files:
                _ensure_files_active(st.session_state.uploaded_files)
                contents_to_send = [
                    types.Part.from_text(text=user_prompt)
                ] + [meta["file"] for meta in st.session_state.uploaded_files]

            # Show spinner with message
            with st.spinner("🔍 Thinking about what I know about this ..."):
                if contents_to_send is None:
                    # No files attached: keep original behavior
                    response = st.session_state.chat.send_message(user_prompt)
                else:
                    # Files attached: pass a parts list (text + File objects)
                    response = st.session_state.chat.send_message(contents_to_send)

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

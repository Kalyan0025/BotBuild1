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
import streamlit as st
from PIL import Image
import io
import time
import mimetypes

# --- Google GenAI Models import ---------------------------
from google import genai
from google.genai import types # <--Allows for tool use, like Google Search
# ----------------------------------------------------

# Streamlit page setup <--this should be the first streamlit command after imports
st.set_page_config(page_title="ReadySetRole", # <-- Change this also but always keep " " this will be the name on the browser tag
                    layout="centered",   # <--- options are "centered", "wide", or nothing for default
                    initial_sidebar_state="expanded") # <-- will expand the sidebar automatically

# --- Centered Logo and Title Section ---
# Create three columns to center the content
left_col, center_col, right_col = st.columns([1, 2, 1])

with center_col:
    try:
        st.image(Image.open("Bot1.png"), width=100)
    except Exception as e:
        st.error(f"Error loading image: {e}")
    
    st.markdown("<p style='text-align: center; margin: 0; font-size: 14px;'>Bot Created by GommaBelt</p>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-top: 0;'>Ready Set Role</h2>", unsafe_allow_html=True)
# ----------------------------------------


# --- Helper -----------------------------------------
def load_developer_prompt() -> str:
    try:
        with open("identity.txt") as f: # <-- Make sure your rules.text name matches this exactly
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
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"]) # <-- make sure you have your google API key (from Google AI Studio) and put it in streamlit secrets as GEMINI_API_KEY = "yourapikey" use " "

    # System instructions
    system_instructions = load_developer_prompt()

    # Enable Google Search Tool
    search_tool = types.Tool(google_search=types.GoogleSearch()) # <-- optional Google Search tool

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
st.session_state.setdefault("uploaded_files", [])
st.session_state.setdefault("master_resume_uploaded", False)
st.session_state.setdefault("job_description_pasted", "")
st.session_state.setdefault("initial_prompt_sent", False)

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
            index=2, # Default to gemini-2.5-flash-lite
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
        st.session_state.uploaded_files.clear()
        st.session_state.master_resume_uploaded = False
        st.session_state.job_description_pasted = ""
        st.session_state.initial_prompt_sent = False
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
                st.session_state.master_resume_uploaded = True
                st.toast(f"Uploaded: {', '.join(newly_added)}")
                st.rerun()
            
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
                        if not st.session_state.uploaded_files:
                            st.session_state.master_resume_uploaded = False
                        st.rerun()
            st.caption(f"{5 - len(st.session_state.uploaded_files)} slots remaining.")
        else:
            st.caption("No files attached.")

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
                    st.write(f"• **{f.name}** ({f.mime_type}, {size_str}) Expires: {exp_str}")
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
# Main chat container and chat set-up
#######################################
with st.container():
    # Replay chat history
    for msg in st.session_state.chat_history:
        avatar = "👤" if msg["role"] == "user" else ":material/robot_2:"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["parts"])

    # File and JD input for first-time use
    if not st.session_state.master_resume_uploaded or not st.session_state.job_description_pasted:
        if not st.session_state.master_resume_uploaded:
            st.info("Please upload your master resume to get started.")
        
        # Display the text area for JD input only if a resume is uploaded
        if st.session_state.master_resume_uploaded and not st.session_state.job_description_pasted:
            jd_input = st.text_area("Paste your Job Description (JD) here:", key="jd_input", height=300)
            
            if st.button("Submit JD"):
                if jd_input:
                    st.session_state.job_description_pasted = jd_input
                    user_initial_prompt = f"Here is my master resume: [attached files]. Here is the job description: {st.session_state.job_description_pasted}"
                    st.session_state.chat_history.append({"role": "user", "parts": user_initial_prompt})
                    st.rerun()
                else:
                    st.warning("Please paste the job description before submitting.")

    # Handle chat input for subsequent interactions
    if st.session_state.master_resume_uploaded and st.session_state.job_description_pasted:
        if user_prompt := st.chat_input("Ask for a score, changes, or more boosters:"):
            st.session_state.chat_history.append({"role": "user", "parts": user_prompt})
            with st.chat_message("user", avatar="👤"):
                st.markdown(user_prompt)

            with st.chat_message("assistant", avatar=":material/robot_2:"):
                try:
                    contents_to_send = [types.Part.from_text(text=user_prompt)]
                    if st.session_state.uploaded_files:
                        _ensure_files_active(st.session_state.uploaded_files)
                        contents_to_send += [meta["file"] for meta in st.session_state.uploaded_files]
                    if st.session_state.job_description_pasted:
                        contents_to_send.append(types.Part.from_text(text=f"Job Description: {st.session_state.job_description_pasted}"))

                    with st.spinner("🔍 Thinking..."):
                        response = st.session_state.chat.send_message(contents_to_send)
                    
                    full_response = response.text if hasattr(response, "text") else str(response)
                    st.markdown(full_response)
                except Exception as e:
                    full_response = f"❌ Error from Gemini: {e}"
                    st.error(full_response)
                
                st.session_state.chat_history.append({"role": "assistant", "parts": full_response})

# Footer
st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px;'>"
    "I can make mistakes—please verify important information."
    "</div>",
    unsafe_allow_html=True,
)

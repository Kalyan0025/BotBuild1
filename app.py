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
# THE USE OR OTHER DEALINGS IN THE SOFTWARE.

import streamlit as st
from PIL import Image
import io
import time
import mimetypes

from google import genai
from google.genai import types

st.set_page_config(
    page_title="My Bot",
    layout="centered",
    initial_sidebar_state="expanded"
)

try:
    st.image(
        Image.open("Bot1.png"),
        caption="Bot Created by YOUR NAME (2025)",
        use_container_width=True
    )
except Exception as e:
    st.error(f"Error loading image: {e}")

st.markdown("<h1 style='text-align: center;'>Ready Set Role</h1>", unsafe_allow_html=True)

def load_developer_prompt() -> str:
    try:
        with open("identity.txt") as f:
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

try:
    client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    system_instructions = load_developer_prompt()
    search_tool = types.Tool(google_search=types.GoogleSearch())

    # Removed thinking_config to avoid validation error
    generation_cfg = types.GenerateContentConfig(
        system_instruction=system_instructions,
        tools=[search_tool],
        temperature=1.0,
        max_output_tokens=2048,
    )

except Exception as e:
    st.error(
        "Error initializing the Gemini client. "
        "Check your `GEMINI_API_KEY` in Streamlit → Settings → Secrets. "
        f"Details: {e}"
    )
    st.stop()

st.session_state.setdefault("chat_history", [])
st.session_state.setdefault("uploaded_files", [])

with st.sidebar:
    st.title("⚙️ Controls")
    st.markdown("### About: Briefly describe your bot here for users.")

    with st.expander(":material/text_fields_alt: Model Selection", expanded=True):
        selected_model = st.selectbox(
            "Choose a model:",
            options=[
                "gemini-2.5-pro",
                "gemini-2.5-flash",
                "gemini-2.5-flash-lite"
            ],
            index=2,
            label_visibility="visible",
            help="Response Per Day Limits: Pro = 100, Flash = 250, Flash-lite = 1000"
        )
        st.caption(f"Selected: **{selected_model}**")

        if "chat" not in st.session_state:
            st.session_state.chat = client.chats.create(model=selected_model, config=generation_cfg)
        elif getattr(st.session_state.chat, "model", None) != selected_model:
            st.session_state.chat = client.chats.create(model=selected_model, config=generation_cfg)

    if st.button("🧹 Clear chat", use_container_width=True):
        st.session_state.chat_history.clear()
        st.session_state.chat = client.chats.create(model=selected_model, config=generation_cfg)
        st.toast("Chat cleared.")
        st.rerun()

    with st.expander(":material/attach_file: Files (PDF/TXT/DOCX)", expanded=True):
        st.caption(
            "Attach up to **5** files. They’ll be uploaded once and reused across turns. "
            "Files are stored temporarily (~48 hrs) and count toward your 20GB cap."
        )
        uploads = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
            label_visibility="collapsed"
        )

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
                st.toast(f"Uploaded: {', '.join(newly_added)}")

        st.markdown("**Attached files**")
        if st.session_state.uploaded_files:
            for idx, meta in enumerate(st.session_state.uploaded_files):
                left, right = st.columns([0.88, 0.12])
                with left:
                    st.write(
                        f"• {meta['name']} <small>{human_size(meta['size'])} · {meta['mime']}</small>",
                        unsafe_allow_html=True
                    )
                with right:
                    if st.button("✖", key=f"remove_{idx}"):
                        try:
                            client.files.delete(name=meta['file'].name)
                        except Exception:
                            pass
                        st.session_state.uploaded_files.pop(idx)
                        st.rerun()
            st.caption(f"{5 - len(st.session_state.uploaded_files)} slots remaining.")
        else:
            st.caption("No files attached.")

    with st.expander("🛠️ Developer: See and Delete all files on Google", expanded=False):
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
                        st.success("All files deleted.")
                        st.rerun()
        except Exception as e:
            st.error(f"Could not fetch files list: {e}")

with st.container():
    for msg in st.session_state.chat_history:
        avatar = "👤" if msg["role"] == "user" else ":material/robot_2:"
        with st.chat_message(msg["role"], avatar=avatar):
            st.markdown(msg["parts"])

def _ensure_files_active(files, max_wait_s: float = 12.0):
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
    st.session_state.chat_history.append({"role": "user", "parts": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)

    with st.chat_message("assistant", avatar=":material/robot_2:"):
        try:
            contents_to_send = None
            if st.session_state.uploaded_files:
                _ensure_files_active(st.session_state.uploaded_files)
                contents_to_send = [
                    types.Part.from_text(text=user_prompt)
                ] + [meta["file"] for meta in st.session_state.uploaded_files]

            with st.spinner("🔍 Thinking about what I know about this ..."):
                if contents_to_send is None:
                    response = st.session_state.chat.send_message(user_prompt)
                else:
                    response = st.session_state.chat.send_message(contents_to_send)

            full_response = response.text if hasattr(response, "text") else str(response)
            st.markdown(full_response)
        except Exception as e:
            st.error(f"❌ Error from Gemini: {e}")

        st.session_state.chat_history.append({"role": "assistant", "parts": full_response})

st.markdown(
    "<div style='text-align:center;color:gray;font-size:12px;'>"
    "I can make mistakes—please verify important information."
    "</div>",
    unsafe_allow_html=True,
)

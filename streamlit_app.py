#streamlit run streamlit_app.py

import os
import time
import traceback
from typing import List, Dict, Any, Optional

import streamlit as st
from dotenv import load_dotenv

# Reuse RAG logic by importing functions from Rag_final.py
# (Assumes Rag_final.py is in same directory.)
from Rag_final import (
    init_vector_store,
    answer_query,
    load_generation_pipeline,
    hf_model_name,
    transcribe_audio,
)
# Optional multilingual helpers (Sarvam AI)
try:
    from Rag_final import sarvam_translate, detect_language_simple  # type: ignore
    _SARVAM_HELPERS = True
except Exception:
    _SARVAM_HELPERS = False

# Optional mic recorder component
try:
    from streamlit_mic_recorder import mic_recorder  # type: ignore
    _MIC_AVAILABLE = True
except Exception:
    _MIC_AVAILABLE = False

load_dotenv()

st.set_page_config(
    page_title="Medical Data Analysis",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------- ChatGPT-Style Theme (CSS) --------------

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* === ChatGPT Color Palette === */
    :root {
        --bg-main: #0d0d0d;
        --bg-sidebar: #171717;
        --surface: #1e1e1e;
        --surface-hover: #292929;
        --border: #2f2f2f;
        --text-primary: #ececec;
        --text-secondary: #adadad;
        --text-muted: #767676;
        --accent-user: #2c5fee;
        --accent-assistant: #2f2f2f;
        --accent-hover: #3d3d3d;
        --radius-sm: 8px;
        --radius-md: 12px;
        --radius-lg: 20px;
        --radius-full: 999px;
        --shadow: 0 2px 8px rgba(0,0,0,0.3);
    }
    
    /* === Base Layout === */
    html, body, [data-baseweb="baseweb"], .stApp {
        background: var(--bg-main);
        color: var(--text-primary);
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        font-size: 15px;
        line-height: 1.6;
    }
    
    /* Hide default header and adjust sidebar */
    header[data-testid="stHeader"] {
        background: transparent;
        border-bottom: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] > div {
        background: var(--bg-sidebar);
        border-right: 1px solid var(--border);
    }
    
    /* === Centered Chat Container === */
    .main .block-container {
        max-width: 48rem;
        padding-left: 1rem;
        padding-right: 1rem;
        margin: 0 auto;
        padding-bottom: 100px;
    }
    
    /* === Message Bubbles === */
    .chat-messages {
        padding-bottom: 20px;
        display: flex;
        flex-direction: column;
        gap: 1.5rem;
    }
    
    .message-row {
        display: flex;
        gap: 1rem;
        padding: 1rem 0;
    }
    .message-row.user {
        background: var(--bg-main);
    }
    .message-row.assistant {
        background: var(--surface);
    }
    .message-avatar {
        width: 32px;
        height: 32px;
        border-radius: var(--radius-full);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 1.125rem;
        flex-shrink: 0;
        background: var(--surface-hover);
    }
    .message-row.user .message-avatar {
        background: var(--accent-user);
        color: white;
    }
    .message-content {
        flex: 1;
        min-width: 0;
        padding-top: 4px;
    }
    .message-content p {
        margin: 0;
        color: var(--text-primary);
    }
    
    /* Override Streamlit chat_message defaults */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    
    /* === Bottom Composer === */
    .composer-container {
        position: fixed;
        bottom: 0;
        left: 320px;
        right: 0;
        padding: 0.75rem 1rem;
        background: var(--bg-main);
        border-top: 1px solid var(--border);
        z-index: 200;
    }
    @media (max-width: 1000px) {
        .composer-container { left: 0; }
    }
    
    .composer-inner {
        max-width: 48rem;
        margin: 0 auto;
    }
    
    .composer-box {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: var(--radius-lg);
        padding: 0.5rem;
        box-shadow: var(--shadow);
    }
    
    .composer-box input[type="text"],
    .composer-box textarea {
        background: transparent !important;
        color: var(--text-primary) !important;
        border: none !important;
        outline: none !important;
        resize: none !important;
        font-family: inherit !important;
        font-size: 0.9375rem !important;
        line-height: 1.25 !important;
        padding: 0.5rem !important;
        height: 40px !important;
        min-height: 40px !important;
        max-height: 40px !important;
    }
    
    .composer-box textarea::placeholder,
    .composer-box input[type="text"]::placeholder {
        color: var(--text-muted) !important;
    }
    
    /* Icon Buttons - ChatGPT style */
    .icon-button {
        width: 40px;
        height: 40px;
        min-width: 40px !important;
        min-height: 40px !important;
        border-radius: var(--radius-md) !important;
        background: transparent !important;
        border: none !important;
        color: var(--text-secondary) !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        cursor: pointer !important;
        transition: all 0.15s ease !important;
        font-size: 1.125rem !important;
        padding: 0 !important;
    }
    
    .icon-button:hover {
        background: var(--surface-hover) !important;
        color: var(--text-primary) !important;
    }
    
    .icon-button.send-btn {
        background: var(--accent-user) !important;
        color: white !important;
    }
    
    .icon-button.send-btn:hover {
        background: #1d4ed8 !important;
    }
    
    .icon-button:disabled {
        opacity: 0.4 !important;
        cursor: not-allowed !important;
    }
    
    /* Attachments */
    .attachment-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.25rem 0.5rem;
        border-radius: var(--radius-full);
        background: var(--surface-hover);
        color: var(--text-secondary);
        font-size: 0.8125rem;
        margin: 0.25rem 0.25rem 0 0;
    }
    .attachment-chip button {
        background: none;
        border: none;
        color: var(--text-muted);
        cursor: pointer;
        padding: 0;
        margin-left: 0.25rem;
        font-size: 0.875rem;
    }
    .attachment-chip button:hover {
        color: var(--text-primary);
    }
    
    /* Popover */
    div[data-testid="stPopover"] > div {
        background: var(--surface) !important;
        border: 1px solid var(--border) !important;
        border-radius: var(--radius-md) !important;
        box-shadow: var(--shadow) !important;
    }
    
    /* Streamlit overrides */
    .stTextArea textarea {
        background: transparent !important;
        border: none !important;
    }
    button[kind="primary"] {
        background: var(--accent-user) !important;
        border: none !important;
    }
    button[kind="secondary"] {
        background: transparent !important;
        border: 1px solid var(--border) !important;
        color: var(--text-secondary) !important;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: transparent;
        border-bottom: 1px solid var(--border);
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-secondary);
        border: none;
        padding: 0.5rem 1rem;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: var(--text-primary);
        border-bottom: 2px solid var(--accent-user);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    ::-webkit-scrollbar-track {
        background: var(--bg-main);
    }
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: var(--radius-sm);
    }
    ::-webkit-scrollbar-thumb:hover {
        background: var(--surface-hover);
    }
    
    </style>
    """,
    unsafe_allow_html=True,
)

# -------------- Utility & State --------------

def init_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []  # list of dicts {role, content}
    if "retriever" not in st.session_state:
        st.session_state.retriever = None
    if "model_ready" not in st.session_state:
        st.session_state.model_ready = False
    if "last_answer" not in st.session_state:
        st.session_state.last_answer = None
    if "retrieval_mode" not in st.session_state:
        st.session_state.retrieval_mode = os.getenv("RETRIEVAL_MODE", "dense")
    if "top_k" not in st.session_state:
        st.session_state.top_k = int(os.getenv("RETRIEVER_K", "6"))
    if "composer_text" not in st.session_state:
        st.session_state.composer_text = ""
    if "attachments" not in st.session_state:
        st.session_state.attachments = []  # list of dicts: {name, content}
    if "pending_voice" not in st.session_state:
        st.session_state.pending_voice = None  # store transcript temporarily
    if "last_audio_sig" not in st.session_state:
        st.session_state.last_audio_sig = None  # dedupe mic recordings
    if "transcript_preview" not in st.session_state:
        st.session_state.transcript_preview = None  # show latest transcript text
    if "last_input_preview" not in st.session_state:
        st.session_state.last_input_preview = None  # show latest submitted input

init_session_state()

# Lazy load model (user-triggered to save memory)
@st.cache_resource(show_spinner=True)
def get_pipeline():
    pipe = load_generation_pipeline()
    return pipe

# -------------- Sidebar (Settings) --------------
with st.sidebar:
    st.markdown("## ⚙️ Settings")

    st.markdown("Model in use:")
    st.code(hf_model_name, language="text")

    col_a, col_b = st.columns([1,1])
    with col_a:
        load_model_clicked = st.button("Load / Ensure Model", type="primary")
    with col_b:
        clear_chat_clicked = st.button("Clear Chat")

    # Retrieval controls
    retrieval_mode = st.selectbox("Retrieval mode", options=["dense", "hybrid"], index=(0 if st.session_state.retrieval_mode=="dense" else 1))
    st.session_state.retrieval_mode = retrieval_mode
    top_k = st.slider("Retriever k", min_value=1, max_value=10, value=st.session_state.top_k, step=1, help="Number of similar chunks fetched")
    st.session_state.top_k = top_k
    ensure_retriever = st.button("Apply Retriever Settings")

    skip_llm = os.getenv("SKIP_LLM", "false").lower() in {"1", "true", "yes"}
    st.toggle("Retrieval Only Mode (SKIP_LLM)", value=skip_llm, disabled=True, help="Enable by setting SKIP_LLM=true in .env and restarting.")
    if skip_llm:
        st.info("LLM generation is disabled (SKIP_LLM=true). The app will only retrieve and display context.")

    # Retriever health
    retriever_ok = st.session_state.get("retriever") is not None
    st.markdown(f"Retriever: {'✅ Ready' if retriever_ok else '❌ Not loaded'}")

    st.markdown("---")
    st.markdown("### About")
    st.write("Medical Dataset Retrieval-Augmented Generation interface. Build the vector DB first via `python Rag_1.py` if empty.")
    
    st.markdown("---")
    st.caption("**Built with Streamlit**")
    st.caption("RAG pipeline using HuggingFace + Chroma")
    st.caption("Theme: ChatGPT-style dark mode")

    if clear_chat_clicked:
        st.session_state.messages = []
        st.session_state.last_answer = None
        st.success("Chat history cleared.")

    if load_model_clicked:
        with st.spinner("Loading / validating generation pipeline ..."):
            try:
                pipe = get_pipeline()
                if pipe is None:
                    st.warning("Model not loaded (retrieval-only mode active or failure).")
                else:
                    st.session_state.model_ready = True
                    st.success("Model pipeline ready.")
            except Exception as e:
                st.error(f"Model load failed: {e}")
                st.session_state.model_ready = False

    if ensure_retriever:
        with st.spinner("Applying retriever settings ..."):
            try:
                st.session_state.retriever = init_vector_store(mode=st.session_state.retrieval_mode, k=st.session_state.top_k)
                st.success(f"Retriever ready ({st.session_state.retrieval_mode}, k={st.session_state.top_k})")
            except Exception as e:
                st.error(f"Failed to initialize retriever: {e}")
                st.session_state.retriever = None

# -------------- Main Tabs --------------
chat_tab, context_tab = st.tabs(["💬 Chat", "📄 Retrieved Context"])

with chat_tab:
    # Header
    st.markdown(
        """
        <div style="text-align: center; padding: 1rem 0 1.5rem;">
            <h1 style="font-size: 1.5rem; font-weight: 600; margin: 0; color: var(--text-primary);">🩺 Medical Data Analysis</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )
    
    # Ensure a retriever is available (lazy create on first load)
    if st.session_state.retriever is None:
        with st.spinner("Loading vector store ..."):
            try:
                st.session_state.retriever = init_vector_store(mode=st.session_state.retrieval_mode, k=st.session_state.top_k)
            except Exception as e:
                st.error(f"Retriever init failed: {e}")
                st.session_state.retriever = None

    if st.session_state.retriever is not None:
        # Display messages
        for msg in st.session_state.messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            avatar = "👤" if role == "user" else "🤖"
            
            with st.chat_message(role, avatar=avatar):
                st.markdown(content)

        # Input controls row
        col1, col2, col3 = st.columns([0.80, 0.10, 0.10])
        
        with col1:
            user_query = st.chat_input("Type your medical question...")
        
        with col2:
            # Voice recorder button
            if _MIC_AVAILABLE:
                with st.popover("🎙️", use_container_width=True):
                    st.write("**🎙️ Voice Input (Multilingual)**")
                    st.caption("Record your question. We’ll transcribe and translate it automatically if needed.")
                    audio = mic_recorder(
                        start_prompt="▶️ Start Recording",
                        stop_prompt="⏹️ Stop Recording",
                        just_once=False,
                        use_container_width=True,
                        key="mic_recorder",
                    )
                    # If new audio bytes appear, transcribe once (debounced by simple signature)
                    if audio and isinstance(audio, dict) and audio.get("bytes"):
                        sig = (len(audio["bytes"]) if isinstance(audio["bytes"], (bytes, bytearray)) else None)
                        if sig and sig != st.session_state.last_audio_sig:
                            st.session_state.last_audio_sig = sig
                            with st.spinner("Transcribing..."):
                                try:
                                    transcript = transcribe_audio(audio["bytes"])  # type: ignore[arg-type]
                                    transcript = (transcript or "").strip()
                                    if transcript:
                                        st.session_state.transcript_preview = transcript
                                        # Optional: detect language and produce English preview for bilingual display
                                        preview_en = None
                                        lang = "unknown"
                                        if _SARVAM_HELPERS:
                                            try:
                                                lang = detect_language_simple(transcript)
                                            except Exception:
                                                lang = "unknown"
                                            # Attempt English translation regardless of detection to be robust
                                            with st.spinner("Translating transcript → English ..."):
                                                try:
                                                    preview_en = sarvam_translate(transcript, target_lang="en")
                                                except Exception:
                                                    preview_en = None
                                        # Store preview meta for UI rendering below
                                        st.session_state["_transcript_meta"] = {"lang": lang, "en": preview_en}
                                        st.success("✅ Voice transcribed. Review below.")
                                    else:
                                        st.warning("No speech detected. Please try again.")
                                except Exception as e:
                                    st.error(f"❌ Transcription failed: {str(e)}")
                                    st.caption("Make sure faster-whisper is installed and FFmpeg is available.")

                    # Show transcript preview with actions
                    if st.session_state.transcript_preview:
                        # Show transcript in original language and English (if different)
                        meta = st.session_state.get("_transcript_meta", {}) or {}
                        lang = meta.get("lang", "unknown")
                        st.text_area(f"Transcript ({lang})", value=st.session_state.transcript_preview, height=100)
                        en_prev = meta.get("en")
                        if en_prev and isinstance(en_prev, str) and en_prev.strip():
                            st.text_area("Transcript (English)", value=en_prev, height=100)
                        c1, c2 = st.columns(2)
                        with c1:
                            if st.button("Send Transcript", use_container_width=True):
                                st.session_state.pending_voice = st.session_state.transcript_preview
                                st.session_state.transcript_preview = None
                                st.session_state["_transcript_meta"] = None
                                st.rerun()
                        with c2:
                            if st.button("Discard", use_container_width=True):
                                st.session_state.transcript_preview = None
                                st.session_state["_transcript_meta"] = None
            else:
                st.button("🎙️", disabled=True, use_container_width=True, help="Mic recorder not available. Install streamlit-mic-recorder and faster-whisper.")
        
        with col3:
            # Attach files button
            with st.popover("📎", use_container_width=True):
                st.write("**Attach Files**")
                up = st.file_uploader("Upload .txt or .md files", type=["txt", "md"], accept_multiple_files=True, label_visibility="collapsed")
                if up:
                    for f in up:
                        try:
                            data = f.read().decode("utf-8", errors="ignore")
                            if not any(a["name"] == f.name for a in st.session_state.attachments):
                                st.session_state.attachments.append({"name": f.name, "content": data[:5000]})
                                st.success(f"✓ Attached: {f.name}")
                        except Exception as e:
                            st.error(f"Failed to read {getattr(f,'name','file')}: {e}")
                
                if st.session_state.attachments:
                    st.divider()
                    st.write(f"**{len(st.session_state.attachments)} file(s) attached:**")
                    for att in st.session_state.attachments:
                        st.caption(f"📎 {att['name']}")
                    if st.button("Clear all", use_container_width=True):
                        st.session_state.attachments = []
                        st.rerun()

        # Handle input - do not auto-submit; submit when: user typed OR clicked 'Send Transcript'
        effective_query: Optional[str] = None
        if user_query:
            effective_query = user_query.strip()
            st.session_state.last_input_preview = effective_query
        elif st.session_state.pending_voice and st.session_state.pending_voice.strip():
            effective_query = st.session_state.pending_voice.strip()
            st.session_state.last_input_preview = effective_query
            st.session_state.pending_voice = None

        if effective_query:
            # Add attachments if any
            if st.session_state.attachments:
                extra = "\n\nAttached Notes:\n" + "\n---\n".join(
                    [a.get("content", "") for a in st.session_state.attachments]
                )
                effective_query = effective_query + extra
            
            # For display: show both original input and English translation when applicable
            try:
                in_lang = detect_language_simple(effective_query) if _SARVAM_HELPERS else "en"
            except Exception:
                in_lang = "unknown"
            if _SARVAM_HELPERS and in_lang != "en":
                try:
                    effective_query_en = sarvam_translate(effective_query, target_lang="en")
                except Exception:
                    effective_query_en = None
            else:
                effective_query_en = None

            # Persist full user message content plus a bilingual note in chat history for rendering
            user_display = effective_query
            bilingual_note = ""
            if effective_query_en and effective_query_en.strip():
                bilingual_note = f"\n\n**English translation:**\n\n{effective_query_en}"
            st.session_state.messages.append({"role": "user", "content": user_display + (bilingual_note if bilingual_note else "")})
            with st.chat_message("user"):
                st.markdown(user_display)
                if bilingual_note:
                    st.markdown(bilingual_note)

            with st.chat_message("assistant", avatar="🤖"):
                placeholder = st.empty()
                try:
                    result = answer_query(effective_query, st.session_state.retriever)
                    st.session_state.last_answer = result
                    answer = result.get("answer") or "No answer (retrieval only mode)."
                    answer_en = result.get("answer_en")
                    # Stream effect (fake typing) for main answer (displayed in user's language)
                    streamed = ""
                    for token in answer.split():
                        streamed += token + " "
                        placeholder.markdown(streamed + "▌")
                        time.sleep(0.02)
                    # Replace placeholder with final block including bilingual rendering
                    if answer_en and isinstance(answer_en, str) and answer_en.strip() and answer_en.strip() != answer.strip():
                        placeholder.markdown(answer)
                        st.caption("English version:")
                        st.markdown(answer_en)
                        combined_for_history = answer + "\n\n**English version:**\n\n" + answer_en
                        st.session_state.messages.append({"role": "assistant", "content": combined_for_history})
                    else:
                        placeholder.markdown(answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err_text = f"Error generating answer: {e}" \
                               f"\n\n````\n{traceback.format_exc()}\n````"
                    placeholder.error(err_text)

with context_tab:
    st.markdown("### Retrieved Context")
    if st.session_state.last_answer and st.session_state.last_answer.get("context"):
        ctx = st.session_state.last_answer["context"]
        # Provide collapsible sections if context large
        if len(ctx) > 4000:
            st.write(f"Context length: {len(ctx)} chars")
        st.text_area("Raw Retrieved Context", value=ctx, height=400)
    else:
        st.info("No context yet. Ask a question in Chat tab.")

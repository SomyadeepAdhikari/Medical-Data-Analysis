import os
import time
import traceback
from typing import List, Dict, Any

import streamlit as st
from dotenv import load_dotenv

# Reuse RAG logic by importing functions from Rag_final.py
# (Assumes Rag_final.py is in same directory.)
from Rag_final import init_vector_store, answer_query, load_generation_pipeline, hf_model_name, transcribe_audio

# Optional mic recorder component
try:
    from streamlit_mic_recorder import mic_recorder  # type: ignore
    _MIC_AVAILABLE = True
except Exception:
    _MIC_AVAILABLE = False

load_dotenv()

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
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
        st.session_state.top_k = int(os.getenv("RETRIEVER_K", "3"))

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
    st.title("🩺 Medical RAG Assistant")
    st.caption("Ask questions grounded in your medical dataset.")

    # Ensure a retriever is available (lazy create on first load)
    if st.session_state.retriever is None:
        with st.spinner("Loading vector store ..."):
            try:
                st.session_state.retriever = init_vector_store(mode=st.session_state.retrieval_mode, k=st.session_state.top_k)
            except Exception as e:
                st.error(f"Retriever init failed: {e}")
                st.session_state.retriever = None

    if st.session_state.retriever is not None:
        # Guard: don't show chat input until retriever is ready
        user_query = st.chat_input("Type your medical question...")

        # Voice input: microphone (preferred) or file upload (fallback)
        voice_transcript = None
        if _MIC_AVAILABLE:
            st.caption("Speak your question and click Stop to transcribe")
            audio = mic_recorder(start_prompt="🎙️ Start recording", stop_prompt="⏹️ Stop", just_once=False, use_container_width=True)
            if audio and isinstance(audio, dict) and audio.get("bytes"):
                try:
                    voice_transcript = transcribe_audio(audio["bytes"])  # WAV bytes provided by component
                    st.info(f"Transcribed: {voice_transcript}")
                except Exception as e:
                    st.warning(f"Mic transcription failed: {e}")
        else:
            voice_file = st.file_uploader("Optional: Upload a WAV file with your question (any language)", type=["wav"], accept_multiple_files=False)
            if voice_file is not None:
                try:
                    audio_bytes = voice_file.read()
                    voice_transcript = transcribe_audio(audio_bytes)
                    st.info(f"Transcribed voice query: {voice_transcript}")
                except Exception as e:
                    st.warning(f"Voice transcription unavailable or failed: {e}")

        # Render previous conversation
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])    

        effective_query = voice_transcript or user_query

        if effective_query:
            st.session_state.messages.append({"role": "user", "content": effective_query})
            with st.chat_message("user"):
                st.markdown(effective_query)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                try:
                    # Build a lightweight messages list compatible with answer_query
                    # answer_query itself will manage message appends internally but we only store plain text here
                    result = answer_query(effective_query, st.session_state.retriever)
                    answer = result.get("answer") or "No answer (retrieval only mode)."
                    st.session_state.last_answer = result
                    # Stream effect (fake typing)
                    streamed = ""
                    for token in answer.split():
                        streamed += token + " "
                        placeholder.markdown(streamed + "▌")
                        time.sleep(0.02)
                    placeholder.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    err_text = f"Error generating answer: {e}"\
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

# -------------- Footer --------------
st.markdown("---")
st.caption("Built with Streamlit · RAG pipeline using HuggingFace + Chroma · Theme: custom dark mode")

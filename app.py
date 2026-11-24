import os
import streamlit as st
from main import run_pipeline
from core.rag_engine import ask_question

# Page configuration
st.set_page_config(
    page_title="AI Audio Assistant",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize ephemeral session state variables (wiped on tab close/refresh)
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_source" not in st.session_state:
    st.session_state.processed_source = None

# --- SIDEBAR: INPUT & SETTINGS ---
with st.sidebar:
    st.title("🎙️ Audio Assistant")
    st.caption("Powered by Whisper, Gemini & ChromaDB")
    st.divider()

    source_path = None
    uploaded_temp_path = None

    uploaded_file = st.file_uploader(
        "Upload Audio File",
        type=["mp4", "mp3", "wav", "m4a", "mkv", "webm"]
    )
    if uploaded_file is not None:
        os.makedirs("downloades", exist_ok=True)
        uploaded_temp_path = os.path.join("downloades", uploaded_file.name)
        with open(uploaded_temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        source_path = uploaded_temp_path

    language = st.selectbox(
        "Spoken Language",
        options=["english", "hindi", "hinglish"],
        index=0,
        help="Hindi and Hinglish audio are automatically translated into English for notes and Q&A."
    )

    process_btn = st.button(
        "🚀 Process Audio",
        type="primary",
        use_container_width=True,
        disabled=(source_path is None)
    )

    if st.session_state.pipeline_result is not None:
        st.divider()
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.pipeline_result = None
            st.session_state.chat_history = []
            st.session_state.processed_source = None
            st.rerun()

    st.divider()
    st.caption("🔒 Session data is temporary and clears when you close or refresh the tab.")

# --- PIPELINE EXECUTION ---
if process_btn and source_path:
    with st.status("Running AI Audio Assistant Pipeline...", expanded=True) as status:
        st.write("📥 Converting and chunking audio...")
        st.write("🗣️ Transcribing / translating speech with Whisper (`base`)...")
        st.write("🧠 Generating title, summary & extracting insights via Gemini...")
        st.write("🗂️ Building Chroma vector store for RAG Q&A...")

        try:
            result = run_pipeline(source_path, language)
            st.session_state.pipeline_result = result
            st.session_state.chat_history = []
            st.session_state.processed_source = source_path

            # Remove temporary uploaded file after processing
            if uploaded_temp_path and os.path.exists(uploaded_temp_path):
                os.remove(uploaded_temp_path)

            status.update(label="✅ Processing Complete!", state="complete", expanded=False)
        except Exception as e:
            status.update(label="❌ Pipeline Failed", state="error", expanded=True)
            st.error(f"An error occurred: {e}")

# --- MAIN CONTENT AREA ---
if st.session_state.pipeline_result is None:
    st.header("Welcome to AI Audio Assistant")
    st.markdown(
        """
        Turn any audio recording into structured notes and an interactive knowledge base.
        
        **How to get started:**
1. Upload an **Audio file** in the left sidebar.
2. Select the spoken language (**English**, **Hindi**, or **Hinglish**).
3. Click **🚀 Process Audio** to extract summaries, action items, decisions, and chat with the content.
        """
    )
else:
    res = st.session_state.pipeline_result

    # Header & Title
    st.subheader(f"📌 {res['title']}")
    st.divider()

    # Two-column layout: Left for Insights/Transcript, Right for RAG Chat
    col_insights, col_chat = st.columns([1.3, 1], gap="large")

    with col_insights:
        tab_summary, tab_actions, tab_questions, tab_transcript = st.tabs([
            "📋 Summary",
            "✅ Actions & Decisions",
            "❓ Open Questions",
            "📝 Transcript"
        ])

        with tab_summary:
            st.markdown(res["summary"])

        with tab_actions:
            st.markdown("#### ✅ Action Items")
            st.markdown(res["action_items"])
            st.divider()
            st.markdown("#### 🔑 Key Decisions")
            st.markdown(res["key_decisions"])

        with tab_questions:
            st.markdown("#### ❓ Open Questions")
            st.markdown(res["open_questions"])

        with tab_transcript:
            st.download_button(
                label="⬇️ Download Transcript (.txt)",
                data=res["transcript"],
                file_name=f"{res['title'][:40].strip().replace(' ', '_')}_transcript.txt",
                mime="text/plain"
            )
            st.text_area(
                "Full Raw Transcript",
                value=res["transcript"],
                height=420,
                label_visibility="collapsed"
            )

    with col_chat:
        st.markdown("### 💬 Chat with Audio")
        st.caption("Ask questions grounded strictly in the audio transcript.")

        # Scrollable chat container
        chat_container = st.container(height=430, border=True)
        with chat_container:
            for msg in st.session_state.chat_history:
                with st.chat_message(msg["role"]):
                    st.markdown(msg["content"])

        # Chat input box
        user_query = st.chat_input("Ask a question about the transcript...")
        if user_query:
            # Append user message
            st.session_state.chat_history.append({"role": "user", "content": user_query})
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_query)

                with st.chat_message("assistant"):
                    with st.spinner("Searching transcript..."):
                        answer = ask_question(res["rag_chain"], user_query)
                        st.markdown(answer)

            st.session_state.chat_history.append({"role": "assistant", "content": answer})

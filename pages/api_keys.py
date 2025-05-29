import streamlit as st

def show():
    st.header("🔑 API Key Configuration")
    
    st.markdown("""
    Enter your API keys below. These will be stored in your session and used for processing.
    
    **Note:** API keys are stored only for this session and are not saved permanently.
    """)
    
    # API key inputs
    st.subheader("Enter API Keys")
    
    anthropic_key = st.text_input(
        "Anthropic API Key (for Claude)",
        value=st.session_state.api_keys.get('ANTHROPIC_API_KEY', ''),
        type="password",
        help="Get your API key from https://console.anthropic.com/"
    )
    if anthropic_key:
        st.session_state.api_keys['ANTHROPIC_API_KEY'] = anthropic_key
    
    openai_key = st.text_input(
        "OpenAI API Key (for GPT-4o)",
        value=st.session_state.api_keys.get('OPENAI_API_KEY', ''),
        type="password",
        help="Get your API key from https://platform.openai.com/api-keys"
    )
    if openai_key:
        st.session_state.api_keys['OPENAI_API_KEY'] = openai_key
    
    google_key = st.text_input(
        "Google API Key (for Gemini)",
        value=st.session_state.api_keys.get('GOOGLE_API_KEY', ''),
        type="password",
        help="Get your API key from https://makersuite.google.com/app/apikey"
    )
    if google_key:
        st.session_state.api_keys['GOOGLE_API_KEY'] = google_key
    
    st.subheader("API Key Status")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.session_state.api_keys.get('ANTHROPIC_API_KEY'):
            st.success("✅ Anthropic API Key Set")
        else:
            st.error("❌ Anthropic API Key Missing")
    
    with col2:
        if st.session_state.api_keys.get('OPENAI_API_KEY'):
            st.success("✅ OpenAI API Key Set")
        else:
            st.error("❌ OpenAI API Key Missing")
    
    with col3:
        if st.session_state.api_keys.get('GOOGLE_API_KEY'):
            st.success("✅ Google API Key Set")
        else:
            st.error("❌ Google API Key Missing")
    
    if st.button("✅ Confirm API Keys", type="primary"):
        st.success("API keys saved for this session!")
        st.info("You can now proceed to upload data.")

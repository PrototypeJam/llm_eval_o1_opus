# uv venv
# source .venv/bin/activate
# uv pip install -r requirements.txt
# streamlit run app.py

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="LLM Evaluation Tool",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'processed_results' not in st.session_state:
    st.session_state.processed_results = None
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'api_keys' not in st.session_state:
    st.session_state.api_keys = {
        'ANTHROPIC_API_KEY': '',
        'OPENAI_API_KEY': '',
        'GOOGLE_API_KEY': ''
    }

# Title
st.title("🤖 Multi-Model LLM Evaluation Tool")

# Sidebar navigation
page = st.sidebar.selectbox(
    "Navigation",
    ["Home", "API Keys", "Upload Data", "Process", "Results"]
)

# Page routing
if page == "Home":
    from pages import home
    home.show()
elif page == "API Keys":
    from pages import api_keys
    api_keys.show()
elif page == "Upload Data":
    from pages import upload
    upload.show()
elif page == "Process":
    from pages import process
    process.show()
elif page == "Results":
    from pages import results
    results.show()

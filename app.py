import streamlit as st
from streamlit_option_menu import option_menu

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

# Title
st.title("🤖 Multi-Model LLM Evaluation Tool")

# Sidebar navigation
with st.sidebar:
    st.header("Navigation")
    page = option_menu(
        menu_title=None,
        options=["Home", "Upload Data", "Process", "Results"],
        icons=["house", "cloud-upload", "cpu", "graph-up"],
        default_index=0,
    )

# Page routing
if page == "Home":
    from pages import home
    home.show()
elif page == "Upload Data":
    from pages import upload
    upload.show()
elif page == "Process":
    from pages import process
    process.show()
elif page == "Results":
    from pages import results
    results.show()

import streamlit as st

def show():
    st.header("Welcome to the LLM Evaluation Tool")

    st.markdown("""
    This tool allows you to evaluate and compare responses from multiple Large Language Models (LLMs)
    using your own test data.

    ### 🚀 Features
    - **Multi-Model Support**: Compare Claude Sonnet 3.7, OpenAI GPT-4o, and Google Gemini 2.5
    - **Batch Processing**: Process multiple prompts from CSV files
    - **Real-time Progress**: Watch as your prompts are processed
    - **Export Results**: Download results with model outputs

    ### 📋 How to Use
    1. **Upload Data**: Upload a CSV file with columns: `input`, `output`, `expected`
    2. **Select Model**: Choose which LLM to use for processing
    3. **Process**: Run your prompts through the selected model
    4. **Download Results**: Export the results with model responses

    ### 🔑 API Keys Required
    Make sure the following API keys are configured in your deployment:
    - `ANTHROPIC_API_KEY` for Claude
    - `OPENAI_API_KEY` for GPT-4o
    - `GOOGLE_API_KEY` for Gemini

    ### 📊 CSV Format Example
    ```csv
    input,output,expected
    "What is 2+2?","","4"
    "Capital of France?","","Paris"
    ```
    """)

    # Check API key status
    st.subheader("🔐 API Key Status")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.secrets.get("ANTHROPIC_API_KEY"):
            st.success("✅ Anthropic API Key")
        else:
            st.error("❌ Anthropic API Key Missing")

    with col2:
        if st.secrets.get("OPENAI_API_KEY"):
            st.success("✅ OpenAI API Key")
        else:
            st.error("❌ OpenAI API Key Missing")

    with col3:
        if st.secrets.get("GOOGLE_API_KEY"):
            st.success("✅ Google API Key")
        else:
            st.error("❌ Google API Key Missing")

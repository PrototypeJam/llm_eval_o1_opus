import streamlit as st
import pandas as pd
from services.csv_processor import CSVProcessor

def show():
    st.header("📤 Upload CSV Data")

    st.markdown("""
    Upload a CSV file containing your test prompts. The file must have the following columns:
    - **input**: The prompt to send to the LLM
    - **output**: Leave empty (will be filled by the LLM)
    - **expected**: The expected response (optional, for comparison)
    """)

    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=['csv'],
        help="Upload a CSV file with 'input', 'output', and 'expected' columns"
    )

    if uploaded_file is not None:
        try:
            # Read CSV
            df = pd.read_csv(uploaded_file)

            # Validate CSV
            is_valid, message = CSVProcessor.validate_csv(df)

            if is_valid:
                st.success(f"✅ {message}")

                # Show preview
                st.subheader("Data Preview")
                st.dataframe(df.head(10))

                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Rows", len(df))
                with col2:
                    st.metric("Non-empty Expected", df['expected'].notna().sum())
                with col3:
                    st.metric("Unique Prompts", df['input'].nunique())

                # Save to session state
                if st.button("✅ Confirm and Proceed", type="primary"):
                    st.session_state.uploaded_data = df
                    st.success("Data uploaded successfully! Go to the Process page to continue.")

            else:
                st.error(f"❌ {message}")

        except Exception as e:
            st.error(f"Error reading CSV file: {str(e)}")

    # Show current data status
    if st.session_state.uploaded_data is not None:
        st.info(f"📊 Current data: {len(st.session_state.uploaded_data)} rows loaded")

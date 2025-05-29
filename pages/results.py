import streamlit as st
import pandas as pd
from services.csv_processor import CSVProcessor
from datetime import datetime

def show():
    st.header("📊 Results & Export")

    # Check if results exist
    if st.session_state.processed_results is None:
        st.warning("⚠️ No results to display. Please process data first!")
        st.stop()

    results_df = st.session_state.processed_results
    model_used = st.session_state.selected_model or "Unknown"

    # Summary statistics
    st.subheader("Processing Summary")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Model Used", model_used)
    with col2:
        st.metric("Total Prompts", len(results_df))
    with col3:
        success_rate = (results_df['status'] == 'Success').mean() * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    with col4:
        avg_length = results_df[results_df['status'] == 'Success']['output'].str.len().mean()
        st.metric("Avg Response Length", f"{avg_length:.0f} chars")

    # Display options
    st.subheader("View Results")

    # Filter options
    col1, col2 = st.columns(2)
    with col1:
        show_all = st.checkbox("Show all columns", value=True)
    with col2:
        filter_errors = st.checkbox("Show only errors", value=False)

    # Apply filters
    display_df = results_df.copy()
    if filter_errors:
        display_df = display_df[display_df['status'] != 'Success']

    if not show_all:
        display_df = display_df[['input', 'output', 'expected', 'status']]

    # Display data
    st.dataframe(display_df, use_container_width=True)

    # Export section
    st.subheader("📥 Export Results")

    # Prepare export data
    export_df = results_df.copy()
    export_df['model'] = model_used
    export_df['processed_at'] = datetime.now().isoformat()

    # Download button
    csv_bytes = CSVProcessor.export_results(export_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"llm_results_{model_used.replace(' ', '_')}_{timestamp}.csv"

    st.download_button(
        label="📥 Download Results as CSV",
        data=csv_bytes,
        file_name=filename,
        mime="text/csv",
        help="Download the complete results including model outputs",
    )

    # Additional analysis
    if st.checkbox("Show detailed analysis"):
        st.subheader("Detailed Analysis")

        # Response length distribution
        if success_rate > 0:
            st.write("**Response Length Distribution**")
            success_outputs = results_df[results_df['status'] == 'Success']['output']
            lengths = success_outputs.str.len()

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min Length", f"{lengths.min()} chars")
            with col2:
                st.metric("Max Length", f"{lengths.max()} chars")
            with col3:
                st.metric("Median Length", f"{lengths.median():.0f} chars")

        # Error analysis
        errors = results_df[results_df['status'] != 'Success']
        if len(errors) > 0:
            st.write("**Error Summary**")
            error_counts = errors['status'].value_counts()
            st.dataframe(error_counts)

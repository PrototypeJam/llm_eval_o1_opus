import streamlit as st
import time
from services.llm_factory import LLMFactory
from services.csv_processor import CSVProcessor

def show():
    st.header("🤖 Process Data with LLM")

    # Check if data is uploaded
    if st.session_state.uploaded_data is None:
        st.warning("⚠️ Please upload data first!")
        st.stop()

    # Model selection
    st.subheader("Select LLM Model")
    available_models = LLMFactory.get_available_providers()

    selected_model = st.selectbox(
        "Choose a model:",
        available_models,
        index=0,
        help="Select which LLM to use for processing your prompts"
    )

    # Validate API key for selected model
    provider = LLMFactory.create_provider(selected_model)

    if not provider or not provider.validate_api_key():
        st.error(f"❌ API key not configured for {selected_model}")
        st.stop()

    st.success(f"✅ {selected_model} is ready to use")

    # Processing options
    st.subheader("Processing Options")

    col1, col2 = st.columns(2)
    with col1:
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=10,
            value=1,
            help="Number of prompts to process in parallel (1 for sequential)"
        )

    with col2:
        delay_between = st.number_input(
            "Delay (seconds)",
            min_value=0.0,
            max_value=5.0,
            value=0.5,
            step=0.1,
            help="Delay between API calls to avoid rate limits"
        )

    # Show data summary
    df = st.session_state.uploaded_data
    st.info(f"Ready to process {len(df)} prompts with {selected_model}")

    # Process button
    if st.button("🚀 Start Processing", type="primary"):
        st.session_state.selected_model = selected_model

        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        results_container = st.container()

        # Process data
        def update_progress(current, total):
            progress = current / total
            progress_bar.progress(progress)
            status_text.text(f"Processing: {current}/{total} prompts")
            time.sleep(delay_between)  # Rate limiting

        with st.spinner("Processing prompts..."):
            results_df = CSVProcessor.process_csv_with_llm(
                df,
                provider,
                progress_callback=update_progress
            )

        # Save results
        st.session_state.processed_results = results_df

        # Show completion
        progress_bar.progress(1.0)
        status_text.text("✅ Processing complete!")

        # Show sample results
        with results_container:
            st.subheader("Sample Results")
            st.dataframe(results_df.head(5))

            # Success metrics
            success_count = (results_df['status'] == 'Success').sum()
            error_count = len(results_df) - success_count

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Processed", len(results_df))
            with col2:
                st.metric("Successful", success_count, delta=f"{success_count/len(results_df)*100:.1f}%")
            with col3:
                st.metric("Errors", error_count)

        st.success("✅ Processing complete! Go to Results page to download your data.")

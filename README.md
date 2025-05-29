# Response - 4 Opus

Based on my research, I now have comprehensive information about deploying multi-file Streamlit applications to Community Cloud and integrating various LLM APIs. Let me provide you with the detailed documentation and development plan you requested.

## 1. Detailed Documentation: Creating and Deploying Multi-File Streamlit Apps to Community Cloud

### Overview
Streamlit Community Cloud allows you to deploy Python applications with multiple files that work together as a seamless app. Here's how it works:

### File Organization Structure

For a multi-file Streamlit app, you can organize your project in two main ways:

**Option 1: Using the pages/ directory (Simple Multi-Page)**
```
my-streamlit-app/
├── app.py                 # Main entrypoint file
├── pages/                 # Directory for additional pages
│   ├── 1_📊_Data.py      # Page files (numbered for ordering)
│   ├── 2_📈_Analysis.py
│   └── 3_⚙️_Settings.py
├── utils/                 # Helper modules
│   ├── __init__.py
│   ├── api_helpers.py
│   └── data_processing.py
├── requirements.txt       # Python dependencies
├── .streamlit/           # Configuration directory
│   └── config.toml       # App configuration
└── README.md             # Documentation
```

**Option 2: Using st.Page and st.navigation (Advanced)**
```
my-streamlit-app/
├── app.py                # Router/entrypoint
├── views/               # Page components
│   ├── home.py
│   ├── processor.py
│   └── results.py
├── services/            # Business logic
│   ├── llm_service.py
│   └── csv_handler.py
├── requirements.txt
└── .streamlit/
    └── secrets.toml     # Local secrets (don't commit!)
```

### Key Concepts for Multi-File Apps

1. **Entrypoint File**: The main file you run with `streamlit run app.py`
2. **Module Imports**: You can import functions and classes from other Python files
3. **Shared State**: Use `st.session_state` to share data between pages
4. **Secrets Management**: Store API keys securely using `st.secrets`

### Deployment Steps to Streamlit Community Cloud

1. **Prepare Your Repository**
   - Create a GitHub repository
   - Push all your Python files
   - Include `requirements.txt` with all dependencies
   - Never commit secrets or API keys

2. **Set Up Streamlit Community Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"

3. **Configure Your App**
   - Select your GitHub repository
   - Choose the branch (usually `main`)
   - Specify the main file path (e.g., `app.py`)
   - Add secrets in the "Advanced settings" section

4. **Add Secrets**
   ```toml
   # Format for secrets.toml
   OPENAI_API_KEY = "sk-..."
   ANTHROPIC_API_KEY = "sk-ant-..."
   GOOGLE_API_KEY = "..."
   
   [database]
   host = "localhost"
   port = 5432
   ```

5. **Deploy**
   - Click "Deploy"
   - Your app will be available at: `https://[yourapp].streamlit.app`

## 2. Detailed Development Plan for Your LLM Evaluation App

Here's a comprehensive plan for building your multi-model LLM evaluation app:

### Project Structure
```
llm-evaluation-app/
├── app.py                    # Main entry point with navigation
├── pages/
│   ├── 1_🏠_Home.py         # Welcome page and instructions
│   ├── 2_📤_Upload.py       # CSV upload interface
│   ├── 3_🤖_Process.py      # LLM processing page
│   └── 4_📊_Results.py      # Results visualization
├── services/
│   ├── __init__.py
│   ├── llm_factory.py       # Factory pattern for LLM providers
│   ├── csv_processor.py     # CSV handling logic
│   └── api_clients/
│       ├── __init__.py
│       ├── anthropic_client.py
│       ├── openai_client.py
│       └── google_client.py
├── utils/
│   ├── __init__.py
│   └── validators.py        # Input validation
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── README.md
```

## Implementation Plan 

## LLM Evaluation App - Complete Implementation Guide

## Phase 1: Core Infrastructure Setup

### 1.1 Main Application Entry (app.py)
```python
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
```

### 1.2 LLM Factory Pattern (services/llm_factory.py)
```python
from abc import ABC, abstractmethod
import streamlit as st
from typing import Dict, Any, Optional
import time

class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    def validate_api_key(self) -> bool:
        pass

class AnthropicProvider(LLMProvider):
    def __init__(self):
        self.api_key = st.secrets.get("ANTHROPIC_API_KEY")
        
    def validate_api_key(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            from anthropic import Anthropic
            
            client = Anthropic(api_key=self.api_key)
            
            response = client.messages.create(
                model="claude-3.7-sonnet-latest",
                max_tokens=1000,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            return response.content[0].text
            
        except Exception as e:
            return f"Error: {str(e)}"

class OpenAIProvider(LLMProvider):
    def __init__(self):
        self.api_key = st.secrets.get("OPENAI_API_KEY")
        
    def validate_api_key(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=self.api_key)
            
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error: {str(e)}"

class GoogleVertexProvider(LLMProvider):
    def __init__(self):
        self.api_key = st.secrets.get("GOOGLE_API_KEY")
        self.project_id = st.secrets.get("GOOGLE_PROJECT_ID", "")
        
    def validate_api_key(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, **kwargs) -> str:
        try:
            import google.generativeai as genai
            
            genai.configure(api_key=self.api_key)
            
            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(prompt)
            
            return response.text
            
        except Exception as e:
            return f"Error: {str(e)}"

class LLMFactory:
    """Factory class to create LLM providers"""
    
    _providers = {
        "Claude Sonnet 3.7": AnthropicProvider,
        "OpenAI GPT-4o": OpenAIProvider,
        "Google Gemini 2.5": GoogleVertexProvider
    }
    
    @classmethod
    def create_provider(cls, provider_name: str) -> Optional[LLMProvider]:
        provider_class = cls._providers.get(provider_name)
        if provider_class:
            return provider_class()
        return None
    
    @classmethod
    def get_available_providers(cls) -> list:
        return list(cls._providers.keys())
```

### 1.3 CSV Processor (services/csv_processor.py)
```python
import pandas as pd
import streamlit as st
from typing import List, Dict, Any
import io
from datetime import datetime

class CSVProcessor:
    """Handles CSV upload, validation, and processing"""
    
    @staticmethod
    def validate_csv(df: pd.DataFrame) -> tuple[bool, str]:
        """Validate CSV has required columns"""
        required_columns = ['input', 'output', 'expected']
        
        # Check if all required columns exist
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Check for empty dataframe
        if df.empty:
            return False, "CSV file is empty"
        
        # Check for null values in input column
        if df['input'].isnull().any():
            return False, "Input column contains null values"
        
        return True, "CSV is valid"
    
    @staticmethod
    def process_csv_with_llm(df: pd.DataFrame, llm_provider, progress_callback=None) -> pd.DataFrame:
        """Process each row of the CSV through the selected LLM"""
        results = []
        total_rows = len(df)
        
        for idx, row in df.iterrows():
            if progress_callback:
                progress_callback(idx + 1, total_rows)
            
            input_prompt = row['input']
            
            # Generate response
            try:
                output = llm_provider.generate(input_prompt)
                status = "Success"
            except Exception as e:
                output = ""
                status = f"Error: {str(e)}"
            
            results.append({
                'input': input_prompt,
                'output': output,
                'expected': row.get('expected', ''),
                'status': status,
                'timestamp': datetime.now().isoformat()
            })
        
        return pd.DataFrame(results)
    
    @staticmethod
    def export_results(df: pd.DataFrame) -> bytes:
        """Export results to CSV bytes"""
        output = io.BytesIO()
        df.to_csv(output, index=False, encoding='utf-8')
        return output.getvalue()
```

### 1.4 Pages Implementation

#### Home Page (pages/home.py)
```python
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
```

#### Upload Page (pages/upload.py)
```python
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
```

#### Process Page (pages/process.py)
```python
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
```

#### Results Page (pages/results.py)
```python
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
        help="Download the complete results including model outputs"
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
```

### 1.5 Requirements File (requirements.txt)
```
streamlit>=1.28.0
pandas>=2.0.0
anthropic>=0.18.0
openai>=1.0.0
google-generativeai>=0.5.0
streamlit-option-menu>=0.3.0
python-dotenv>=1.0.0
```

### 1.6 Configuration File (.streamlit/config.toml)
```toml
[theme]
primaryColor = "#FF6B6B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

## Phase 2: Deployment Configuration

### 2.1 Secrets Configuration (for Streamlit Community Cloud)
```toml
# Add this in the "Secrets" section of Advanced Settings
ANTHROPIC_API_KEY = "sk-ant-api03-..."
OPENAI_API_KEY = "sk-..."
GOOGLE_API_KEY = "AIza..."
GOOGLE_PROJECT_ID = "your-project-id"  # Optional for Vertex AI

[app]
name = "LLM Evaluation Tool"
version = "1.0.0"
```

### 2.2 Local Development Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create local secrets file (don't commit!)
mkdir .streamlit
touch .streamlit/secrets.toml

# Run locally
streamlit run app.py
```

## Phase 3: Advanced Features (Optional Enhancements)

### 3.1 Parallel Processing Enhancement
```python
# In services/csv_processor.py
import asyncio
from concurrent.futures import ThreadPoolExecutor
import streamlit as st

class AsyncCSVProcessor(CSVProcessor):
    @staticmethod
    async def process_csv_async(df, llm_provider, max_workers=5):
        """Process CSV with parallel API calls"""
        
        async def process_row(row):
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor() as executor:
                output = await loop.run_in_executor(
                    executor, 
                    llm_provider.generate, 
                    row['input']
                )
            return output
        
        tasks = [process_row(row) for _, row in df.iterrows()]
        results = await asyncio.gather(*tasks)
        
        df['output'] = results
        return df
```

### 3.2 Response Comparison Feature
```python
# Add to pages/compare.py
def show_comparison():
    st.header("🔍 Model Comparison")
    
    if st.session_state.get('comparison_results'):
        results = st.session_state.comparison_results
        
        # Create comparison visualization
        for idx, row in results.iterrows():
            with st.expander(f"Prompt {idx + 1}: {row['input'][:50]}..."):
                st.write("**Input:**", row['input'])
                
                cols = st.columns(len(row['outputs']))
                for i, (model, output) in enumerate(row['outputs'].items()):
                    with cols[i]:
                        st.write(f"**{model}:**")
                        st.write(output)
                
                if row.get('expected'):
                    st.write("**Expected:**", row['expected'])
```

### 3.3 Evaluation Metrics
```python
# In utils/metrics.py
from difflib import SequenceMatcher
import re

class EvaluationMetrics:
    @staticmethod
    def calculate_similarity(output: str, expected: str) -> float:
        """Calculate text similarity score"""
        return SequenceMatcher(None, output, expected).ratio()
    
    @staticmethod
    def calculate_metrics(df: pd.DataFrame) -> dict:
        """Calculate various evaluation metrics"""
        metrics = {}
        
        # Only for rows with expected values
        valid_rows = df[df['expected'].notna() & (df['status'] == 'Success')]
        
        if len(valid_rows) > 0:
            similarities = valid_rows.apply(
                lambda row: EvaluationMetrics.calculate_similarity(
                    row['output'], row['expected']
                ), axis=1
            )
            
            metrics['avg_similarity'] = similarities.mean()
            metrics['max_similarity'] = similarities.max()
            metrics['min_similarity'] = similarities.min()
        
        return metrics
```

## Phase 4: Testing & Best Practices

### 4.1 Sample Test CSV
```csv
input,output,expected
"What is the capital of France?","","Paris"
"Explain photosynthesis in simple terms","","Plants use sunlight to make food"
"Write a haiku about coding","",""
"What is 25 * 4?","","100"
"Translate 'Hello World' to Spanish","","Hola Mundo"
```

### 4.2 Error Handling Best Practices
1. Always validate API keys before processing
2. Implement retry logic for transient API errors
3. Handle rate limiting with exponential backoff
4. Provide clear error messages to users
5. Log errors for debugging (but not sensitive data)

### 4.3 Security Best Practices
1. Never commit API keys to version control
2. Use Streamlit's secrets management
3. Validate and sanitize all user inputs
4. Implement request throttling
5. Add user authentication if needed

## Deployment Checklist

- [ ] All Python files are in the repository
- [ ] requirements.txt includes all dependencies
- [ ] No hardcoded API keys in code
- [ ] README.md with clear instructions
- [ ] .gitignore includes .streamlit/secrets.toml
- [ ] Test locally with all three LLM providers
- [ ] Verify CSV upload and validation works
- [ ] Test error handling for API failures
- [ ] Confirm download functionality works
- [ ] Set up secrets in Streamlit Community Cloud

## Sample README.md

```markdown
# LLM Evaluation Tool

A Streamlit application for evaluating and comparing responses from multiple Large Language Models (LLMs).

## Features

- Support for Claude Sonnet 3.7, OpenAI GPT-4o, and Google Gemini 2.5
- Batch processing of prompts from CSV files
- Real-time progress tracking
- Export results with model outputs
- Secure API key management

## Installation

1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run the app: `streamlit run app.py`
4. Enter your API keys on the **API Keys** page

## CSV Format

Your CSV must have these columns:
- `input`: The prompt to send to the LLM
- `output`: Leave empty (will be filled by the model)
- `expected`: Optional expected response for comparison

## Deployment

This app is designed for Streamlit Community Cloud deployment. Add your API keys in the deployment settings.

## License

MIT
```


## 3. Responses to Your Questions

### a) Questions and Ideas for Discussion

Based on my research and the implementation plan above, here are some questions and ideas we could discuss:

**Technical Questions:**
1. **Rate Limiting Strategy**: Would you like to implement more sophisticated rate limiting, such as token bucket algorithms or per-model rate limits?
2. **Caching**: Should we implement response caching to avoid re-processing identical prompts?
3. **Async Processing**: The current design processes prompts sequentially. Would you benefit from parallel processing for faster throughput?
4. **Model Parameters**: Should users be able to adjust temperature, max_tokens, and other model parameters?

**Feature Enhancement Ideas:**
1. **Multi-Model Comparison**: Instead of selecting one model, process the same prompts through all models simultaneously for direct comparison
2. **Evaluation Metrics**: Add automatic evaluation metrics like BLEU scores, semantic similarity, or custom scoring functions
3. **Prompt Templates**: Allow users to define prompt templates with variables that get filled from CSV columns
4. **Response Filtering**: Add options to filter/transform model outputs (e.g., extract JSON, remove certain patterns)
5. **Progress Persistence**: Save progress to allow resuming interrupted batch processes
6. **Cost Estimation**: Show estimated API costs before processing based on token counts

**Architecture Considerations:**
1. **Database Integration**: For larger datasets, consider using a database instead of in-memory processing
2. **Queue System**: For production use, implement a job queue system for better reliability
3. **Authentication**: Add user authentication to track usage and manage API key quotas
4. **Monitoring**: Implement logging and monitoring for API usage and errors

### b) Regarding Working Code Examples

Yes, I would find it very helpful if you could share your working code that does similar things! This would help me:

1. **Understand your coding style and preferences** - I can align the implementation to match your existing patterns
2. **Identify any specific requirements or edge cases** you've already solved
3. **Learn from your deployment experience** - See what works well in production
4. **Avoid reinventing the wheel** - Reuse proven components and patterns
5. **Ensure compatibility** - Make sure the new features integrate well with your existing infrastructure

Specifically, it would be helpful to see:
- How you currently handle API authentication and secrets
- Your approach to error handling and retries
- Any custom UI components or styling preferences
- How you structure multi-page Streamlit apps
- Your CSV processing and validation logic

Please feel free to share any code examples, and I can adapt the implementation plan to better match your existing work while adding the new features you need.

## Summary

I've provided:
1. **Comprehensive documentation** on how Streamlit Community Cloud handles multi-file deployments
2. **A detailed implementation plan** with complete code for your LLM evaluation app
3. **Questions and enhancement ideas** for further discussion

The implementation follows best practices for:
- Modular architecture with separate services and pages
- Secure API key management using Streamlit secrets
- Clean separation of concerns
- Comprehensive error handling
- User-friendly interface with progress tracking

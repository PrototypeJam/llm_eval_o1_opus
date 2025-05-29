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

from abc import ABC, abstractmethod
import streamlit as st
from typing import Optional

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
        self.api_key = st.session_state.api_keys.get("ANTHROPIC_API_KEY", "")

    def validate_api_key(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            from anthropic import Anthropic

            client = Anthropic(api_key=self.api_key)

            response = client.messages.create(
                model="claude-3-sonnet-20240229",
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
        self.api_key = st.session_state.api_keys.get("OPENAI_API_KEY", "")

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
        self.api_key = st.session_state.api_keys.get("GOOGLE_API_KEY", "")

    def validate_api_key(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        try:
            import google.generativeai as genai

            genai.configure(api_key=self.api_key)

            model = genai.GenerativeModel('gemini-1.5-pro')
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

import anthropic

class AnthropicClient:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.messages.create(
            model="claude-3.7-sonnet-latest",
            max_tokens=1000,
            temperature=kwargs.get("temperature", 0.7),
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

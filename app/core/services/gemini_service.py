from ...handlers.gemini_handler import GeminiHandler, Strategy, KeyRotationStrategy

class GeminiService:
    def __init__(self, config_path: str = "config.yaml"):
        self.handler = GeminiHandler(
            config_path=config_path,
            content_strategy=Strategy.ROUND_ROBIN,
            key_strategy=KeyRotationStrategy.SMART_COOLDOWN
        )
    
    def generate_content(self, prompt: str, model_name: str = "gemini-2.0-flash-thinking-exp-01-21") -> str:
        response = self.handler.generate_content(
            prompt=prompt,
            model_name=model_name,
            return_stats=False
        )
        return response.get("text", "No response from model")
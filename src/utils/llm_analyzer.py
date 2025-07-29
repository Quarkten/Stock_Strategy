import requests
import json
import logging
from typing import Dict, Optional, Tuple

class LLMAnalyzer:
    # Model configuration with timeout parameters
    MODEL_CONFIG = {
        "gemma": {"model_name": "gemma3:12b", "timeout": 120},
        "deepseek": {"model_name": "deepseek-r1:8b", "timeout": 120}
    }
    
    VALID_BIASES = {"BULLISH", "BEARISH", "NEUTRAL"}

    def __init__(self, config: Dict[str, str]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    def _call_llm(self, api_url: str, model_key: str, prompt: str) -> Optional[str]:
        """Call LLM API with error handling and timeout"""
        if not api_url:
            self.logger.warning(f"LLM API URL not configured for {model_key}. Skipping call.")
            return None

        config = self.MODEL_CONFIG.get(model_key, {"timeout": 30})
        headers = {'Content-Type': 'application/json'}
        payload = {
            "model": config["model_name"],
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.3, "top_p": 0.9}  # Add generation parameters
        }

        try:
            response = requests.post(
                api_url,
                headers=headers,
                data=json.dumps(payload),
                timeout=config.get("timeout", 60)  # Add a 60-second timeout
            )
            response.raise_for_status()
            return response.json().get('response', '').strip()
        except requests.exceptions.Timeout:
            self.logger.error(f"Timeout calling {model_key} model after {config['timeout']}s")
        except requests.exceptions.RequestException as e:
            self.logger.error(f"API error for {model_key}: {str(e)}")
        except json.JSONDecodeError:
            self.logger.error(f"Invalid JSON response from {model_key}")
        return None

    def get_spy_bias(self, macro_news: str, news_calendar: str) -> str:
        """Determine SPY bias based on macro news and calendar"""
        prompt = f"""
        As a financial analysis tool, classify SPY's short-term bias based ONLY on:
        
        MACRO NEWS:
        {macro_news}
        
        NEWS CALENDAR:
        {news_calendar}
        
        Rules:
        1. No disclaimers
        2. No explanations
        3. Output ONLY one word: BULLISH OR BEARISH
        """

        results = {}
        for model_key in self.MODEL_CONFIG:
            if api_url := self.config.get(f"{model_key}_api"):
                results[model_key] = self._call_llm(api_url, model_key, prompt)

        self.logger.info(f"Model responses: {json.dumps(results, indent=2)}")
        return self._resolve_bias(results)

    def _resolve_bias(self, results: Dict[str, Optional[str]]) -> str:
        """Determine final bias from model responses"""
        valid_responses = {}
        for model, response in results.items():
            if bias := self._extract_bias(response):
                valid_responses[model] = bias

        if not valid_responses:
            self.logger.warning("All models failed - defaulting to NEUTRAL")
            return "NEUTRAL"

        # Get unique valid responses
        unique_biases = set(valid_responses.values())
        
        if len(unique_biases) == 1:
            return next(iter(unique_biases))
        
        # Handle conflicts
        if "NEUTRAL" in unique_biases:
            return "NEUTRAL"
        return max(unique_biases, key=list(valid_responses.values()).count)

    def _extract_bias(self, response: Optional[str]) -> Optional[str]:
        """Extract valid bias from response text"""
        if not response:
            return None
            
        # Find first valid bias keyword in response
        for word in response.upper().split():
            if word in self.VALID_BIASES:
                return word
        return None

    def summarize_text(self, text: str, max_length: int = 200) -> Dict[str, str]:
        """Generate summaries using available models"""
        prompt = f"""
        Create a concise professional summary ({max_length} words max) focusing on key outcomes:
        
        TEXT:
        {text}
        
        Rules:
        1. Omit disclaimers
        2. Avoid conversational filler
        3. Include only factual insights
        """
        
        summaries = {}
        for model_key in self.MODEL_CONFIG:
            if api_url := self.config.get(f"{model_key}_api"):
                if summary := self._call_llm(api_url, model_key, prompt):
                    summaries[model_key] = summary
        
        # Add placeholders for unavailable models
        for model_key in self.MODEL_CONFIG:
            if model_key not in summaries:
                summaries[model_key] = "(Summary not available)"
                
        return summaries
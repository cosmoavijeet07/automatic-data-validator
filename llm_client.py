"""
LLM client module for handling API interactions with language models
"""

import json
import time
from typing import Dict, Any, List, Optional
from config import MODELS, OPENAI_API_KEY, MAX_RETRIES, RETRY_DELAY
import streamlit as st

class LLMClient:
    """Client for interacting with Large Language Models"""
    
    def __init__(self, model_name: str = "GPT 4.1"):
        self.model_name = model_name
        self.model_id = MODELS.get(model_name, MODELS["GPT 4.1"])
        self.max_retries = MAX_RETRIES
        self.retry_delay = RETRY_DELAY
        self.client = None
        
        # Initialize OpenAI client
        if OPENAI_API_KEY:
            try:
                import openai
                self.client = openai.OpenAI(api_key=OPENAI_API_KEY)
            except ImportError:
                st.error("OpenAI library not installed. Please run: pip install openai")
                self.client = None
            except Exception as e:
                st.error(f"Failed to initialize LLM client: {str(e)}")
                self.client = None
        else:
            st.warning("OpenAI API key not found. Please set OPENAI_API_KEY environment variable.")
            self.client = None
    
    def get_completion(self, prompt: str, system_prompt: str = None, 
                      temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """
        Get completion from LLM
        
        Args:
            prompt: User prompt
            system_prompt: System prompt (optional)
            temperature: Response randomness (0-2)
            max_tokens: Maximum response length
            
        Returns:
            LLM response as string
        """
        if not self.client:
            return "LLM client not available. Please check your API key."
        
        messages = []
        
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        messages.append({"role": "user", "content": prompt})
        
        for attempt in range(self.max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    timeout=60
                )
                
                return response.choices[0].message.content.strip()
                
            except Exception as e:
                error_str = str(e)
                
                # Handle rate limits
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                        continue
                    else:
                        return f"Rate limit exceeded. Please try again later. Error: {error_str}"
                
                # Handle API errors
                elif "api" in error_str.lower() or "400" in error_str or "401" in error_str:
                    if "api_key" in error_str.lower() or "401" in error_str:
                        return "Invalid API key. Please check your OPENAI_API_KEY environment variable."
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return f"API error: {error_str}"
                
                # Handle other errors
                else:
                    if attempt < self.max_retries - 1:
                        time.sleep(self.retry_delay)
                        continue
                    else:
                        return f"Unexpected error: {error_str}"
        
        return "Failed to get completion after maximum retries"
    
    def get_json_completion(self, prompt: str, system_prompt: str = None) -> Dict[str, Any]:
        """
        Get JSON completion from LLM
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            
        Returns:
            Parsed JSON response
        """
        # Add JSON instruction to prompt
        json_prompt = f"""
        {prompt}
        
        Please respond with valid JSON only. Do not include any text outside the JSON structure.
        Do not include markdown code blocks or backticks.
        """
        
        response = self.get_completion(json_prompt, system_prompt)
        
        # Check for error messages
        if "LLM client not available" in response or "error" in response.lower():
            return {
                "error": response,
                "raw_response": response
            }
        
        try:
            # Try to parse as JSON
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from response
            import re
            
            # Remove markdown code blocks if present
            response = re.sub(r'```json\s*', '', response)
            response = re.sub(r'```\s*', '', response)
            
            # Try to find JSON object
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                try:
                    return json.loads(json_match.group())
                except json.JSONDecodeError:
                    pass
            
            # If JSON parsing fails, return structured response
            return {
                "error": "Failed to parse JSON response",
                "raw_response": response
            }
    
    def generate_code(self, prompt: str, language: str = "python") -> str:
        """
        Generate code using LLM
        
        Args:
            prompt: Code generation prompt
            language: Programming language
            
        Returns:
            Generated code
        """
        if not self.client:
            return "# LLM client not available. Please check your API key."
        
        system_prompt = f"""
        You are an expert {language} programmer. Generate clean, efficient, and well-commented code.
        Follow best practices and include error handling where appropriate.
        Return only the code without any markdown formatting or code blocks.
        Do not include backticks or language identifiers.
        """
        
        code_prompt = f"""
        Generate {language} code for the following requirement:
        
        {prompt}
        
        Requirements:
        - Use only standard libraries and pandas/numpy for data operations
        - Include proper error handling
        - Add comments for complex operations
        - Ensure code is production-ready
        - Return only the raw code, no markdown or explanations
        """
        
        response = self.get_completion(code_prompt, system_prompt, temperature=0.1)
        
        # Clean up response if it contains markdown
        import re
        response = re.sub(r'```python\s*', '', response)
        response = re.sub(r'```\s*', '', response)
        
        return response
    
    def summarize_strategy(self, code: str, context: str = "") -> str:
        """
        Generate a summary of the strategy implemented in code
        
        Args:
            code: The code to summarize
            context: Additional context about the task
            
        Returns:
            Strategy summary (max 100 words)
        """
        if not self.client:
            return "LLM client not available for summary generation."
        
        prompt = f"""
        Analyze the following code and provide a concise summary of the data cleaning strategy:
        
        Context: {context}
        
        Code:
        {code[:2000]}  # Limit code length to avoid token limits
        
        Provide a summary in less than 100 words focusing on:
        1. Main cleaning operations performed
        2. Business impact of these operations
        3. Key issues addressed
        
        Keep it simple and non-technical for business users.
        """
        
        return self.get_completion(prompt, temperature=0.2, max_tokens=150)
    
    def validate_code(self, code: str) -> Dict[str, Any]:
        """
        Validate generated code for safety and correctness
        
        Args:
            code: Code to validate
            
        Returns:
            Validation results
        """
        if not self.client:
            return {
                "is_valid": False,
                "issues": ["LLM client not available"],
                "suggestions": [],
                "risk_level": "unknown"
            }
        
        prompt = f"""
        Analyze the following Python code for potential issues:
        
        {code[:3000]}  # Limit for token management
        
        Check for:
        1. Syntax errors
        2. Security issues (dangerous functions)
        3. Logic errors
        4. Best practice violations
        5. Missing imports
        
        Respond with JSON in this format:
        {{
            "is_valid": true/false,
            "issues": ["list of issues found"],
            "suggestions": ["list of improvement suggestions"],
            "risk_level": "low/medium/high"
        }}
        """
        
        return self.get_json_completion(prompt)
    
    def enhance_prompt(self, base_prompt: str, context: Dict[str, Any]) -> str:
        """
        Enhance a base prompt with additional context
        
        Args:
            base_prompt: Base prompt template
            context: Additional context to include
            
        Returns:
            Enhanced prompt
        """
        enhanced_prompt = base_prompt
        
        # Add data context
        if 'schema' in context:
            enhanced_prompt += f"\n\nData Schema:\n{json.dumps(context['schema'], indent=2, default=str)}"
        
        if 'shape' in context:
            enhanced_prompt += f"\n\nData Shape: {context['shape']}"
        
        if 'quality_issues' in context:
            enhanced_prompt += f"\n\nQuality Issues:\n{json.dumps(context['quality_issues'], indent=2, default=str)}"
        
        if 'user_instructions' in context:
            enhanced_prompt += f"\n\nUser Instructions: {context['user_instructions']}"
        
        if 'previous_errors' in context:
            enhanced_prompt += f"\n\nPrevious Errors: {context['previous_errors']}"
        
        return enhanced_prompt
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model"""
        return {
            "name": self.model_name,
            "id": self.model_id,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "client_available": self.client is not None
        }
    
    def switch_model(self, model_name: str):
        """Switch to a different model"""
        if model_name in MODELS:
            self.model_name = model_name
            self.model_id = MODELS[model_name]
        else:
            raise ValueError(f"Model {model_name} not available. Available models: {list(MODELS.keys())}")
    
    def estimate_tokens(self, text: str) -> int:
        """Rough estimation of token count"""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
    
    def chunk_large_prompt(self, prompt: str, max_tokens: int = 4000) -> List[str]:
        """Split large prompts into chunks"""
        estimated_tokens = self.estimate_tokens(prompt)
        
        if estimated_tokens <= max_tokens:
            return [prompt]
        
        # Split into chunks
        chunks = []
        lines = prompt.split('\n')
        current_chunk = ""
        
        for line in lines:
            test_chunk = current_chunk + "\n" + line if current_chunk else line
            
            if self.estimate_tokens(test_chunk) > max_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = line
                else:
                    # Line itself is too long, split by sentences
                    sentences = line.split('. ')
                    for sentence in sentences:
                        if self.estimate_tokens(current_chunk + sentence) > max_tokens:
                            if current_chunk:
                                chunks.append(current_chunk)
                            current_chunk = sentence
                        else:
                            current_chunk += sentence + ". " if current_chunk else sentence
            else:
                current_chunk = test_chunk
        
        if current_chunk:
            chunks.append(current_chunk)
        
        return chunks
    
    def test_connection(self) -> bool:
        """Test if the LLM connection is working"""
        if not self.client:
            return False
        
        try:
            response = self.get_completion("Say 'Hello'", temperature=0)
            return "Hello" in response or "hello" in response.lower()
        except:
            return False
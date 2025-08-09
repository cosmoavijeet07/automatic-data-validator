"""
LLM Client Module for OpenAI Integration
"""
import json
import logging
from typing import Dict, Any, List, Optional
import openai
from openai import OpenAI
import time
from config import OPENAI_API_KEY, MAX_RETRIES, RETRY_DELAY

class LLMClient:
    def __init__(self, api_key: str = None, model: str = "gpt-4-turbo-preview"):
        """Initialize LLM client"""
        self.api_key = api_key or OPENAI_API_KEY
        self.model = model
        self.client = OpenAI(api_key=self.api_key) if self.api_key else None
        self.conversation_history = []
        
    def generate_schema_cleaning_code(
        self,
        schema: Dict[str, str],
        user_instructions: str = "",
        issues: Dict[str, Any] = None
    ) -> Dict[str, str]:
        """Generate schema cleaning code"""
        prompt = f"""
        You are a data cleaning expert. Generate Python code to clean a dataset based on the schema and instructions.
        
        Current Schema:
        {json.dumps(schema, indent=2)}
        
        User Instructions:
        {user_instructions}
        
        {"Issues Found:" + json.dumps(issues, indent=2) if issues else ""}
        
        Generate Python code that:
        1. Handles data type conversions safely
        2. Deals with missing values appropriately
        3. Follows pandas best practices
        4. Includes error handling
        5. Preserves data integrity
        
        Return ONLY the Python code without explanations.
        Use 'df' as the DataFrame variable name.
        """
        
        response = self._call_api(prompt)
        code = self._extract_code(response)
        
        return {
            "code": code,
            "prompt": prompt,
            "response": response
        }
    
    def generate_quality_analysis_code(
        self,
        schema: Dict[str, str],
        initial_report: Dict[str, Any]
    ) -> Dict[str, str]:
        """Generate deeper quality analysis code"""
        prompt = f"""
        You are a data quality expert. Generate Python code for deeper quality analysis.
        
        Current Schema:
        {json.dumps(schema, indent=2)}
        
        Initial Quality Report:
        {json.dumps(initial_report, indent=2)}
        
        Generate Python code that performs deeper analysis:
        1. Check for business logic inconsistencies
        2. Detect anomalies and patterns
        3. Validate data relationships
        4. Check for data integrity issues
        5. Identify potential data quality problems
        6. Look for statistical anomalies
        
        Return ONLY the Python code.
        Use 'df' as the DataFrame variable name.
        Store results in a dictionary called 'quality_issues'.
        """
        
        response = self._call_api(prompt)
        code = self._extract_code(response)
        
        return {
            "code": code,
            "prompt": prompt,
            "response": response
        }
    
    def generate_correction_code(
        self,
        schema: Dict[str, str],
        issues: Dict[str, Any],
        user_instructions: str = ""
    ) -> Dict[str, str]:
        """Generate data correction code"""
        prompt = f"""
        You are a data correction expert. Generate Python code to fix data issues.
        
        Current Schema:
        {json.dumps(schema, indent=2)}
        
        Issues to Fix:
        {json.dumps(issues, indent=2)}
        
        Additional Instructions:
        {user_instructions}
        
        Generate Python code that:
        1. Fixes identified issues
        2. Maintains data consistency
        3. Logs all changes made
        4. Handles edge cases
        5. Preserves valid data
        
        Return ONLY the Python code.
        Use 'df' as the DataFrame variable name.
        Create a 'corrections_log' list to track changes.
        """
        
        response = self._call_api(prompt)
        code = self._extract_code(response)
        
        return {
            "code": code,
            "prompt": prompt,
            "response": response
        }
    
    def generate_text_cleaning_code(
        self,
        sample_text: str,
        user_instructions: str
    ) -> Dict[str, str]:
        """Generate text cleaning code"""
        prompt = f"""
        You are a text processing expert. Generate Python code for text cleaning.
        
        Sample Text:
        {sample_text[:500]}...
        
        User Instructions:
        {user_instructions}
        
        Generate Python code that:
        1. Implements text preprocessing (lowercase, remove punctuation, etc.)
        2. Removes stop words
        3. Applies lemmatization or stemming
        4. Handles special characters and encoding
        5. Preserves important information
        
        Return ONLY the Python code.
        Use 'text_data' as the input variable.
        Import necessary NLTK components.
        """
        
        response = self._call_api(prompt)
        code = self._extract_code(response)
        
        return {
            "code": code,
            "prompt": prompt,
            "response": response
        }
    
    def generate_summary(self, code: str) -> str:
        """Generate natural language summary of code"""
        prompt = f"""
        Summarize the following data cleaning code in plain English.
        Keep it under 100 words and focus on what the code does, not how.
        
        Code:
        {code}
        
        Provide a clear, concise summary for non-technical users.
        """
        
        response = self._call_api(prompt, max_tokens=150)
        return response.strip()
    
    def generate_pipeline_code(self, code_history: List[str]) -> str:
        """Generate final pipeline code from history"""
        prompt = f"""
        You are a Python expert. Combine the following code snippets into a single, clean pipeline function.
        
        Code History:
        {chr(10).join([f"Step {i+1}:{chr(10)}{code}" for i, code in enumerate(code_history)])}
        
        Create a complete Python script that:
        1. Combines all cleaning steps in order
        2. Includes proper imports
        3. Has a main function that accepts input/output file paths
        4. Includes error handling
        5. Provides progress updates
        6. Can be run standalone
        
        Return ONLY the complete Python code.
        """
        
        response = self._call_api(prompt)
        code = self._extract_code(response)
        return code
    
    def fix_error(
        self,
        original_code: str,
        error_message: str,
        user_feedback: str = ""
    ) -> Dict[str, str]:
        """Fix code based on error"""
        prompt = f"""
        Fix the following Python code that produced an error.
        
        Original Code:
        {original_code}
        
        Error Message:
        {error_message}
        
        User Feedback:
        {user_feedback}
        
        Generate corrected Python code that:
        1. Fixes the error
        2. Maintains the original intent
        3. Includes better error handling
        
        Return ONLY the corrected Python code.
        """
        
        response = self._call_api(prompt)
        code = self._extract_code(response)
        
        return {
            "code": code,
            "prompt": prompt,
            "response": response
        }
    
    def _call_api(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call OpenAI API with retry logic"""
        if not self.client:
            logging.error("OpenAI client not initialized. Please set API key.")
            return "# Error: OpenAI API key not configured"
        SUPPORTS_TEMPERATURE = {"gpt-4", "gpt-4o", "gpt-4.1-2025-04-14"}
        for attempt in range(MAX_RETRIES):
            try:
                if  self.model in SUPPORTS_TEMPERATURE:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a data cleaning and processing expert."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=max_tokens,
                        temperature=0.1
                    )
                    return response.choices[0].message.content
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=[
                            {"role": "system", "content": "You are a data cleaning and processing expert."},
                            {"role": "user", "content": prompt}
                        ],
                    )
                    return response.choices[0].message.content
            except Exception as e:
                logging.error(f"API call failed (attempt {attempt + 1}): {e}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY * (attempt + 1))
                else:
                    return f"# Error after {MAX_RETRIES} attempts: {str(e)}"
        
        return "# Error: Failed to generate code"
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from response"""
        # Remove markdown code blocks if present
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response
        
        return code.strip()
    
    def analyze_with_natural_language(self, query: str, context: Dict[str, Any]) -> str:
        """Analyze data using natural language query"""
        prompt = f"""
        Analyze the following data context and answer the user's query.
        
        Data Context:
        {json.dumps(context, indent=2)}
        
        User Query:
        {query}
        
        Provide a comprehensive analysis and actionable insights.
        """
        
        response = self._call_api(prompt)
        return response
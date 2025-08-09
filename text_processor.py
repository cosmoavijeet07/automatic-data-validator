"""
Text processor module for handling text data cleaning and vector database operations
"""

import re
import string
import nltk
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
from pathlib import Path
from config import VECTOR_DB_CHUNK_SIZE, VECTOR_DB_OVERLAP, DEFAULT_EMBEDDING_MODEL

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer

class TextProcessor:
    """Handles text data processing and cleaning operations"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.chunk_size = VECTOR_DB_CHUNK_SIZE
        self.chunk_overlap = VECTOR_DB_OVERLAP
        self.embedding_model = DEFAULT_EMBEDDING_MODEL
        
        # Initialize NLTK components
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Common text cleaning patterns
        self.cleaning_patterns = {
            'urls': re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'),
            'emails': re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
            'phone_numbers': re.compile(r'(\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}'),
            'special_chars': re.compile(r'[^a-zA-Z0-9\s]'),
            'extra_whitespace': re.compile(r'\s+'),
            'html_tags': re.compile(r'<[^<]+?>'),
            'numbers': re.compile(r'\b\d+\b')
        }
    
    def analyze_text_data(self, text_data: str) -> Dict[str, Any]:
        """
        Analyze text data and provide statistics
        
        Args:
            text_data: Raw text data to analyze
            
        Returns:
            Dictionary containing text analysis results
        """
        analysis = {
            'basic_stats': self._get_basic_text_stats(text_data),
            'content_analysis': self._analyze_content(text_data),
            'quality_issues': self._identify_quality_issues(text_data),
            'cleaning_suggestions': []
        }
        
        # Generate cleaning suggestions based on analysis
        analysis['cleaning_suggestions'] = self._generate_cleaning_suggestions(analysis)
        
        return analysis
    
    def _get_basic_text_stats(self, text: str) -> Dict[str, Any]:
        """Get basic text statistics"""
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())
        
        return {
            'total_characters': len(text),
            'total_words': len(words),
            'total_sentences': len(sentences),
            'total_lines': len(text.splitlines()),
            'average_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'average_chars_per_word': len(text) / len(words) if words else 0,
            'unique_words': len(set(words)),
            'vocabulary_richness': len(set(words)) / len(words) if words else 0
        }
    
    def _analyze_content(self, text: str) -> Dict[str, Any]:
        """Analyze content patterns in text"""
        content_analysis = {
            'contains_urls': bool(self.cleaning_patterns['urls'].search(text)),
            'contains_emails': bool(self.cleaning_patterns['emails'].search(text)),
            'contains_phone_numbers': bool(self.cleaning_patterns['phone_numbers'].search(text)),
            'contains_html': bool(self.cleaning_patterns['html_tags'].search(text)),
            'number_count': len(self.cleaning_patterns['numbers'].findall(text)),
            'special_char_ratio': self._calculate_special_char_ratio(text),
            'uppercase_ratio': self._calculate_uppercase_ratio(text),
            'encoding_issues': self._detect_encoding_issues(text)
        }
        
        # Language detection (simplified)
        content_analysis['likely_language'] = self._detect_language(text)
        
        return content_analysis
    
    def _identify_quality_issues(self, text: str) -> List[Dict[str, Any]]:
        """Identify quality issues in text data"""
        issues = []
        
        # Check for excessive whitespace
        if len(re.findall(r'\s{3,}', text)) > 0:
            issues.append({
                'type': 'excessive_whitespace',
                'severity': 'medium',
                'description': 'Text contains excessive whitespace'
            })
        
        # Check for mixed encodings
        try:
            text.encode('ascii')
        except UnicodeEncodeError:
            issues.append({
                'type': 'encoding_issues',
                'severity': 'medium',
                'description': 'Text contains non-ASCII characters that may need cleaning'
            })
        
        # Check for very long lines
        long_lines = [line for line in text.splitlines() if len(line) > 1000]
        if long_lines:
            issues.append({
                'type': 'long_lines',
                'severity': 'low',
                'description': f'Found {len(long_lines)} lines longer than 1000 characters'
            })
        
        # Check for inconsistent line endings
        if '\r\n' in text and '\n' in text.replace('\r\n', ''):
            issues.append({
                'type': 'mixed_line_endings',
                'severity': 'low',
                'description': 'Text has mixed line ending formats'
            })
        
        return issues
    
    def _calculate_special_char_ratio(self, text: str) -> float:
        """Calculate ratio of special characters"""
        special_chars = sum(1 for char in text if char in string.punctuation)
        return special_chars / len(text) if text else 0
    
    def _calculate_uppercase_ratio(self, text: str) -> float:
        """Calculate ratio of uppercase characters"""
        uppercase_chars = sum(1 for char in text if char.isupper())
        return uppercase_chars / len(text) if text else 0
    
    def _detect_encoding_issues(self, text: str) -> bool:
        """Detect potential encoding issues"""
        # Look for common encoding artifacts
        encoding_artifacts = ['â€™', 'â€œ', 'â€', 'Ã©', 'Ã¡', 'Ã­', 'Ã³']
        return any(artifact in text for artifact in encoding_artifacts)
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection based on common words"""
        words = word_tokenize(text.lower())
        
        # English common words
        english_words = {'the', 'and', 'is', 'in', 'to', 'of', 'a', 'that', 'it', 'with', 'for', 'as', 'was', 'on', 'are'}
        english_count = sum(1 for word in words if word in english_words)
        
        if english_count / len(words) > 0.1 if words else 0:
            return 'english'
        else:
            return 'unknown'
    
    def _generate_cleaning_suggestions(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate cleaning suggestions based on analysis"""
        suggestions = []
        
        content = analysis['content_analysis']
        issues = analysis['quality_issues']
        
        if content['contains_html']:
            suggestions.append("Remove HTML tags")
        
        if content['contains_urls']:
            suggestions.append("Remove or extract URLs")
        
        if content['contains_emails']:
            suggestions.append("Remove or extract email addresses")
        
        if content['special_char_ratio'] > 0.1:
            suggestions.append("Clean excessive special characters")
        
        if any(issue['type'] == 'excessive_whitespace' for issue in issues):
            suggestions.append("Normalize whitespace")
        
        if content['encoding_issues']:
            suggestions.append("Fix encoding issues")
        
        suggestions.extend([
            "Convert to lowercase",
            "Remove stop words",
            "Apply lemmatization or stemming",
            "Tokenize text"
        ])
        
        return suggestions
    
    def generate_cleaning_code(self, text_data: str, user_instructions: str = "") -> Tuple[str, str]:
        """
        Generate text cleaning code using LLM
        
        Args:
            text_data: Sample text data
            user_instructions: Additional user instructions
            
        Returns:
            Tuple of (cleaning_code, strategy_summary)
        """
        try:
            # Analyze the text
            analysis = self.analyze_text_data(text_data[:5000])  # Use sample for analysis
            
            # Prepare context for LLM
            context = {
                'text_sample': text_data[:1000],  # First 1000 characters
                'analysis': analysis,
                'user_instructions': user_instructions
            }
            
            prompt = f"""
            Generate Python code to clean the following text data:
            
            Text Sample:
            {context['text_sample']}
            
            Analysis Results:
            {analysis}
            
            User Instructions: {user_instructions}
            
            Create a function called 'clean_text' that:
            1. Takes text as input parameter
            2. Applies appropriate cleaning operations
            3. Returns cleaned text
            4. Includes proper error handling
            5. Uses libraries: re, nltk, string
            
            Address the identified issues and apply common text preprocessing steps.
            Include detailed comments explaining each step.
            """
            
            cleaning_code = self.llm_client.generate_code(prompt)
            
            # Generate strategy summary
            strategy_summary = self.llm_client.summarize_strategy(
                cleaning_code, 
                f"Text cleaning for {len(text_data)} characters with {len(analysis['quality_issues'])} quality issues"
            )
            
            return cleaning_code, strategy_summary
            
        except Exception as e:
            raise ValueError(f"Failed to generate text cleaning code: {str(e)}")
    
    def execute_cleaning_code(self, text_data: str, code: str) -> Tuple[str, Dict[str, Any]]:
        """
        Execute text cleaning code
        
        Args:
            text_data: Text to clean
            code: Cleaning code to execute
            
        Returns:
            Tuple of (cleaned_text, execution_log)
        """
        execution_log = {
            'success': False,
            'original_length': len(text_data),
            'final_length': None,
            'execution_time': None,
            'errors': [],
            'output': []
        }
        
        try:
            import time
            from io import StringIO
            import sys
            
            start_time = time.time()
            
            # Capture output
            captured_output = StringIO()
            sys.stdout = captured_output
            
            # Create execution environment
            exec_globals = {
                're': re,
                'string': string,
                'nltk': nltk,
                'word_tokenize': word_tokenize,
                'sent_tokenize': sent_tokenize,
                'stopwords': stopwords,
                'WordNetLemmatizer': WordNetLemmatizer,
                'PorterStemmer': PorterStemmer,
                '__builtins__': {}
            }
            
            # Execute the code
            exec(code, exec_globals)
            
            # Get the cleaning function
            if 'clean_text' not in exec_globals:
                raise ValueError("Generated code must define a 'clean_text' function")
            
            clean_text_func = exec_globals['clean_text']
            
            # Execute cleaning
            cleaned_text = clean_text_func(text_data)
            
            # Restore stdout
            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()
            
            execution_log.update({
                'success': True,
                'final_length': len(cleaned_text),
                'execution_time': time.time() - start_time,
                'output': output.split('\n') if output else [],
                'length_reduction': len(text_data) - len(cleaned_text)
            })
            
            return cleaned_text, execution_log
            
        except Exception as e:
            sys.stdout = sys.__stdout__
            
            execution_log['errors'].append({
                'type': type(e).__name__,
                'message': str(e)
            })
            
            return text_data, execution_log
    
    def chunk_text_for_vector_db(self, text: str, chunk_size: int = None, 
                                 overlap: int = None) -> List[Dict[str, Any]]:
        """
        Chunk text for vector database storage
        
        Args:
            text: Text to chunk
            chunk_size: Maximum characters per chunk
            overlap: Overlap between chunks
            
        Returns:
            List of chunk dictionaries
        """
        chunk_size = chunk_size or self.chunk_size
        overlap = overlap or self.chunk_overlap
        
        chunks = []
        sentences = sent_tokenize(text)
        
        current_chunk = ""
        current_length = 0
        chunk_id = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            # If adding this sentence would exceed chunk size
            if current_length + sentence_length > chunk_size and current_chunk:
                # Save current chunk
                chunks.append({
                    'id': chunk_id,
                    'text': current_chunk.strip(),
                    'length': current_length,
                    'start_sentence': chunk_id * (chunk_size // 100),  # Approximate
                    'metadata': {
                        'chunk_size': current_length,
                        'sentence_count': len(sent_tokenize(current_chunk))
                    }
                })
                
                chunk_id += 1
                
                # Start new chunk with overlap
                if overlap > 0:
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else current_chunk
                    current_chunk = overlap_text + " " + sentence
                    current_length = len(current_chunk)
                else:
                    current_chunk = sentence
                    current_length = sentence_length
            else:
                # Add sentence to current chunk
                if current_chunk:
                    current_chunk += " " + sentence
                else:
                    current_chunk = sentence
                current_length += sentence_length
        
        # Add final chunk if it has content
        if current_chunk.strip():
            chunks.append({
                'id': chunk_id,
                'text': current_chunk.strip(),
                'length': current_length,
                'start_sentence': chunk_id * (chunk_size // 100),
                'metadata': {
                    'chunk_size': current_length,
                    'sentence_count': len(sent_tokenize(current_chunk))
                }
            })
        
        return chunks
    
    def create_vector_database(self, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create a simple vector database representation
        Note: This is a simplified implementation. In production, you'd use
        libraries like langchain, llamaindex, or chromadb
        
        Args:
            chunks: List of text chunks
            
        Returns:
            Vector database information
        """
        vector_db_info = {
            'total_chunks': len(chunks),
            'total_text_length': sum(chunk['length'] for chunk in chunks),
            'average_chunk_size': sum(chunk['length'] for chunk in chunks) / len(chunks) if chunks else 0,
            'chunks_metadata': [chunk['metadata'] for chunk in chunks],
            'embedding_model': self.embedding_model,
            'created_at': pd.Timestamp.now().isoformat()
        }
        
        # In a real implementation, you would:
        # 1. Generate embeddings for each chunk using the embedding model
        # 2. Store embeddings in a vector database (e.g., Pinecone, Weaviate, ChromaDB)
        # 3. Create indexes for efficient similarity search
        
        # For now, we'll store the chunks as structured data
        vector_db_info['chunks'] = chunks
        
        return vector_db_info
    
    def get_text_cleaning_templates(self) -> Dict[str, str]:
        """Get predefined text cleaning templates"""
        templates = {
            'basic_cleaning': '''
def clean_text(text):
    """Basic text cleaning"""
    import re
    import string
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove HTML tags
    text = re.sub(r'<[^<]+?>', '', text)
    
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    
    # Remove email addresses
    text = re.sub(r'\\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Z|a-z]{2,}\\b', '', text)
    
    # Remove special characters
    text = re.sub(r'[^a-zA-Z0-9\\s]', '', text)
    
    # Remove extra whitespace
    text = re.sub(r'\\s+', ' ', text).strip()
    
    return text
            ''',
            
            'advanced_nlp_cleaning': '''
def clean_text(text):
    """Advanced NLP text cleaning"""
    import re
    import nltk
    from nltk.corpus import stopwords
    from nltk.tokenize import word_tokenize
    from nltk.stem import WordNetLemmatizer
    
    # Initialize
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    
    # Basic cleaning
    text = text.lower()
    text = re.sub(r'<[^<]+?>', '', text)  # Remove HTML
    text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z\\s]', '', text)  # Keep only letters and spaces
    text = re.sub(r'\\s+', ' ', text).strip()  # Normalize whitespace
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Remove stop words and lemmatize
    cleaned_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            cleaned_tokens.append(lemmatizer.lemmatize(token))
    
    return ' '.join(cleaned_tokens)
            ''',
            
            'document_processing': '''
def clean_text(text):
    """Document-specific text cleaning"""
    import re
    
    # Fix common encoding issues
    text = text.replace('â€™', "'")
    text = text.replace('â€œ', '"')
    text = text.replace('â€', '"')
    text = text.replace('â€¢', '•')
    
    # Normalize line breaks
    text = text.replace('\\r\\n', '\\n')
    text = text.replace('\\r', '\\n')
    
    # Remove excessive whitespace but preserve paragraph structure
    text = re.sub(r'[ \\t]+', ' ', text)  # Multiple spaces/tabs to single space
    text = re.sub(r'\\n{3,}', '\\n\\n', text)  # Multiple newlines to double newline
    
    # Remove page numbers and headers/footers (common patterns)
    text = re.sub(r'\\n\\s*\\d+\\s*\\n', '\\n', text)  # Standalone numbers
    text = re.sub(r'\\n\\s*Page \\d+.*\\n', '\\n', text)  # Page headers
    
    # Clean up start and end
    text = text.strip()
    
    return text
            '''
        }
        
        return templates
    
    def apply_template_cleaning(self, text: str, template_name: str) -> str:
        """Apply a predefined cleaning template"""
        templates = self.get_text_cleaning_templates()
        
        if template_name not in templates:
            raise ValueError(f"Template '{template_name}' not found")
        
        try:
            # Execute the template
            exec_globals = {
                're': re,
                'string': string,
                'nltk': nltk,
                'word_tokenize': word_tokenize,
                'sent_tokenize': sent_tokenize,
                'stopwords': stopwords,
                'WordNetLemmatizer': WordNetLemmatizer,
                'PorterStemmer': PorterStemmer
            }
            
            exec(templates[template_name], exec_globals)
            
            # Get the function
            clean_text_func = exec_globals['clean_text']
            return clean_text_func(text)
            
        except Exception as e:
            raise ValueError(f"Failed to apply template '{template_name}': {str(e)}")
    
    def get_cleaning_suggestions(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get automated cleaning suggestions based on text analysis"""
        suggestions = []
        
        content = analysis['content_analysis']
        issues = analysis['quality_issues']
        
        # HTML cleaning
        if content['contains_html']:
            suggestions.append({
                'type': 'html_removal',
                'priority': 'high',
                'description': 'Remove HTML tags from text',
                'template': 'basic_cleaning',
                'impact': 'Removes formatting artifacts and improves text quality'
            })
        
        # URL cleaning
        if content['contains_urls']:
            suggestions.append({
                'type': 'url_removal',
                'priority': 'medium',
                'description': 'Remove or extract URLs',
                'template': 'basic_cleaning',
                'impact': 'Reduces noise and focuses on actual content'
            })
        
        # Encoding issues
        if content['encoding_issues']:
            suggestions.append({
                'type': 'encoding_fix',
                'priority': 'high',
                'description': 'Fix text encoding issues',
                'template': 'document_processing',
                'impact': 'Improves text readability and processing accuracy'
            })
        
        # Advanced NLP processing
        if content['likely_language'] == 'english':
            suggestions.append({
                'type': 'nlp_processing',
                'priority': 'medium',
                'description': 'Apply NLP preprocessing (tokenization, lemmatization, stop word removal)',
                'template': 'advanced_nlp_cleaning',
                'impact': 'Prepares text for machine learning and analysis'
            })
        
        # Whitespace normalization
        if any(issue['type'] == 'excessive_whitespace' for issue in issues):
            suggestions.append({
                'type': 'whitespace_normalization',
                'priority': 'low',
                'description': 'Normalize whitespace and line breaks',
                'template': 'document_processing',
                'impact': 'Improves text consistency and readability'
            })
        
        # Sort by priority
        priority_order = {'high': 3, 'medium': 2, 'low': 1}
        suggestions.sort(key=lambda x: priority_order[x['priority']], reverse=True)
        
        return suggestions
    
    def generate_text_report(self, original_text: str, cleaned_text: str, 
                           execution_log: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive text processing report"""
        report = {
            'summary': {
                'success': execution_log['success'],
                'original_length': execution_log['original_length'],
                'final_length': execution_log['final_length'],
                'length_reduction': execution_log.get('length_reduction', 0),
                'reduction_percentage': 0
            },
            'text_changes': {},
            'quality_improvements': {},
            'execution_details': execution_log
        }
        
        if execution_log['success']:
            # Calculate reduction percentage
            if execution_log['original_length'] > 0:
                reduction_pct = (execution_log.get('length_reduction', 0) / execution_log['original_length']) * 100
                report['summary']['reduction_percentage'] = reduction_pct
            
            # Analyze text changes
            report['text_changes'] = self._analyze_text_changes(original_text, cleaned_text)
            
            # Quality improvements
            original_analysis = self.analyze_text_data(original_text)
            cleaned_analysis = self.analyze_text_data(cleaned_text)
            report['quality_improvements'] = self._compare_text_quality(original_analysis, cleaned_analysis)
        
        return report
    
    def _analyze_text_changes(self, original: str, cleaned: str) -> Dict[str, Any]:
        """Analyze changes between original and cleaned text"""
        original_words = word_tokenize(original.lower())
        cleaned_words = word_tokenize(cleaned.lower())
        
        return {
            'character_change': len(cleaned) - len(original),
            'word_change': len(cleaned_words) - len(original_words),
            'sentence_change': len(sent_tokenize(cleaned)) - len(sent_tokenize(original)),
            'vocabulary_change': len(set(cleaned_words)) - len(set(original_words)),
            'removed_elements': {
                'urls_removed': len(self.cleaning_patterns['urls'].findall(original)) - len(self.cleaning_patterns['urls'].findall(cleaned)),
                'emails_removed': len(self.cleaning_patterns['emails'].findall(original)) - len(self.cleaning_patterns['emails'].findall(cleaned)),
                'html_tags_removed': len(self.cleaning_patterns['html_tags'].findall(original)) - len(self.cleaning_patterns['html_tags'].findall(cleaned))
            }
        }
    
    def _compare_text_quality(self, original_analysis: Dict[str, Any], 
                            cleaned_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Compare text quality before and after cleaning"""
        return {
            'vocabulary_richness': {
                'original': original_analysis['basic_stats']['vocabulary_richness'],
                'cleaned': cleaned_analysis['basic_stats']['vocabulary_richness'],
                'improvement': cleaned_analysis['basic_stats']['vocabulary_richness'] - original_analysis['basic_stats']['vocabulary_richness']
            },
            'average_word_length': {
                'original': original_analysis['basic_stats']['average_chars_per_word'],
                'cleaned': cleaned_analysis['basic_stats']['average_chars_per_word'],
                'improvement': cleaned_analysis['basic_stats']['average_chars_per_word'] - original_analysis['basic_stats']['average_chars_per_word']
            },
            'quality_issues_resolved': {
                'original_issues': len(original_analysis['quality_issues']),
                'remaining_issues': len(cleaned_analysis['quality_issues']),
                'issues_resolved': len(original_analysis['quality_issues']) - len(cleaned_analysis['quality_issues'])
            },
            'content_improvements': {
                'html_removed': original_analysis['content_analysis']['contains_html'] and not cleaned_analysis['content_analysis']['contains_html'],
                'urls_removed': original_analysis['content_analysis']['contains_urls'] and not cleaned_analysis['content_analysis']['contains_urls'],
                'encoding_fixed': original_analysis['content_analysis']['encoding_issues'] and not cleaned_analysis['content_analysis']['encoding_issues']
            }
        }
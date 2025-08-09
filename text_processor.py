"""
Text Processing Module for handling text data cleaning (without ChromaDB)
"""
import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer, PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
    nltk.download('averaged_perceptron_tagger', quiet=True)
except:
    pass

class TextProcessor:
    def __init__(self, session_id: str):
        """Initialize text processor"""
        self.session_id = session_id
        self.text_data = []
        self.cleaned_text = []
        self.vectorizer = None
        self.vectors = None
        self.lemmatizer = WordNetLemmatizer()
        self.stemmer = PorterStemmer()
        
        # Store vectors in memory instead of ChromaDB
        self.vector_store = {}
        self.collection = None
    
    def load_text_data(self, text_data: Union[str, List[str], pd.Series]) -> Tuple[bool, str]:
        """Load text data from various sources"""
        try:
            if isinstance(text_data, str):
                self.text_data = [text_data]
            elif isinstance(text_data, pd.Series):
                self.text_data = text_data.tolist()
            elif isinstance(text_data, list):
                self.text_data = text_data
            else:
                return False, "Unsupported text data format"
            
            return True, f"Loaded {len(self.text_data)} text documents"
            
        except Exception as e:
            logging.error(f"Error loading text data: {e}")
            return False, str(e)
    
    def get_text_preview(self, n_chars: int = 500) -> str:
        """Get preview of text data"""
        if not self.text_data:
            return "No text data loaded"
        
        preview = self.text_data[0][:n_chars]
        if len(self.text_data[0]) > n_chars:
            preview += "..."
        
        return preview
    
    def clean_text(
        self,
        lowercase: bool = True,
        remove_punctuation: bool = True,
        remove_numbers: bool = False,
        remove_whitespace: bool = True,
        remove_stopwords: bool = True,
        lemmatize: bool = True,
        stem: bool = False,
        custom_stopwords: List[str] = None
    ) -> Tuple[bool, str]:
        """Clean text data with various options"""
        try:
            self.cleaned_text = []
            
            # Get stopwords
            try:
                stop_words = set(stopwords.words('english'))
            except:
                # If NLTK data not available, use basic stopwords
                stop_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
                                 'of', 'with', 'by', 'from', 'as', 'is', 'was', 'are', 'were'])
            
            if custom_stopwords:
                stop_words.update(custom_stopwords)
            
            for text in self.text_data:
                # Lowercase
                if lowercase:
                    text = text.lower()
                
                # Remove HTML tags
                text = re.sub(r'<[^>]+>', '', text)
                
                # Remove URLs
                text = re.sub(r'http\S+|www.\S+', '', text)
                
                # Remove email addresses
                text = re.sub(r'\S+@\S+', '', text)
                
                # Remove punctuation
                if remove_punctuation:
                    text = re.sub(r'[^\w\s]', ' ', text)
                
                # Remove numbers
                if remove_numbers:
                    text = re.sub(r'\d+', '', text)
                
                # Remove extra whitespace
                if remove_whitespace:
                    text = ' '.join(text.split())
                
                # Tokenize
                try:
                    tokens = word_tokenize(text)
                except:
                    # Fallback to simple tokenization
                    tokens = text.split()
                
                # Remove stopwords
                if remove_stopwords:
                    tokens = [t for t in tokens if t not in stop_words]
                
                # Lemmatization
                if lemmatize and not stem:
                    try:
                        tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
                    except:
                        pass  # Skip if lemmatization fails
                
                # Stemming
                if stem and not lemmatize:
                    try:
                        tokens = [self.stemmer.stem(t) for t in tokens]
                    except:
                        pass  # Skip if stemming fails
                
                # Join tokens back
                cleaned = ' '.join(tokens)
                self.cleaned_text.append(cleaned)
            
            return True, "Text cleaned successfully"
            
        except Exception as e:
            logging.error(f"Error cleaning text: {e}")
            return False, str(e)
    
    def extract_features(
        self,
        method: str = 'tfidf',
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 1)
    ) -> Tuple[bool, str]:
        """Extract features from text using various methods"""
        try:
            text_to_use = self.cleaned_text if self.cleaned_text else self.text_data
            
            if method == 'tfidf':
                self.vectorizer = TfidfVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range
                )
            elif method == 'count':
                self.vectorizer = CountVectorizer(
                    max_features=max_features,
                    ngram_range=ngram_range
                )
            else:
                return False, f"Unknown method: {method}"
            
            self.vectors = self.vectorizer.fit_transform(text_to_use)
            
            # Store vectors in memory
            for i, text in enumerate(text_to_use):
                self.vector_store[f"doc_{i}"] = {
                    'text': text,
                    'vector': self.vectors[i].toarray()[0]
                }
            
            return True, f"Extracted {self.vectors.shape[1]} features from {self.vectors.shape[0]} documents"
            
        except Exception as e:
            logging.error(f"Error extracting features: {e}")
            return False, str(e)
    
    def get_text_statistics(self) -> Dict[str, Any]:
        """Get statistics about text data"""
        if not self.text_data:
            return {}
        
        stats = {
            'num_documents': len(self.text_data),
            'total_characters': sum(len(t) for t in self.text_data),
            'avg_characters': np.mean([len(t) for t in self.text_data]),
            'total_words': sum(len(t.split()) for t in self.text_data),
            'avg_words': np.mean([len(t.split()) for t in self.text_data]),
            'unique_words': len(set(' '.join(self.text_data).split()))
        }
        
        if self.cleaned_text:
            stats['cleaned_total_words'] = sum(len(t.split()) for t in self.cleaned_text)
            stats['cleaned_avg_words'] = np.mean([len(t.split()) for t in self.cleaned_text])
            stats['cleaned_unique_words'] = len(set(' '.join(self.cleaned_text).split()))
        
        if self.vectors is not None:
            stats['vector_shape'] = self.vectors.shape
            stats['vector_density'] = self.vectors.nnz / (self.vectors.shape[0] * self.vectors.shape[1])
        
        return stats
    
    def get_top_terms(self, n: int = 20) -> List[Tuple[str, float]]:
        """Get top terms from vectorized text"""
        if self.vectorizer is None or self.vectors is None:
            return []
        
        # Get feature names
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Calculate term frequencies
        if hasattr(self.vectorizer, 'idf_'):
            # TF-IDF: use IDF scores
            scores = self.vectorizer.idf_
        else:
            # Count vectorizer: use sum of counts
            scores = self.vectors.sum(axis=0).A1
        
        # Get top terms
        top_indices = scores.argsort()[-n:][::-1]
        top_terms = [(feature_names[i], scores[i]) for i in top_indices]
        
        return top_terms
    
    def create_vector_database(
        self,
        collection_name: str = "text_collection"
    ) -> Tuple[bool, str]:
        """Create in-memory vector database for text data"""
        try:
            # Store collection name
            self.collection = f"{collection_name}_{self.session_id}"
            
            # Add documents to memory store
            text_to_use = self.cleaned_text if self.cleaned_text else self.text_data
            
            # Create embeddings if not already done
            if self.vectors is None:
                # Create simple embeddings
                vectorizer = TfidfVectorizer(max_features=100)
                vectors = vectorizer.fit_transform(text_to_use)
                
                for i, text in enumerate(text_to_use):
                    self.vector_store[f"doc_{i}"] = {
                        'text': text,
                        'vector': vectors[i].toarray()[0],
                        'metadata': {'index': i}
                    }
            
            return True, f"Created in-memory vector store with {len(text_to_use)} documents"
            
        except Exception as e:
            logging.error(f"Error creating vector database: {e}")
            return False, str(e)
    
    def search_similar(
        self,
        query: str,
        n_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar documents using cosine similarity"""
        try:
            if not self.vector_store or self.vectorizer is None:
                return []
            
            # Clean query same way as documents
            cleaned_query = self.clean_single_text(query)
            
            # Create query embedding
            query_vector = self.vectorizer.transform([cleaned_query]).toarray()[0]
            
            # Calculate cosine similarities
            similarities = []
            for doc_id, doc_data in self.vector_store.items():
                doc_vector = doc_data['vector']
                
                # Cosine similarity
                similarity = np.dot(query_vector, doc_vector) / (
                    np.linalg.norm(query_vector) * np.linalg.norm(doc_vector) + 1e-10
                )
                
                similarities.append({
                    'id': doc_id,
                    'text': doc_data['text'],
                    'similarity': similarity
                })
            
            # Sort by similarity and return top n
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            return similarities[:n_results]
            
        except Exception as e:
            logging.error(f"Error searching: {e}")
            return []
    
    def clean_single_text(self, text: str) -> str:
        """Clean a single text string using same settings as batch cleaning"""
        # Apply same cleaning as in clean_text method
        text = text.lower()
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'http\S+|www.\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = ' '.join(text.split())
        
        try:
            tokens = word_tokenize(text)
            stop_words = set(stopwords.words('english'))
            tokens = [t for t in tokens if t not in stop_words]
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        except:
            # Fallback to simple processing
            tokens = text.split()
            basic_stops = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for'}
            tokens = [t for t in tokens if t not in basic_stops]
        
        return ' '.join(tokens)
    
    def export_cleaned_text(self, file_path: str) -> Tuple[bool, str]:
        """Export cleaned text data"""
        try:
            text_to_export = self.cleaned_text if self.cleaned_text else self.text_data
            
            if file_path.endswith('.txt'):
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write('\n\n'.join(text_to_export))
            elif file_path.endswith('.csv'):
                df = pd.DataFrame({'text': text_to_export})
                df.to_csv(file_path, index=False)
            elif file_path.endswith('.json'):
                import json
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(text_to_export, f, indent=2)
            else:
                return False, "Unsupported export format"
            
            return True, f"Text exported to {file_path}"
            
        except Exception as e:
            logging.error(f"Error exporting text: {e}")
            return False, str(e)
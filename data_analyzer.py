"""
Data analyzer module for comprehensive data quality analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, List, Tuple
import warnings
from scipy import stats
from config import (MISSING_VALUE_THRESHOLD, OUTLIER_THRESHOLD, 
                   DUPLICATE_THRESHOLD, DATA_ANALYSIS_PROMPT, DEFAULT_SAMPLE_SIZE)

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Try to import profiling libraries (optional)
PROFILING_AVAILABLE = False
try:
    import ydata_profiling
    PROFILING_AVAILABLE = True
    PROFILING_LIB = 'ydata_profiling'
except ImportError:
    try:
        import pandas_profiling
        PROFILING_AVAILABLE = True
        PROFILING_LIB = 'pandas_profiling'
    except ImportError:
        try:
            import sweetviz
            PROFILING_AVAILABLE = True
            PROFILING_LIB = 'sweetviz'
        except ImportError:
            PROFILING_AVAILABLE = False

class DataAnalyzer:
    """Comprehensive data quality analysis"""
    
    def __init__(self, llm_client):
        self.llm_client = llm_client
        self.missing_threshold = MISSING_VALUE_THRESHOLD
        self.outlier_threshold = OUTLIER_THRESHOLD
        self.duplicate_threshold = DUPLICATE_THRESHOLD
        self.sample_size = DEFAULT_SAMPLE_SIZE
    
    def generate_quality_report(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive data quality report
        
        Args:
            data: pandas DataFrame
            
        Returns:
            Dictionary containing quality analysis results
        """
        report = {
            'basic_info': self._get_basic_info(data),
            'missing_values': self._analyze_missing_values(data),
            'duplicates': self._analyze_duplicates(data),
            'data_types': self._analyze_data_types(data),
            'outliers': self._detect_outliers(data),
            'categorical_analysis': self._analyze_categorical_columns(data),
            'numeric_analysis': self._analyze_numeric_columns(data),
            'text_analysis': self._analyze_text_columns(data),
            'data_consistency': self._check_data_consistency(data),
            'quality_score': 0  # Will be calculated
        }
        
        # Calculate overall quality score
        report['quality_score'] = self._calculate_quality_score(report)
        
        return report
    
    def _get_basic_info(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Get basic dataset information"""
        return {
            'shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'column_count': len(data.columns),
            'row_count': len(data),
            'total_cells': data.size,
            'non_null_cells': data.count().sum(),
            'null_cells': data.isnull().sum().sum()
        }
    
    def _analyze_missing_values(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze missing values pattern"""
        missing_info = {
            'total_missing': data.isnull().sum().sum(),
            'missing_percentage': (data.isnull().sum().sum() / data.size) * 100,
            'columns_with_missing': {},
            'missing_patterns': {},
            'problematic_columns': []
        }
        
        # Analyze each column
        for column in data.columns:
            missing_count = data[column].isnull().sum()
            if missing_count > 0:
                missing_percentage = (missing_count / len(data)) * 100
                missing_info['columns_with_missing'][column] = {
                    'count': missing_count,
                    'percentage': missing_percentage
                }
                
                # Flag problematic columns
                if missing_percentage > self.missing_threshold * 100:
                    missing_info['problematic_columns'].append(column)
        
        # Detect missing value patterns
        missing_info['missing_patterns'] = self._detect_missing_patterns(data)
        
        return missing_info
    
    def _detect_missing_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns in missing values"""
        patterns = {}
        
        # Find rows with multiple missing values
        missing_per_row = data.isnull().sum(axis=1)
        patterns['rows_with_multiple_missing'] = (missing_per_row > 1).sum()
        patterns['max_missing_per_row'] = missing_per_row.max()
        
        # Find columns that are missing together
        if len(data.columns) > 1:
            missing_correlations = data.isnull().corr()
            high_correlations = []
            
            for i, col1 in enumerate(data.columns):
                for j, col2 in enumerate(data.columns):
                    if i < j and abs(missing_correlations.loc[col1, col2]) > 0.5:
                        high_correlations.append({
                            'column1': col1,
                            'column2': col2,
                            'correlation': missing_correlations.loc[col1, col2]
                        })
            
            patterns['correlated_missing'] = high_correlations
        
        return patterns
    
    def _analyze_duplicates(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze duplicate records"""
        duplicate_info = {
            'total_duplicates': data.duplicated().sum(),
            'duplicate_percentage': (data.duplicated().sum() / len(data)) * 100,
            'unique_rows': len(data.drop_duplicates()),
            'is_problematic': False
        }
        
        # Check if duplicates are problematic
        if duplicate_info['duplicate_percentage'] > self.duplicate_threshold * 100:
            duplicate_info['is_problematic'] = True
        
        # Analyze partial duplicates (same values in subset of columns)
        if len(data.columns) > 1:
            duplicate_info['partial_duplicates'] = {}
            
            # Check key columns for partial duplicates
            for column in data.select_dtypes(include=['object', 'category']).columns[:5]:
                partial_dups = data[column].duplicated().sum()
                if partial_dups > 0:
                    duplicate_info['partial_duplicates'][column] = {
                        'count': partial_dups,
                        'percentage': (partial_dups / len(data)) * 100
                    }
        
        return duplicate_info
    
    def _analyze_data_types(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze data types and suggest improvements"""
        type_info = {
            'current_types': data.dtypes.value_counts().to_dict(),
            'memory_usage': data.memory_usage(deep=True).to_dict(),
            'type_suggestions': {},
            'problematic_types': []
        }
        
        for column in data.columns:
            current_type = str(data[column].dtype)
            suggestions = []
            
            # Analyze object columns for better types
            if current_type == 'object':
                # Check if it can be numeric
                numeric_convertible = pd.to_numeric(data[column], errors='coerce').notna().sum()
                if numeric_convertible / len(data) > 0.8:
                    suggestions.append('numeric (int/float)')
                
                # Check if it can be datetime
                try:
                    datetime_convertible = pd.to_datetime(data[column], errors='coerce').notna().sum()
                    if datetime_convertible / len(data) > 0.8:
                        suggestions.append('datetime')
                except:
                    pass
                
                # Check if it should be categorical
                unique_ratio = data[column].nunique() / len(data)
                if unique_ratio < 0.05:
                    suggestions.append('category')
            
            if suggestions:
                type_info['type_suggestions'][column] = suggestions
        
        return type_info
    
    def _detect_outliers(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers in numeric columns"""
        outlier_info = {
            'columns_with_outliers': {},
            'total_outlier_rows': 0,
            'outlier_methods': ['iqr', 'zscore']
        }
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        
        for column in numeric_columns:
            if data[column].notna().sum() == 0:
                continue
                
            outliers = self._detect_column_outliers(data[column])
            
            if outliers['count'] > 0:
                outlier_info['columns_with_outliers'][column] = outliers
        
        # Calculate total rows with any outliers
        outlier_rows = set()
        for col_outliers in outlier_info['columns_with_outliers'].values():
            outlier_rows.update(col_outliers.get('indices', []))
        
        outlier_info['total_outlier_rows'] = len(outlier_rows)
        outlier_info['outlier_percentage'] = (len(outlier_rows) / len(data)) * 100
        
        return outlier_info
    
    def _detect_column_outliers(self, series: pd.Series) -> Dict[str, Any]:
        """Detect outliers in a single column using multiple methods"""
        clean_series = series.dropna()
        outlier_info = {
            'count': 0,
            'indices': [],
            'methods': {}
        }
        
        if len(clean_series) == 0:
            return outlier_info
        
        # IQR method
        Q1 = clean_series.quantile(0.25)
        Q3 = clean_series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        iqr_outliers = series[(series < lower_bound) | (series > upper_bound)].index.tolist()
        outlier_info['methods']['iqr'] = {
            'count': len(iqr_outliers),
            'indices': iqr_outliers,
            'bounds': [lower_bound, upper_bound]
        }
        
        # Z-score method
        if clean_series.std() > 0:
            z_scores = np.abs(stats.zscore(clean_series))
            z_outliers = clean_series[z_scores > self.outlier_threshold].index.tolist()
            outlier_info['methods']['zscore'] = {
                'count': len(z_outliers),
                'indices': z_outliers,
                'threshold': self.outlier_threshold
            }
            
            # Combine outliers from both methods
            all_outliers = list(set(iqr_outliers + z_outliers))
            outlier_info['count'] = len(all_outliers)
            outlier_info['indices'] = all_outliers
        
        return outlier_info
    
    def _analyze_categorical_columns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze categorical columns"""
        categorical_info = {
            'categorical_columns': [],
            'value_distributions': {},
            'inconsistencies': {},
            'suggestions': {}
        }
        
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        
        for column in categorical_cols:
            categorical_info['categorical_columns'].append(column)
            
            # Value distribution
            value_counts = data[column].value_counts()
            categorical_info['value_distributions'][column] = {
                'unique_count': data[column].nunique(),
                'top_values': value_counts.head().to_dict(),
                'rare_values': value_counts[value_counts == 1].count()
            }
            
            # Check for inconsistencies (case sensitivity, whitespace, etc.)
            inconsistencies = self._find_categorical_inconsistencies(data[column])
            if inconsistencies:
                categorical_info['inconsistencies'][column] = inconsistencies
        
        return categorical_info
    
    def _find_categorical_inconsistencies(self, series: pd.Series) -> List[Dict[str, Any]]:
        """Find inconsistencies in categorical data"""
        inconsistencies = []
        unique_values = series.dropna().unique()
        
        # Group similar values (case-insensitive, whitespace)
        value_groups = {}
        for value in unique_values:
            clean_value = str(value).lower().strip()
            if clean_value not in value_groups:
                value_groups[clean_value] = []
            value_groups[clean_value].append(value)
        
        # Find groups with multiple variants
        for clean_value, variants in value_groups.items():
            if len(variants) > 1:
                inconsistencies.append({
                    'type': 'case_whitespace',
                    'variants': variants,
                    'suggested_value': max(variants, key=lambda x: series[series == x].count())
                })
        
        return inconsistencies
    
    def _analyze_numeric_columns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze numeric columns"""
        numeric_info = {
            'numeric_columns': [],
            'statistics': {},
            'distributions': {},
            'correlations': None
        }
        
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        numeric_info['numeric_columns'] = numeric_cols.tolist()
        
        if len(numeric_cols) > 0:
            # Basic statistics
            numeric_info['statistics'] = data[numeric_cols].describe().to_dict()
            
            # Distribution analysis
            for column in numeric_cols:
                if data[column].notna().sum() > 0:
                    numeric_info['distributions'][column] = {
                        'skewness': data[column].skew(),
                        'kurtosis': data[column].kurtosis(),
                        'is_normal': self._test_normality(data[column])
                    }
            
            # Correlation analysis
            if len(numeric_cols) > 1:
                corr_matrix = data[numeric_cols].corr()
                high_correlations = []
                
                for i, col1 in enumerate(numeric_cols):
                    for j, col2 in enumerate(numeric_cols):
                        if i < j and abs(corr_matrix.loc[col1, col2]) > 0.8:
                            high_correlations.append({
                                'column1': col1,
                                'column2': col2,
                                'correlation': corr_matrix.loc[col1, col2]
                            })
                
                numeric_info['correlations'] = {
                    'high_correlations': high_correlations,
                    'matrix': corr_matrix.to_dict()
                }
        
        return numeric_info
    
    def _test_normality(self, series: pd.Series) -> bool:
        """Test if data follows normal distribution"""
        clean_series = series.dropna()
        if len(clean_series) < 8:  # Minimum sample size for normality test
            return False
        
        try:
            _, p_value = stats.shapiro(clean_series.sample(min(5000, len(clean_series))))
            return p_value > 0.05
        except:
            return False
    
    def _analyze_text_columns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze text columns"""
        text_info = {
            'text_columns': [],
            'length_statistics': {},
            'character_analysis': {},
            'potential_issues': {}
        }
        
        text_cols = data.select_dtypes(include=['object']).columns
        
        for column in text_cols:
            if data[column].dtype == 'object':
                text_info['text_columns'].append(column)
                
                # Length analysis
                lengths = data[column].astype(str).str.len()
                text_info['length_statistics'][column] = {
                    'min_length': lengths.min(),
                    'max_length': lengths.max(),
                    'avg_length': lengths.mean(),
                    'empty_strings': (data[column] == '').sum()
                }
                
                # Character analysis
                text_info['character_analysis'][column] = {
                    'contains_numbers': data[column].str.contains(r'\d', na=False).sum(),
                    'contains_special_chars': data[column].str.contains(r'[^a-zA-Z0-9\s]', na=False).sum(),
                    'all_caps': data[column].str.isupper().sum(),
                    'all_lower': data[column].str.islower().sum()
                }
        
        return text_info
    
    def _check_data_consistency(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Check for data consistency issues"""
        consistency_info = {
            'issues': [],
            'severity_score': 0,
            'recommendations': []
        }
        
        # Check for mixed data types in object columns
        for column in data.select_dtypes(include=['object']).columns:
            types_found = set()
            for value in data[column].dropna().sample(min(1000, len(data[column].dropna()))):
                if isinstance(value, (int, float)):
                    types_found.add('numeric')
                elif isinstance(value, str):
                    if value.isdigit():
                        types_found.add('numeric_string')
                    else:
                        types_found.add('text')
            
            if len(types_found) > 1:
                consistency_info['issues'].append({
                    'type': 'mixed_types',
                    'column': column,
                    'types_found': list(types_found)
                })
        
        # Check for inconsistent date formats
        # Check for encoding issues
        # Add more consistency checks as needed
        
        return consistency_info
    
    def _calculate_quality_score(self, report: Dict[str, Any]) -> float:
        """Calculate overall data quality score (0-100)"""
        score = 100.0
        
        # Deduct for missing values
        missing_percentage = report['missing_values']['missing_percentage']
        score -= min(missing_percentage * 0.5, 20)  # Max 20 points deduction
        
        # Deduct for duplicates
        duplicate_percentage = report['duplicates']['duplicate_percentage']
        score -= min(duplicate_percentage * 0.3, 15)  # Max 15 points deduction
        
        # Deduct for outliers
        outlier_percentage = report['outliers'].get('outlier_percentage', 0)
        score -= min(outlier_percentage * 0.2, 10)  # Max 10 points deduction
        
        # Deduct for inconsistencies
        inconsistency_count = len(report['data_consistency']['issues'])
        score -= min(inconsistency_count * 5, 15)  # Max 15 points deduction
        
        return max(score, 0)
    
    def enhanced_llm_analysis(self, data: pd.DataFrame, schema: Dict[str, Any], 
                            quality_report: Dict[str, Any]) -> Dict[str, Any]:
        """Use LLM for enhanced data analysis"""
        try:
            # Prepare context for LLM
            context = {
                'data_shape': data.shape,
                'schema': schema,
                'quality_score': quality_report['quality_score'],
                'key_issues': {
                    'missing_percentage': quality_report['missing_values']['missing_percentage'],
                    'duplicate_percentage': quality_report['duplicates']['duplicate_percentage'],
                    'outlier_percentage': quality_report['outliers'].get('outlier_percentage', 0),
                    'problematic_columns': quality_report['missing_values']['problematic_columns']
                }
            }
            
            # Sample data for LLM analysis (to avoid token limits)
            sample_data = data.head(min(10, len(data))).to_dict()
            
            prompt = f"""
            {DATA_ANALYSIS_PROMPT}
            
            Dataset Context:
            {context}
            
            Sample Data:
            {sample_data}
            
            Provide your analysis as JSON with the following structure:
            {{
                "summary": "Brief summary of data quality",
                "critical_issues": ["list of critical issues"],
                "recommendations": ["prioritized recommendations"],
                "cleaning_strategy": "suggested approach for cleaning",
                "business_impact": "potential business impact of quality issues"
            }}
            """
            
            if self.llm_client and self.llm_client.client:
                response = self.llm_client.get_json_completion(prompt)
                
                if 'error' not in response:
                    return response
                else:
                    # Fallback to basic analysis
                    return self._fallback_analysis(quality_report)
            else:
                # No LLM available, use fallback
                return self._fallback_analysis(quality_report)
                
        except Exception as e:
            print(f"Warning: LLM analysis failed: {str(e)}")
            return self._fallback_analysis(quality_report)
    
    def _fallback_analysis(self, quality_report: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback analysis when LLM is unavailable"""
        return {
            "summary": f"Data quality score: {quality_report['quality_score']:.1f}/100",
            "critical_issues": [
                f"Missing values: {quality_report['missing_values']['missing_percentage']:.1f}%",
                f"Duplicates: {quality_report['duplicates']['duplicate_percentage']:.1f}%"
            ],
            "recommendations": [
                "Handle missing values appropriately",
                "Remove or investigate duplicate records",
                "Validate data types and formats"
            ],
            "cleaning_strategy": "Sequential cleaning: missing values → duplicates → outliers → standardization",
            "business_impact": "Data quality issues may affect analysis accuracy and decision making"
        }
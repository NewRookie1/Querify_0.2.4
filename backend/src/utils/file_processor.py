"""
Handles CSV and Excel files with automatic data analysis
"""

import pandas as pd
from typing import Dict, List, Tuple, Optional
import os


class DataFileProcessor:
    """
    Process CSV and Excel files for data analysis
    
    Supported formats: CSV, XLSX, XLS
    Max rows for preview: 1000
    """
    
    def __init__(self):
        self.max_preview_rows = 1000
        self.supported_formats = ['.csv', '.xlsx', '.xls']
        print("✓ Data file processor initialized")
    
    def validate_file(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate data file
        
        Args:
            file_path: Path to data file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if file exists
        if not os.path.exists(file_path):
            return False, "File not found"
        
        # Check file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() not in self.supported_formats:
            return False, f"Unsupported format. Supported: {', '.join(self.supported_formats)}"
        
        # Check file size (warn if > 50MB)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > 50:
            return False, f"File too large ({size_mb:.1f}MB). Max: 50MB"
        
        return True, ""
    
    def load_dataframe(self, file_path: str) -> pd.DataFrame:
        """
        Load file as pandas DataFrame
        
        Args:
            file_path: Path to data file
            
        Returns:
            pandas DataFrame
        """
        _, ext = os.path.splitext(file_path)
        
        try:
            if ext.lower() == '.csv':
                # Try different encodings
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1']:
                    try:
                        df = pd.read_csv(file_path, encoding=encoding, nrows=self.max_preview_rows)
                        return df
                    except UnicodeDecodeError:
                        continue
                raise Exception("Could not decode CSV file")
            
            elif ext.lower() in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path, nrows=self.max_preview_rows)
                return df
            
            else:
                raise Exception(f"Unsupported file format: {ext}")
        
        except Exception as e:
            raise Exception(f"Failed to load file: {str(e)}")
    
    def generate_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate comprehensive data summary
        
        Args:
            df: pandas DataFrame
            
        Returns:
            Dictionary with data summary
        """
        summary = {
            'shape': df.shape,
            'n_rows': df.shape[0],
            'n_columns': df.shape[1],
            'columns': df.columns.tolist(),
            'dtypes': df.dtypes.astype(str).to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'missing_percentage': (df.isnull().sum() / len(df) * 100).round(2).to_dict()
        }
        
        # Numeric columns summary
        numeric_cols = df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            summary['numeric_summary'] = df[numeric_cols].describe().to_dict()
        
        # Memory usage
        summary['memory_usage_mb'] = df.memory_usage(deep=True).sum() / (1024 * 1024)
        
        return summary
    
    def detect_issues(self, df: pd.DataFrame) -> List[str]:
        """
        Detect common data issues
        
        Args:
            df: pandas DataFrame
            
        Returns:
            List of warnings/issues
        """
        issues = []
        
        # High missing values
        missing_pct = (df.isnull().sum() / len(df) * 100)
        high_missing = missing_pct[missing_pct > 5]
        if len(high_missing) > 0:
            for col, pct in high_missing.items():
                issues.append(f"⚠️ Column '{col}' has {pct:.1f}% missing values")
        
        # Duplicate rows
        n_duplicates = df.duplicated().sum()
        if n_duplicates > 0:
            pct_dup = (n_duplicates / len(df)) * 100
            issues.append(f"⚠️ Found {n_duplicates} duplicate rows ({pct_dup:.1f}%)")
        
        # Single-value columns
        for col in df.columns:
            if df[col].nunique() == 1:
                issues.append(f"⚠️ Column '{col}' has only one unique value")
        
        # High cardinality categorical columns
        object_cols = df.select_dtypes(include=['object']).columns
        for col in object_cols:
            n_unique = df[col].nunique()
            if n_unique > len(df) * 0.5:  # More than 50% unique values
                issues.append(f"ℹ️ Column '{col}' has high cardinality ({n_unique} unique values)")
        
        return issues
    
    def format_for_llm(self, df: pd.DataFrame, include_preview: bool = True) -> str:
        """
        Format DataFrame info for LLM context
        
        Args:
            df: pandas DataFrame
            include_preview: Whether to include data preview
            
        Returns:
            Human-readable text summary
        """
        summary = self.generate_summary(df)
        issues = self.detect_issues(df)
        
        # Build formatted text
        text_parts = []
        
        # Basic info
        text_parts.append("📊 DATASET SUMMARY")
        text_parts.append(f"Shape: {summary['n_rows']} rows × {summary['n_columns']} columns")
        text_parts.append("")
        
        # Columns
        text_parts.append("📋 COLUMNS:")
        for col, dtype in summary['dtypes'].items():
            missing = summary['missing_values'][col]
            missing_pct = summary['missing_percentage'][col]
            
            if missing > 0:
                text_parts.append(f"  • {col} ({dtype}) - {missing} missing ({missing_pct:.1f}%)")
            else:
                text_parts.append(f"  • {col} ({dtype})")
        text_parts.append("")
        
        # Numeric summary (if any)
        if 'numeric_summary' in summary:
            text_parts.append("📈 NUMERIC COLUMNS STATISTICS:")
            for col in summary['numeric_summary']:
                stats = summary['numeric_summary'][col]
                text_parts.append(f"  • {col}:")
                text_parts.append(f"    - Mean: {stats.get('mean', 'N/A'):.2f}" if isinstance(stats.get('mean'), (int, float)) else f"    - Mean: {stats.get('mean', 'N/A')}")
                text_parts.append(f"    - Std: {stats.get('std', 'N/A'):.2f}" if isinstance(stats.get('std'), (int, float)) else f"    - Std: {stats.get('std', 'N/A')}")
                text_parts.append(f"    - Min: {stats.get('min', 'N/A'):.2f}" if isinstance(stats.get('min'), (int, float)) else f"    - Min: {stats.get('min', 'N/A')}")
                text_parts.append(f"    - Max: {stats.get('max', 'N/A'):.2f}" if isinstance(stats.get('max'), (int, float)) else f"    - Max: {stats.get('max', 'N/A')}")
            text_parts.append("")
        
        # Issues
        if issues:
            text_parts.append("🔍 DETECTED ISSUES:")
            for issue in issues:
                text_parts.append(f"  {issue}")
            text_parts.append("")
        
        # Data preview
        if include_preview:
            text_parts.append("👀 DATA PREVIEW (first 5 rows):")
            preview = df.head(5).to_string(index=False)
            text_parts.append(preview)
        
        return "\n".join(text_parts)
    
    def suggest_analysis(self, df: pd.DataFrame) -> List[str]:
        """
        Suggest appropriate analysis based on data characteristics
        
        Args:
            df: pandas DataFrame
            
        Returns:
            List of analysis suggestions
        """
        suggestions = []
        
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) >= 2:
            suggestions.append("📊 Correlation analysis between numeric variables")
            suggestions.append("📈 Distribution plots for numeric columns")
        
        if len(numeric_cols) >= 1 and len(categorical_cols) >= 1:
            suggestions.append("📊 Group comparison (numeric by categorical)")
            suggestions.append("📈 Box plots by category")
        
        if len(categorical_cols) >= 1:
            suggestions.append("📊 Frequency analysis of categorical variables")
        
        # Check for potential target variable (binary classification)
        for col in df.columns:
            if df[col].nunique() == 2:
                suggestions.append(f"🎯 '{col}' could be a binary target variable")
        
        return suggestions


def test_data_processor():
    """Test function for data processor"""
    processor = DataFileProcessor()
    
    print("\n🧪 Testing Data File Processor...")
    print(f"✓ Supported formats: {processor.supported_formats}")
    print(f"✓ Max preview rows: {processor.max_preview_rows}")
    print("✓ Data file processor ready!")


if __name__ == "__main__":
    test_data_processor()

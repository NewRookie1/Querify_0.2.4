"""
Multi-Modal Handler for 
Orchestrates all file processors and routes to appropriate handler
"""

from typing import Dict, Optional, Tuple, Any
import os
from PIL import Image

# Import processors
try:
    from .image_processor import ImageProcessor
    from .file_processor import DataFileProcessor
    from .pdf_processor import PDFProcessor
    INTERNAL_IMPORT = True
except ImportError:
    # Fallback for standalone testing
    try:
        from image_processor import ImageProcessor
        from file_processor import DataFileProcessor
        from pdf_processor import PDFProcessor
        INTERNAL_IMPORT = True
    except ImportError:
        INTERNAL_IMPORT = False
        print("⚠️  Could not import processors")


class MultiModalHandler:
    """
    Central handler for all file types
    Routes to appropriate processor based on file type
    """
    
    def __init__(self):
        """Initialize all processors"""
        if not INTERNAL_IMPORT:
            raise Exception("Required processors not available")
        
        self.image_processor = ImageProcessor() # type: ignore
        self.file_processor = DataFileProcessor() # type: ignore
        self.pdf_processor = PDFProcessor() # type: ignore
        
        print("✓ Multi-modal handler initialized")
    
    def detect_file_type(self, file_path: str) -> str:
        """
        Detect file type from extension
        
        Args:
            file_path: Path to file
            
        Returns:
            File type: 'image', 'data', 'pdf', 'unknown'
        """
        if not os.path.exists(file_path):
            return 'unknown'
        
        _, ext = os.path.splitext(file_path)
        ext = ext.lower()
        
        if ext in self.image_processor.supported_formats:
            return 'image'
        elif ext in self.file_processor.supported_formats:
            return 'data'
        elif ext in [self.pdf_processor.supported_format]:
            return 'pdf'
        else:
            return 'unknown'
    
    def process_file(self, file_path: str, user_query: str = "") -> Dict[str, Any]: # type: ignore
        """
        Process any supported file type
        
        Args:
            file_path: Path to file
            user_query: Optional user question about the file
            
        Returns:
            Dictionary with:
                - file_type: str
                - processed_content: Any (depends on type)
                - metadata: Dict
                - analysis: str (formatted for LLM)
                - error: Optional[str]
        """
        # Detect file type
        file_type = self.detect_file_type(file_path)
        
        if file_type == 'unknown':
            return {
                'file_type': 'unknown',
                'processed_content': None,
                'metadata': {},
                'analysis': '',
                'error': 'Unsupported file type'
            }
        
        # Route to appropriate processor
        try:
            if file_type == 'image':
                return self._process_image(file_path, user_query)
            elif file_type == 'data':
                return self._process_data_file(file_path, user_query)
            elif file_type == 'pdf':
                return self._process_pdf(file_path, user_query)
        
        except Exception as e:
            return {
                'file_type': file_type,
                'processed_content': None,
                'metadata': {},
                'analysis': '',
                'error': str(e)
            }
    
    def _process_image(self, file_path: str, user_query: str) -> Dict[str, Any]:
        """Process image file"""
        # Prepare image
        image, metadata = self.image_processor.prepare_for_gemini(file_path)
        
        # Generate context prompt
        image_type = metadata.get('detected_type', 'unknown')
        context_prompt = self.image_processor.generate_context_prompt(image_type, user_query)
        
        # Build analysis text
        analysis = f"""📸 IMAGE ANALYSIS

File: {os.path.basename(file_path)}
Dimensions: {metadata['width']} × {metadata['height']} pixels
Format: {metadata['format']}
Detected Type: {image_type}

{context_prompt}
"""
        
        return {
            'file_type': 'image',
            'processed_content': image,  # PIL Image object
            'metadata': metadata,
            'analysis': analysis,
            'error': None
        }
    
    def _process_data_file(self, file_path: str, user_query: str) -> Dict[str, Any]:
        """Process CSV/Excel file"""
        # Load dataframe
        df = self.file_processor.load_dataframe(file_path)
        
        # Generate summary and analysis
        formatted_summary = self.file_processor.format_for_llm(df, include_preview=True)
        issues = self.file_processor.detect_issues(df)
        suggestions = self.file_processor.suggest_analysis(df)
        
        # Build analysis text
        analysis_parts = [
            f"📊 DATA FILE ANALYSIS\n",
            f"File: {os.path.basename(file_path)}\n",
            formatted_summary,
            "\n"
        ]
        
        if suggestions:
            analysis_parts.append("\n💡 SUGGESTED ANALYSES:")
            for suggestion in suggestions:
                analysis_parts.append(f"  {suggestion}")
        
        if user_query:
            analysis_parts.append(f"\nUSER QUESTION: {user_query}")
        
        analysis = "\n".join(analysis_parts)
        
        return {
            'file_type': 'data',
            'processed_content': formatted_summary,  # Text summary
            'metadata': {
                'shape': df.shape,
                'columns': df.columns.tolist(),
                'n_issues': len(issues)
            },
            'analysis': analysis,
            'error': None
        }
    
    def _process_pdf(self, file_path: str, user_query: str) -> Dict[str, Any]:
        """Process PDF file"""
        # Get PDF info
        pdf_info = self.pdf_processor.get_pdf_info(file_path)
        
        if 'error' in pdf_info:
            raise Exception(pdf_info['error'])
        
        # Check if PDF has text
        has_text = self.pdf_processor.check_has_text(file_path)
        
        if not has_text:
            return {
                'file_type': 'pdf',
                'processed_content': None,
                'metadata': pdf_info,
                'analysis': '',
                'error': 'PDF appears to be scanned/image-only. No extractable text found.'
            }
        
        # Extract text
        text = self.pdf_processor.extract_text(file_path)
        
        # Format for LLM
        formatted_text = self.pdf_processor.format_for_llm(text)
        
        # Build analysis
        analysis_parts = [
            f"📄 PDF DOCUMENT ANALYSIS\n",
            f"File: {os.path.basename(file_path)}",
            f"Pages: {pdf_info['num_pages']}",
            f"Size: {pdf_info['file_size_mb']:.2f} MB\n",
            formatted_text
        ]
        
        if user_query:
            analysis_parts.append(f"\n\nUSER QUESTION: {user_query}")
        
        analysis = "\n".join(analysis_parts)
        
        return {
            'file_type': 'pdf',
            'processed_content': formatted_text,  # Extracted text
            'metadata': pdf_info,
            'analysis': analysis,
            'error': None
        }
    
    def prepare_multimodal_context(
        self, 
        user_query: str, 
        processed_file: Dict
    ) -> Tuple[str, Optional[Image.Image]]:
        """
        Prepare context for Gemini API
        
        Args:
            user_query: User's question
            processed_file: Result from process_file()
            
        Returns:
            Tuple of (text_context, image_object)
        """
        # If there was an error, return error message
        if processed_file.get('error'):
            return f"Error processing file: {processed_file['error']}", None
        
        file_type = processed_file['file_type']
        
        if file_type == 'image':
            # For images, return both text context and image
            return processed_file['analysis'], processed_file['processed_content']
        
        else:
            # For data files and PDFs, return only text context
            return processed_file['analysis'], None


def test_multimodal_handler():
    """Test function for multi-modal handler"""
    if not INTERNAL_IMPORT:
        print("❌ Cannot test - processors not available")
        return
    
    handler = MultiModalHandler()
    
    print("\n🧪 Testing Multi-Modal Handler...")
    print(f"✓ Image processor: Ready")
    print(f"✓ Data file processor: Ready")
    print(f"✓ PDF processor: Ready")
    print("✓ Multi-modal handler ready!")


if __name__ == "__main__":
    test_multimodal_handler()

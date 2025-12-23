"""
PDF Processor for
Extracts and processes text from PDF documents
"""

from typing import Dict, List, Tuple, Optional
import os

try:
    from PyPDF2 import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False
    print("⚠️  PyPDF2 not available. Install with: pip install PyPDF2")


class PDFProcessor:
    """
    Extract and process text from PDF documents
    
    Supported: Text-based PDFs
    Max pages: 50
    """
    
    def __init__(self):
        self.max_pages = 50
        self.supported_format = '.pdf'
        
        if not PDF_AVAILABLE:
            print("⚠️  PDF processing unavailable - PyPDF2 not installed")
        else:
            print("✓ PDF processor initialized")
    
    def validate_pdf(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate PDF file
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not PDF_AVAILABLE:
            return False, "PyPDF2 not installed"
        
        # Check if file exists
        if not os.path.exists(file_path):
            return False, "File not found"
        
        # Check file extension
        _, ext = os.path.splitext(file_path)
        if ext.lower() != self.supported_format:
            return False, f"Not a PDF file"
        
        # Check file size (warn if > 20MB)
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > 20:
            return False, f"File too large ({size_mb:.1f}MB). Max: 20MB"
        
        # Try to open as PDF
        try:
            reader = PdfReader(file_path) # type: ignore
            _ = len(reader.pages)
            return True, ""
        except Exception as e:
            return False, f"Invalid or encrypted PDF: {str(e)}"
    
    def get_pdf_info(self, file_path: str) -> Dict:
        """
        Get PDF metadata
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Dictionary with PDF info
        """
        if not PDF_AVAILABLE:
            return {'error': 'PyPDF2 not installed'}
        
        try:
            reader = PdfReader(file_path) # type: ignore
            info = {
                'num_pages': len(reader.pages),
                'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
                'encrypted': reader.is_encrypted,
            }
            
            # Try to get metadata
            if reader.metadata:
                info['title'] = reader.metadata.get('/Title', 'Unknown')
                info['author'] = reader.metadata.get('/Author', 'Unknown')
            
            return info
        except Exception as e:
            return {'error': str(e)}
    
    def extract_text(self, file_path: str, max_pages: Optional[int] = None) -> str:
        """
        Extract all text from PDF
        
        Args:
            file_path: Path to PDF file
            max_pages: Maximum number of pages to extract (default: self.max_pages)
            
        Returns:
            Extracted text
        """
        if not PDF_AVAILABLE:
            raise Exception("PyPDF2 not installed")
        
        max_pages = max_pages or self.max_pages
        
        try:
            reader = PdfReader(file_path) # type: ignore
            text_parts = []
            
            # Extract text from each page
            num_pages = min(len(reader.pages), max_pages)
            
            for page_num in range(num_pages):
                page = reader.pages[page_num]
                text = page.extract_text()
                
                if text.strip():
                    text_parts.append(f"--- Page {page_num + 1} ---\n{text}")
            
            if len(reader.pages) > max_pages:
                text_parts.append(f"\n... (truncated at {max_pages} pages, total: {len(reader.pages)} pages)")
            
            return "\n\n".join(text_parts)
        
        except Exception as e:
            raise Exception(f"Failed to extract text: {str(e)}")
    
    def extract_by_pages(self, file_path: str, start_page: int = 0, end_page: Optional[int] = None) -> List[str]:
        """
        Extract text page by page
        
        Args:
            file_path: Path to PDF file
            start_page: Starting page (0-indexed)
            end_page: Ending page (0-indexed, None for all)
            
        Returns:
            List of page texts
        """
        if not PDF_AVAILABLE:
            raise Exception("PyPDF2 not installed")
        
        try:
            reader = PdfReader(file_path) # type: ignore
            page_texts = []
            
            end_page = end_page or len(reader.pages)
            end_page = min(end_page, len(reader.pages), start_page + self.max_pages)
            
            for page_num in range(start_page, end_page):
                page = reader.pages[page_num]
                text = page.extract_text()
                page_texts.append(text)
            
            return page_texts
        
        except Exception as e:
            raise Exception(f"Failed to extract pages: {str(e)}")
    
    def format_for_llm(self, text: str, max_chars: int = 15000) -> str:
        """
        Format extracted text for LLM
        
        Args:
            text: Extracted text
            max_chars: Maximum characters to include
            
        Returns:
            Formatted text
        """
        if len(text) <= max_chars:
            return f"📄 PDF CONTENT:\n\n{text}"
        
        # Truncate intelligently
        truncated = text[:max_chars]
        
        # Try to end at a sentence
        last_period = truncated.rfind('.')
        if last_period > max_chars * 0.8:  # If we can find a sentence end in last 20%
            truncated = truncated[:last_period + 1]
        
        remaining_chars = len(text) - len(truncated)
        
        return f"📄 PDF CONTENT (showing first {len(truncated)} of {len(text)} characters):\n\n{truncated}\n\n... ({remaining_chars} more characters not shown)"
    
    def check_has_text(self, file_path: str) -> bool:
        """
        Check if PDF has extractable text
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            True if PDF has text, False if scanned/image-only
        """
        if not PDF_AVAILABLE:
            return False
        
        try:
            reader = PdfReader(file_path) # type: ignore
            
            # Check first few pages
            pages_to_check = min(3, len(reader.pages))
            
            for page_num in range(pages_to_check):
                text = reader.pages[page_num].extract_text().strip()
                if len(text) > 50:  # Has substantial text
                    return True
            
            return False
        
        except:
            return False


def test_pdf_processor():
    """Test function for PDF processor"""
    processor = PDFProcessor()
    
    print("\n🧪 Testing PDF Processor...")
    print(f"✓ Max pages: {processor.max_pages}")
    print(f"✓ PyPDF2 available: {PDF_AVAILABLE}")
    
    if PDF_AVAILABLE:
        print("✓ PDF processor ready!")
    else:
        print("⚠️  Install PyPDF2 to enable PDF processing")


if __name__ == "__main__":
    test_pdf_processor()

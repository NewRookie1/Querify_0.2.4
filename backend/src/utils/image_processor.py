"""
Handles image uploads, validation, and preparation for Gemini API
"""

from PIL import Image
from typing import Dict, Tuple, Optional
import os


class ImageProcessor:
    """
    Process images for multi-modal AI analysis
    
    Supported formats: PNG, JPEG, WEBP
    Max size: 20MB (Gemini limit)
    """
    
    def __init__(self):
        self.max_size_mb = 20
        self.supported_formats = ['.png', '.jpg', '.jpeg', '.webp', '.heic', '.heif']
        print("✓ Image processor initialized")
    
    def validate_image(self, file_path: str) -> Tuple[bool, str]:
        """
        Validate image file
        
        Args:
            file_path: Path to image file
            
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
        
        # Check file size
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if size_mb > self.max_size_mb:
            return False, f"File too large ({size_mb:.1f}MB). Max: {self.max_size_mb}MB"
        
        # Try to open as image
        try:
            with Image.open(file_path) as img:
                img.verify()
            return True, ""
        except Exception as e:
            return False, f"Invalid image file: {str(e)}"
    
    def load_image(self, file_path: str) -> Image.Image:
        """
        Load and return PIL Image object
        
        Args:
            file_path: Path to image file
            
        Returns:
            PIL Image object
        """
        try:
            image = Image.open(file_path)
            # Convert to RGB if needed (for consistency)
            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')
            return image
        except Exception as e:
            raise Exception(f"Failed to load image: {str(e)}")
    
    def get_image_info(self, image: Image.Image) -> Dict:
        """
        Extract image metadata
        
        Args:
            image: PIL Image object
            
        Returns:
            Dictionary with image metadata
        """
        return {
            'width': image.width,
            'height': image.height,
            'format': image.format or 'Unknown',
            'mode': image.mode,
            'size_pixels': image.width * image.height
        }
    
    def analyze_image_type(self, image: Image.Image) -> str:
        """
        Detect likely image type using heuristics
        
        Args:
            image: PIL Image object
            
        Returns:
            Detected type: 'plot', 'diagram', 'code', 'document', 'photo', 'unknown'
        """
        width, height = image.size
        aspect_ratio = width / height if height > 0 else 1.0
        
        # Very wide or very tall images are often plots or documents
        if aspect_ratio > 2.5 or aspect_ratio < 0.4:
            return 'plot'
        
        # Square-ish images with moderate size might be confusion matrices
        if 0.8 < aspect_ratio < 1.2 and 400 < width < 1000:
            return 'diagram'
        
        # Large high-res images are likely photos
        if width > 1920 or height > 1080:
            return 'photo'
        
        # Default
        return 'unknown'
    
    def prepare_for_gemini(self, file_path: str) -> Tuple[Image.Image, Dict]:
        """
        Prepare image for Gemini API
        
        Args:
            file_path: Path to image file
            
        Returns:
            Tuple of (PIL Image object, metadata dict)
        """
        # Validate
        is_valid, error_msg = self.validate_image(file_path)
        if not is_valid:
            raise ValueError(error_msg)
        
        # Load image
        image = self.load_image(file_path)
        
        # Get metadata
        metadata = self.get_image_info(image)
        metadata['detected_type'] = self.analyze_image_type(image)
        metadata['file_path'] = file_path
        
        return image, metadata
    
    def generate_context_prompt(self, image_type: str, user_query: str = "") -> str:
        """
        Generate appropriate context prompt based on image type
        
        Args:
            image_type: Detected or specified image type
            user_query: User's question about the image
            
        Returns:
            Formatted prompt for Gemini
        """
        base_prompt = f"USER QUERY: {user_query}\n\n" if user_query else ""
        
        type_prompts = {
            'plot': "This appears to be a data visualization or plot. Please analyze:\n"
                   "- What type of plot is this?\n"
                   "- What are the key patterns or trends?\n"
                   "- What insights can you derive?\n"
                   "- Are there any issues or improvements needed?",
            
            'diagram': "This appears to be a diagram or technical illustration. Please analyze:\n"
                      "- What does this diagram represent?\n"
                      "- What are the key components or relationships?\n"
                      "- What can you explain about this?",
            
            'code': "This appears to be a code screenshot. Please analyze:\n"
                   "- What programming language is this?\n"
                   "- What does the code do?\n"
                   "- Are there any issues or improvements?",
            
            'document': "This appears to be a document or text. Please:\n"
                       "- Extract and summarize the key information\n"
                       "- Identify the main points\n"
                       "- Answer any specific questions about it",
            
            'photo': "Please describe this image and answer any questions about it.",
            
            'unknown': "Please analyze this image and provide relevant insights."
        }
        
        return base_prompt + type_prompts.get(image_type, type_prompts['unknown'])


def test_image_processor():
    """Test function for image processor"""
    processor = ImageProcessor()
    
    print("\n🧪 Testing Image Processor...")
    print(f"✓ Supported formats: {processor.supported_formats}")
    print(f"✓ Max size: {processor.max_size_mb}MB")
    print("✓ Image processor ready!")


if __name__ == "__main__":
    test_image_processor()

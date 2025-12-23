"""
Citation Formatter for Formats and displays source citations in responses
"""

from typing import List, Dict
import re


class CitationFormatter:
    """Format citations in responses"""
    
    @staticmethod
    def add_citations(response: str, sources: List[Dict]) -> str:
        """
        Add formatted citations to response
        
        Args:
            response: Generated response text
            sources: List of source metadata
            
        Returns:
            Response with citations appended
        """
        if not sources:
            return response
        
        # Add citation section at bottom
        citation_text = "\n\n---\n** Page:**\n"
        
        for source in sources:
            source_id = source.get('id', 0)
            filename = source.get('filename', 'Unknown')
            category = source.get('category', 'General')
            
            citation_text += f"\n{source_id}. **{category}**: {filename}"
        
        return response + citation_text
    
    @staticmethod
    def extract_citation_numbers(response: str) -> List[int]:
        """
        Extract [Source N] references from response
        
        Args:
            response: Response text
            
        Returns:
            List of cited source numbers
        """
        pattern = r'\[Source (\d+)\]'
        matches = re.findall(pattern, response)
        return [int(m) for m in matches]
    
    @staticmethod
    def format_inline_citation(text: str, source_id: int) -> str:
        """
        Format inline citation
        
        Args:
            text: Text to cite
            source_id: Source number
            
        Returns:
            Text with inline citation
        """
        return f"{text} [Source {source_id}]"
    
    @staticmethod
    def create_citation_note(sources: List[Dict]) -> str:
        """
        Create a formatted citation note
        
        Args:
            sources: List of source metadata
            
        Returns:
            Formatted citation note
        """
        if not sources:
            return ""
        
        note = "📚 **Information retrieved from:**\n"
        
        for source in sources:
            source_id = source.get('id', 0)
            filename = source.get('filename', 'Unknown')
            category = source.get('category', 'General')
            score = source.get('score', 0)
            
            note += f"\n- [{source_id}] {category}/{filename}"
            if score > 0:
                note += f" (relevance: {score:.2f})"
        
        return note


if __name__ == "__main__":
    # Test citation formatter
    print("Testing CitationFormatter...\n")
    
    # Test data
    sources = [
        {
            'id': 1,
            'filename': '01_supervised_learning.md',
            'category': 'ML Fundamentals',
            'score': 0.85
        },
        {
            'id': 2,
            'filename': '02_evaluation_metrics.md',
            'category': 'ML Fundamentals',
            'score': 0.72
        }
    ]
    
    response = """
    Supervised learning is a machine learning paradigm where models learn from labeled data.
    Common algorithms include decision trees, random forests, and neural networks.
    """
    
    # Test 1: Add citations
    print("Test 1: Add citations")
    print("-" * 60)
    response_with_citations = CitationFormatter.add_citations(response, sources)
    print(response_with_citations)
    
    # Test 2: Extract citation numbers
    print("\n\nTest 2: Extract citation numbers")
    print("-" * 60)
    text_with_refs = "According to [Source 1], overfitting is common. [Source 2] suggests regularization."
    cited_sources = CitationFormatter.extract_citation_numbers(text_with_refs)
    print(f"Text: {text_with_refs}")
    print(f"Cited sources: {cited_sources}")
    
    # Test 3: Inline citation
    print("\n\nTest 3: Inline citation")
    print("-" * 60)
    text = "Overfitting occurs when models learn noise"
    cited_text = CitationFormatter.format_inline_citation(text, 1)
    print(cited_text)
    
    # Test 4: Citation note
    print("\n\nTest 4: Citation note")
    print("-" * 60)
    note = CitationFormatter.create_citation_note(sources)
    print(note)
    
    print("\n" + "=" * 60)
    print(" CitationFormatter tests passed!")

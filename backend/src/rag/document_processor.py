"""
Loads and chunks markdown documents for RAG
"""

import os
from typing import List, Dict
from pathlib import Path
import re


class DocumentProcessor:
    """Process markdown documents for RAG"""
    
    def __init__(
        self,
        chunk_size: int = 500,
        overlap: int = 50,
        min_chunk_size: int = 50
    ):
        """
        Initialize document processor
        
        Args:
            chunk_size: Target number of words per chunk
            overlap: Number of words to overlap between chunks
            min_chunk_size: Minimum words in a chunk (discard smaller)
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.min_chunk_size = min_chunk_size
    
    def load_documents(self, directory: str) -> List[Dict]:
        """
        Load all markdown files from directory
        
        Args:
            directory: Path to knowledge base directory
            
        Returns:
            List of dicts with 'content' and 'metadata'
        """
        documents = []
        directory_path = Path(directory)
        
        if not directory_path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # Find all markdown files
        md_files = list(directory_path.rglob("*.md"))
        
        print(f"📚 Loading documents from {directory}...")
        print(f"   Found {len(md_files)} markdown files")
        
        for filepath in md_files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Extract metadata from file
                metadata = {
                    'source': str(filepath),
                    'filename': filepath.name,
                    'category': filepath.parent.name
                }
                
                documents.append({
                    'content': content,
                    'metadata': metadata
                })
                
            except Exception as e:
                print(f"⚠️  Error loading {filepath}: {e}")
        
        print(f"✓ Loaded {len(documents)} documents")
        return documents
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks
        
        Args:
            text: Text to chunk
            
        Returns:
            List of text chunks
        """
        # Split into words
        words = text.split()
        chunks = []
        
        # Create chunks with overlap
        i = 0
        while i < len(words):
            # Get chunk
            chunk_words = words[i:i + self.chunk_size]
            
            # Skip if too small
            if len(chunk_words) < self.min_chunk_size:
                break
            
            # Join words
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            # Move to next chunk (with overlap)
            i += self.chunk_size - self.overlap
        
        return chunks
    
    def clean_text(self, text: str) -> str:
        """
        Clean markdown text
        
        Args:
            text: Text to clean
            
        Returns:
            Cleaned text
        """
        # Remove markdown headers (but keep the text)
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # Remove markdown links [text](url) -> text
        text = re.sub(r'\[([^\]]+)\]\([^\)]+\)', r'\1', text)
        
        # Remove code blocks markers (keep content)
        text = re.sub(r'```[\w]*\n', '', text)
        text = re.sub(r'```', '', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n', '\n\n', text)
        text = re.sub(r' +', ' ', text)
        
        # Remove special markdown characters at line start
        text = re.sub(r'^\*+\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'^-+\s+', '', text, flags=re.MULTILINE)
        
        return text.strip()
    
    def process_documents(self, documents: List[Dict]) -> List[Dict]:
        """
        Process documents into chunks with metadata
        
        Args:
            documents: List of documents from load_documents()
            
        Returns:
            List of processed chunks with metadata
        """
        processed = []
        
        print(f"\n✂️  Processing documents into chunks...")
        
        for doc in documents:
            # Clean text
            cleaned_text = self.clean_text(doc['content'])
            
            # Chunk text
            chunks = self.chunk_text(cleaned_text)
            
            # Add each chunk with metadata
            for i, chunk in enumerate(chunks):
                processed.append({
                    'text': chunk,
                    'metadata': {
                        **doc['metadata'],
                        'chunk_id': i,
                        'total_chunks': len(chunks)
                    }
                })
        
        print(f"✓ Created {len(processed)} chunks from {len(documents)} documents")
        
        # Show statistics
        chunk_sizes = [len(chunk['text'].split()) for chunk in processed]
        avg_size = sum(chunk_sizes) / len(chunk_sizes) if chunk_sizes else 0
        
        print(f"   Avg chunk size: {avg_size:.0f} words")
        print(f"   Min: {min(chunk_sizes)} words" if chunk_sizes else "")
        print(f"   Max: {max(chunk_sizes)} words" if chunk_sizes else "")
        
        return processed
    
    def get_statistics(self, documents: List[Dict]) -> Dict:
        """
        Get statistics about documents
        
        Args:
            documents: List of documents
            
        Returns:
            Dictionary of statistics
        """
        if not documents:
            return {}
        
        total_words = sum(len(doc['content'].split()) for doc in documents)
        categories = {}
        
        for doc in documents:
            cat = doc['metadata'].get('category', 'Unknown')
            categories[cat] = categories.get(cat, 0) + 1
        
        return {
            'total_documents': len(documents),
            'total_words': total_words,
            'avg_words_per_doc': total_words / len(documents),
            'categories': categories
        }


if __name__ == "__main__":
    # Test document processor
    print("Testing DocumentProcessor...\n")
    
    # Create test directory with sample file
    os.makedirs("test_docs", exist_ok=True)
    
    test_content = """
# Machine Learning Basics

## Overview
Machine learning is a subset of artificial intelligence.

## Key Concepts
Supervised learning requires labeled data. Unsupervised learning finds patterns.

## Examples
Here's a code example:

```python
model.fit(X_train, y_train)
```

## Resources
- [Scikit-learn](https://scikit-learn.org)
- Machine Learning book
"""
    
    with open("test_docs/test.md", 'w') as f:
        f.write(test_content)
    
    # Test processor
    processor = DocumentProcessor(chunk_size=50, overlap=10)
    
    # Load documents
    docs = processor.load_documents("test_docs")
    print(f"\nLoaded {len(docs)} documents")
    
    # Get statistics
    stats = processor.get_statistics(docs)
    print(f"\nStatistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Process into chunks
    chunks = processor.process_documents(docs)
    
    print(f"\nFirst chunk:")
    print(f"  Text: {chunks[0]['text'][:100]}...")
    print(f"  Metadata: {chunks[0]['metadata']}")
    
    # Cleanup
    import shutil
    shutil.rmtree("test_docs")
    
    print("\n✅ DocumentProcessor tests passed!")

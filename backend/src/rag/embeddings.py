"""
Converts text into vector embeddings for semantic search
"""

from sentence_transformers import SentenceTransformer
from typing import List, Union
import numpy as np


class EmbeddingGenerator:
    """Generate embeddings for text using sentence transformers"""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize embedding generator
        
        Args:
            model_name: Name of sentence transformer model
                       Default: all-MiniLM-L6-v2 (384 dimensions, fast, good quality)
        """
        print(f"🔧 Loading embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        print(f"✓ Model loaded (embedding dim: {self.embedding_dim})")
    
    def generate(self, texts: List[str], show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of text strings
            show_progress: Show progress bar
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        return embeddings # type: ignore
    
    def generate_single(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text
        
        Args:
            text: Text string
            
        Returns:
            numpy array of shape (embedding_dim,)
        """
        embedding = self.model.encode(
            text,
            show_progress_bar=False,
            convert_to_numpy=True
        )
        return embedding # type: ignore
    
    def generate_batch(
        self,
        texts: List[str],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Generate embeddings in batches (for large datasets)
        
        Args:
            texts: List of text strings
            batch_size: Number of texts per batch
            
        Returns:
            numpy array of shape (len(texts), embedding_dim)
        """
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.generate(batch, show_progress=False)
            all_embeddings.append(batch_embeddings)
            
            if i % (batch_size * 10) == 0:
                print(f"  Processed {i}/{len(texts)} texts...")
        
        return np.vstack(all_embeddings)


if __name__ == "__main__":
    # Test the embedding generator
    print("Testing EmbeddingGenerator...")
    
    embedder = EmbeddingGenerator()
    
    # Test single text
    text = "What is machine learning?"
    embedding = embedder.generate_single(text)
    print(f"\nSingle text embedding shape: {embedding.shape}")
    
    # Test multiple texts
    texts = [
        "Machine learning is a subset of AI",
        "Deep learning uses neural networks",
        "Supervised learning requires labeled data"
    ]
    embeddings = embedder.generate(texts, show_progress=False)
    print(f"Multiple texts embeddings shape: {embeddings.shape}")
    
    # Test similarity
    from numpy.linalg import norm
    
    similarity = np.dot(embeddings[0], embeddings[1]) / (
        norm(embeddings[0]) * norm(embeddings[1])
    )
    print(f"\nSimilarity between first two texts: {similarity:.3f}")
    
    print("\n✅ EmbeddingGenerator tests passed!")

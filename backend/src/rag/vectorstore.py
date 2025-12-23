"""
Supports both ChromaDB and FAISS for vector similarity search
"""

from typing import List, Dict, Optional
import numpy as np
import os
import pickle


class VectorStore:
    """Base class for vector stores"""
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict]
    ):
        """Add documents to vector store"""
        raise NotImplementedError
    
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5
    ) -> Dict:
        """Search for similar documents"""
        raise NotImplementedError
    
    def save(self, path: str):
        """Save vector store to disk"""
        raise NotImplementedError
    
    def load(self, path: str):
        """Load vector store from disk"""
        raise NotImplementedError


class ChromaVectorStore(VectorStore):
    """ChromaDB-based vector store"""
    
    def __init__(self, persist_dir: str = "./data/chromadb"):
        """
        Initialize ChromaDB vector store
        
        Args:
            persist_dir: Directory to persist database
        """
        try:
            import chromadb
            
            print(f"🔧 Initializing ChromaDB at {persist_dir}...")
            os.makedirs(persist_dir, exist_ok=True)
            
            self.client = chromadb.PersistentClient(path=persist_dir)
            self.collection = self.client.get_or_create_collection(
                name="researchpal_knowledge_base"
            )
            print("✓ ChromaDB initialized")
            
        except ImportError:
            raise ImportError(
                "ChromaDB not installed. Install with: "
                "pip install chromadb --break-system-packages"
            )
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict]
    ):
        """Add documents to ChromaDB"""
        n_docs = len(texts)
        ids = [f"doc_{i}" for i in range(n_docs)]
        
        self.collection.add(
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas, # type: ignore
            ids=ids
        )
        
        print(f"✓ Added {n_docs} documents to ChromaDB")
    
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5
    ) -> Dict:
        """Search ChromaDB for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        return {
            'documents': results['documents'],
            'metadatas': results['metadatas'],
            'distances': results['distances']
        }
    
    def count(self) -> int:
        """Get number of documents in collection"""
        return self.collection.count()


class FAISSVectorStore(VectorStore):
    """FAISS-based vector store"""
    
    def __init__(self, embedding_dim: int = 384):
        """
        Initialize FAISS vector store
        
        Args:
            embedding_dim: Dimension of embeddings
        """
        try:
            import faiss
            
            print(f"🔧 Initializing FAISS (dim={embedding_dim})...")
            
            # Use L2 distance (could also use cosine with IndexFlatIP)
            self.index = faiss.IndexFlatL2(embedding_dim)
            self.documents = []
            self.metadatas = []
            self.embedding_dim = embedding_dim
            
            print("✓ FAISS initialized")
            
        except ImportError:
            raise ImportError(
                "FAISS not installed. Install with: "
                "pip install faiss-cpu --break-system-packages"
            )
    
    def add_documents(
        self,
        texts: List[str],
        embeddings: np.ndarray,
        metadatas: List[Dict]
    ):
        """Add documents to FAISS index"""
        # Ensure embeddings are float32
        embeddings = embeddings.astype('float32')
        
        # Add to FAISS index
        self.index.add(embeddings) # type: ignore
        
        # Store documents and metadata
        self.documents.extend(texts)
        self.metadatas.extend(metadatas)
        
        print(f"✓ Added {len(texts)} documents to FAISS")
    
    def query(
        self,
        query_embedding: np.ndarray,
        n_results: int = 5
    ) -> Dict:
        """Search FAISS for similar documents"""
        # Ensure query is float32 and 2D
        query_embedding = query_embedding.astype('float32').reshape(1, -1)
        
        # Search
        distances, indices = self.index.search(query_embedding, n_results) # type: ignore
        
        # Get results
        results = {
            'documents': [[self.documents[i] for i in indices[0]]],
            'metadatas': [[self.metadatas[i] for i in indices[0]]],
            'distances': [distances[0].tolist()]
        }
        
        return results
    
    def save(self, path: str):
        """Save FAISS index and metadata"""
        import faiss
        
        os.makedirs(path, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, os.path.join(path, "index.faiss"))
        
        # Save metadata
        with open(os.path.join(path, "metadata.pkl"), 'wb') as f:
            pickle.dump({
                'documents': self.documents,
                'metadatas': self.metadatas,
                'embedding_dim': self.embedding_dim
            }, f)
        
        print(f"✓ FAISS index saved to {path}")
    
    def load(self, path: str):
        """Load FAISS index and metadata"""
        import faiss
        
        # Load FAISS index
        self.index = faiss.read_index(os.path.join(path, "index.faiss"))
        
        # Load metadata
        with open(os.path.join(path, "metadata.pkl"), 'rb') as f:
            data = pickle.load(f)
            self.documents = data['documents']
            self.metadatas = data['metadatas']
            self.embedding_dim = data['embedding_dim']
        
        print(f"✓ FAISS index loaded from {path}")
    
    def count(self) -> int:
        """Get number of documents in index"""
        return self.index.ntotal


def create_vector_store(
    backend: str = "auto",
    persist_dir: str = "./data/vectorstore",
    embedding_dim: int = 384
) -> VectorStore:
    """
    Factory function to create vector store
    
    Args:
        backend: "chromadb", "faiss", or "auto" (try ChromaDB first)
        persist_dir: Directory for persistence
        embedding_dim: Embedding dimension (for FAISS)
        
    Returns:
        VectorStore instance
    """
    if backend == "auto":
        # Try ChromaDB first
        try:
            return ChromaVectorStore(persist_dir)
        except ImportError:
            print("⚠️  ChromaDB not available, falling back to FAISS")
            return FAISSVectorStore(embedding_dim)
    
    elif backend == "chromadb":
        return ChromaVectorStore(persist_dir)
    
    elif backend == "faiss":
        return FAISSVectorStore(embedding_dim)
    
    else:
        raise ValueError(f"Unknown backend: {backend}")


if __name__ == "__main__":
    # Test vector stores
    print("Testing Vector Stores...\n")
    
    from embeddings import EmbeddingGenerator
    
    # Create test data
    texts = [
        "Machine learning is a subset of artificial intelligence",
        "Deep learning uses neural networks with multiple layers",
        "Supervised learning requires labeled training data"
    ]
    
    metadatas = [
        {'source': 'doc1.md', 'category': 'ML Fundamentals'},
        {'source': 'doc2.md', 'category': 'Deep Learning'},
        {'source': 'doc3.md', 'category': 'Supervised Learning'}
    ]
    
    # Generate embeddings
    embedder = EmbeddingGenerator()
    embeddings = embedder.generate(texts, show_progress=False)
    
    # Test both backends
    for backend in ["chromadb", "faiss"]:
        print(f"\n{'='*50}")
        print(f"Testing {backend.upper()}")
        print('='*50)
        
        try:
            # Create store
            if backend == "chromadb":
                store = ChromaVectorStore("./test_chromadb")
            else:
                store = FAISSVectorStore(embedding_dim=384)
            
            # Add documents
            store.add_documents(texts, embeddings, metadatas)
            print(f"Total documents: {store.count()}")
            
            # Query
            query = "What is supervised learning?"
            query_embedding = embedder.generate_single(query)
            
            results = store.query(query_embedding, n_results=2)
            
            print(f"\nQuery: {query}")
            print("Top 2 results:")
            for i, (doc, meta, dist) in enumerate(zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ), 1):
                print(f"\n{i}. [{meta['category']}] (distance: {dist:.3f})")
                print(f"   {doc[:100]}...")
            
            # Save (FAISS only)
            if backend == "faiss":
                store.save("./test_faiss")
                print("\n✓ Saved to disk")
            
            print(f"\n✅ {backend.upper()} tests passed!")
            
        except ImportError as e:
            print(f"⚠️  {backend.upper()} not available: {e}")
        except Exception as e:
            print(f"❌ {backend.upper()} test failed: {e}")
    
    print("\n" + "="*50)
    print("Testing complete!")

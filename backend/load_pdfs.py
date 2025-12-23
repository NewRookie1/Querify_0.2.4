import os
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vectorstore import create_vector_store
from src.rag.retriever import ConversationAwareRetriever
from src.utils.pdf_processor import PDFProcessor

PDF_FOLDER = "./pdfs"
VECTORSTORE_DIR = "./data/vectorstore"

# Initialize PDF processor and embedding generator
pdf_processor = PDFProcessor()
embedder = EmbeddingGenerator()

# Create vector store
vectorstore = create_vector_store(
    backend="auto",
    persist_dir=VECTORSTORE_DIR,
    embedding_dim=384,
)

# Loop through all PDFs
for filename in os.listdir(PDF_FOLDER):
    if filename.lower().endswith(".pdf"):
        filepath = os.path.join(PDF_FOLDER, filename)
        print(f"Processing {filename} ...")
        raw_text = pdf_processor.extract_text(filepath)

        # Split into chunks
        chunks = pdf_processor.chunk_text(raw_text, chunk_size=500)# type: ignore 

        for i, chunk in enumerate(chunks):
            emb = embedder.embed(chunk) # type: ignore
            vectorstore.add( # type: ignore
                documents=[chunk],
                embeddings=[emb],
                metadatas=[{"source": filename, "chunk": i}],
                ids=[f"{filename}-{i}"],
            )

# Save vector store to disk
if hasattr(vectorstore, "persist"):
    vectorstore.persist() # type: ignore

print("✅ Vector store created and saved.")

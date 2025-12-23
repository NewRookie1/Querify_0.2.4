from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import os
import tempfile
import shutil
from dotenv import load_dotenv 

from src.llm.gemini_client import EnhancedGeminiClient
from src.utils.agent_modes import AgentModeManager
from src.utils.conversation_memory import ConversationMemory
from src.rag.embeddings import EmbeddingGenerator
from src.rag.vectorstore import create_vector_store
from src.rag.retriever import ConversationAwareRetriever
from src.utils.citation_formatter import CitationFormatter
from src.utils.pdf_processor import PDFProcessor

load_dotenv()

# -------------------- Models --------------------

class ChatRequest(BaseModel):
    message: str
    mode: str = "Smart Mode (Auto-detect)"

class ChatResponse(BaseModel):
    reply: str
    used_context: bool
    sources: List[Dict[str, Any]] = Field(default_factory=list)
    active_mode: str

class PDFChatResponse(BaseModel):
    reply: str
    metadata: Dict[str, Any]

# -------------------- App --------------------

app = FastAPI(title="Backend", version="1.0.0")

#  FIXED CORS (LOCALHOST ONLY)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Globals --------------------

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

agent_manager = AgentModeManager()
gemini_client: Optional[EnhancedGeminiClient] = None
memory: Optional[ConversationMemory] = None
retriever: Optional[ConversationAwareRetriever] = None
citation_formatter: Optional[CitationFormatter] = None
pdf_processor: Optional[PDFProcessor] = None

USE_RAG = True

# -------------------- Init --------------------

def init_backend():
    global gemini_client, memory, retriever, citation_formatter, pdf_processor

    if not (GEMINI_API_KEY or GROQ_API_KEY):
        print("⚠️ No API key found. /chat disabled.")
        return

    try:
        print("🚀 Initializing backend...")
        gemini_client = EnhancedGeminiClient()
        memory = ConversationMemory()
        pdf_processor = PDFProcessor()

        if USE_RAG:
            try:
                embedder = EmbeddingGenerator()
                vectorstore = create_vector_store(
                    backend="auto",
                    persist_dir="./data/vectorstore",
                    embedding_dim=384,
                )

                if hasattr(vectorstore, "load"):
                    try:
                        vectorstore.load("./data/vectorstore")
                        print("✓ Vector store loaded")
                    except Exception:
                        print("⚠️ No existing vector store")

                retriever = ConversationAwareRetriever(
                    vectorstore,
                    embedder,
                    relevance_threshold=0.3
                )
                citation_formatter = CitationFormatter()
                print("✅ RAG ready")

            except Exception as e:
                print(f"⚠️ RAG init failed: {e}")
                retriever = None

    except Exception as e:
        print(f"❌ Backend init failed: {e}")

@app.on_event("startup")
async def on_startup():
    init_backend()

# -------------------- Routes --------------------

@app.get("/health")
async def health():
    if gemini_client is None:
        return {"status": "degraded", "reason": "LLM not initialized"}
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    if gemini_client is None:
        raise HTTPException(503, "LLM not initialized")

    message = request.message.strip()
    if not message:
        raise HTTPException(400, "Message cannot be empty")

    mode_display = request.mode
    mode_key = agent_manager.get_mode_key(mode_display)
    mode_prompt = agent_manager.get_system_prompt(mode_key)

    conversation_context = gemini_client.get_conversation_context(max_messages=2)

    response_text = ""
    used_context = False
    sources: List[Dict[str, Any]] = []

    if retriever and USE_RAG and retriever.is_query_in_knowledge_base(message):
        results, relevance_found = retriever.retrieve(
            message,
            n_results=3,
            use_conversation_context=True,
            require_relevance=True,
        )

        if relevance_found and results:
            context = retriever.format_context(results)
            sources = retriever.get_sources(results)

            response_text = gemini_client.generate_with_rag(
                query=message,
                context=context,
                conversation_context=conversation_context,
                has_relevant_context=True,
                system_prompt=mode_prompt,
            )

            if sources and citation_formatter:
                response_text = citation_formatter.add_citations(
                    response_text, sources
                )

            used_context = True

    if not response_text:
        response_text = gemini_client.generate_with_rag(
            query=message,
            context="",
            conversation_context=conversation_context,
            has_relevant_context=False,
            system_prompt=mode_prompt,
        )

    if memory:
        memory.add_message("user", message)
        memory.add_message("assistant", response_text)

    return ChatResponse(
        reply=response_text,
        used_context=used_context,
        sources=sources,
        active_mode=mode_display,
    )

@app.post("/chat/pdf", response_model=PDFChatResponse)
async def chat_with_pdf(
    file: UploadFile = File(...),
    message: str = Form(""),
):
    if gemini_client is None:
        raise HTTPException(503, "LLM not initialized")

    if pdf_processor is None:
        raise HTTPException(503, "PDF processing unavailable")

    if file.content_type not in ("application/pdf", "application/x-pdf"):
        raise HTTPException(400, "Only PDFs supported")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            shutil.copyfileobj(file.file, tmp)
            temp_path = tmp.name

        info = pdf_processor.get_pdf_info(temp_path)
        if "error" in info:
            raise HTTPException(400, info["error"])

        if not pdf_processor.check_has_text(temp_path):
            raise HTTPException(400, "PDF has no extractable text")

        raw_text = pdf_processor.extract_text(temp_path)

        # ----------------- ADD TO VECTOR STORE -----------------
        if retriever:
            # Split PDF manually into chunks of ~500 chars
            chunks = [raw_text[i:i+500] for i in range(0, len(raw_text), 500)]

            for i, chunk in enumerate(chunks):
                emb = retriever.embedder.embed(chunk)
                retriever.vectorstore.add(
                    documents=[chunk],
                    embeddings=[emb],
                    metadatas=[{"source": file.filename, "chunk": i}],
                    ids=[f"{file.filename}-{i}"],
                )

            # Save to disk
            if hasattr(retriever.vectorstore, "persist"):
                retriever.vectorstore.persist()
        # --------------------------------------------------------

        context = pdf_processor.format_for_llm(raw_text)

        reply = gemini_client.generate_with_rag(
            query=message or "Provide a summary of this PDF.",
            context=context,
            conversation_context="",
            has_relevant_context=True,
            system_prompt="You may ONLY use the PDF content.",
        )

        if memory:
            memory.add_message("user", f"[PDF] {message}")
            memory.add_message("assistant", reply)

        return PDFChatResponse(reply=reply, metadata=info)

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except Exception:
                pass

# -------------------- Run --------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

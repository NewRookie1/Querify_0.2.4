"""
Hybrid Groq + Gemini Client

Priority:
1. Groq (llama-3.3-70b-versatile) via GROQ_API_KEY
2. Gemini 2.0 Flash (gemini-2.0-flash) via GEMINI_API_KEY (free tier)

This client keeps the same public interface used by the rest of the app.
"""

from groq import Groq
from typing import List, Dict, Optional, Iterator
import os
import time
from functools import wraps

try:
    import google.generativeai as genai  # type: ignore
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


def retry_with_backoff(max_retries: int = 3, initial_delay: float = 1.0):
    """Decorator for retrying failed API calls with exponential backoff"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    error_str = str(e)
                    last_exception = e
                    
                    if "rate" in error_str.lower() or "429" in error_str:
                        if attempt < max_retries - 1:
                            print(f"⏳ Rate limit hit. Waiting {delay:.1f}s...")
                            time.sleep(delay)
                            delay *= 2
                            continue
                    else:
                        if attempt < max_retries - 1:
                            print(f"⚠️ Attempt {attempt + 1} failed: {str(e)[:100]}...")
                            time.sleep(delay)
                            delay *= 2
                        else:
                            print(f"❌ Max retries reached.")
            
            raise last_exception # type: ignore
        return wrapper
    return decorator


class EnhancedGeminiClient:
    """
    Hybrid client that prefers Groq and falls back to Gemini 2.0 Flash.
    Drop-in replacement - no changes needed to app.py or backend.
    """

    def __init__(
        self,
        model_name: str = "llama-3.3-70b-versatile",
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 40,
        max_output_tokens: int = 8192,
        gemini_model_name: str = "gemini-2.0-flash",
    ):
        # Priority: Groq first, then Gemini
        groq_key = os.getenv("GROQ_API_KEY")
        gemini_key = os.getenv("GEMINI_API_KEY")

        if not groq_key and not gemini_key:
            raise ValueError("Neither GROQ_API_KEY nor GEMINI_API_KEY found in environment variables")

        # Groq client (primary, if key present)
        self.groq_client: Optional[Groq] = Groq(api_key=groq_key) if groq_key else None

        # Gemini client (fallback)
        self.gemini_model_name = gemini_model_name
        self.gemini_client = None
        if gemini_key and GEMINI_AVAILABLE:
            genai.configure(api_key=gemini_key) # type: ignore
            self.gemini_client = genai.GenerativeModel(gemini_model_name) # type: ignore

        # Store configuration
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_output_tokens

        self.conversation_history = []
        self.last_request_time = 0
        self.min_request_interval = 0.5

        backends = []
        if self.groq_client:
            backends.append(f"Groq({model_name})")
        if self.gemini_client:
            backends.append(f"Gemini({self.gemini_model_name})")
        backends_str = " + ".join(backends) if backends else "None"
        print(f"✓ Enhanced hybrid client initialized with backends: {backends_str}")
    
    def _throttle_request(self):
        """Throttle requests to prevent hitting rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    @retry_with_backoff(max_retries=3)
    def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate a response to a prompt"""
        self._throttle_request()

        # Try Groq first, then Gemini
        last_error = None

        if self.groq_client is not None:
            try:
                completion = self.groq_client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                response_text = completion.choices[0].message.content
                self.add_to_conversation_history("assistant", response_text) # type: ignore
                return response_text # type: ignore
            except Exception as e:  # noqa: BLE001
                last_error = e
                err_str = str(e)
                if "rate limit" in err_str.lower() or "rate_limit_exceeded" in err_str.lower() or "429" in err_str:
                    print("⚠️ Groq rate limit hit, attempting Gemini fallback...")
                else:
                    print(f"⚠️ Groq generation failed, attempting Gemini fallback: {err_str[:120]}")

        if self.gemini_client is not None:
            try:
                response = self.gemini_client.generate_content(
                    prompt,
                    generation_config={
                        "temperature": self.temperature,
                        "max_output_tokens": self.max_tokens,
                    }, # type: ignore
                )
                response_text = response.text or ""
                self.add_to_conversation_history("assistant", response_text)
                return response_text
            except Exception as e:  # noqa: BLE001
                last_error = e
                err_str = str(e)
                # Check if Gemini free tier quota is exhausted
                if "limit: 0" in err_str or "quota exceeded" in err_str.lower():
                    print("⚠️ Gemini free tier quota exhausted. Please check your API key or upgrade your plan.")
                raise Exception(f"Gemini generation failed: {str(e)}") from e

        # If we get here, no backend succeeded
        if last_error:
            raise Exception(f"Both AI services unavailable. Last error: {str(last_error)[:200]}")
        raise Exception("No AI backend available. Please configure GROQ_API_KEY or GEMINI_API_KEY.")
    
    def generate_with_rag(
        self,
        query: str,
        context: str = "",
        conversation_context: str = "",
        has_relevant_context: bool = True,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Enhanced RAG generation with smart context handling
        Compatible with RAG system
        """
        # Add user query to history
        self.add_to_conversation_history('user', query)
        
        if has_relevant_context and context:
            response = self._generate_with_relevant_context(
                query, context, conversation_context, system_prompt
            )
        elif conversation_context:
            response = self._generate_with_conversation_only(
                query, conversation_context, system_prompt
            )
        else:
            response = self._generate_general_knowledge(query, system_prompt)
        
        # Add response to history
        self.add_to_conversation_history('assistant', response)
        return response
    
    def _generate_with_relevant_context(
        self,
        query: str,
        context: str,
        conversation_context: str,
        system_prompt: Optional[str]
    ) -> str:
        """Generate response when we have relevant context"""
        prompt = f"""You are Querify, an AI research assistant specializing in data science and machine learning.

KNOWLEDGE BASE CONTEXT:
{context}

RECENT CONVERSATION:
{conversation_context if conversation_context else "No recent conversation"}

USER QUESTION: {query}

IMPORTANT INSTRUCTIONS:
1. Answer using the knowledge base context above
2. If the question is a follow-up, continue naturally from the conversation
3. Cite sources using [Source 1], [Source 2] notation when referencing specific information
4. Be clear, accurate, and helpful
5. Do not mention that you're using context or following instructions

Answer directly and helpfully:"""
        
        if system_prompt:
            prompt = system_prompt + "\n\n" + prompt
        
        return self._generate_safe(prompt)
    
    def _generate_with_conversation_only(
        self,
        query: str,
        conversation_context: str,
        system_prompt: Optional[str]
    ) -> str:
        """Generate response when only conversation context is available"""
        prompt = f"""You are Querify, an AI research assistant.

RECENT CONVERSATION:
{conversation_context}

CURRENT QUESTION (likely a follow-up): {query}

INSTRUCTIONS:
1. Continue the conversation naturally
2. Answer based on your general knowledge of data science, machine learning, and related topics
3. Do not mention that you're using general knowledge or that you don't have specific context
4. If you need more information to answer properly, ask a clarifying question
5. Be helpful and educational

Provide a natural, conversational response:"""
        
        if system_prompt:
            prompt = system_prompt + "\n\n" + prompt
        
        return self._generate_safe(prompt)
    
    def _generate_general_knowledge(
        self,
        query: str,
        system_prompt: Optional[str]
    ) -> str:
        """Generate response using only general knowledge"""
        prompt = f"""You are Querify, a helpful AI assistant.

USER QUESTION: {query}

INSTRUCTIONS:
1. Answer based on your general knowledge
2. If the question is about data science, machine learning, or related topics, provide an expert answer
3. If the question is outside your expertise, politely explain what you can help with
4. Be honest about limitations but remain helpful
5. Do not mention that you're using general knowledge or lack specific context

Provide a helpful, accurate response:"""
        
        if system_prompt:
            prompt = system_prompt + "\n\n" + prompt
        
        return self._generate_safe(prompt)
    
    @retry_with_backoff(max_retries=3)
    def _generate_safe(self, prompt: str) -> str:
        """Safe generation with error handling, using Groq then Gemini fallback."""
        self._throttle_request()

        try:
            return self.generate(prompt)
        except Exception as e:  # noqa: BLE001
            error_str = str(e).lower()
            
            # Check if both APIs are rate-limited/quota exhausted
            is_rate_limit = (
                "rate limit" in error_str or 
                "rate_limit_exceeded" in error_str or 
                "429" in error_str or
                "quota" in error_str or
                "exceeded" in error_str
            )
            
            if is_rate_limit:
                error_msg = (
                    "I apologize, but I've temporarily reached the rate limit for both AI services. "
                    "Please wait a few minutes and try again. If this persists, you may need to "
                    "check your API quotas for Groq and/or Gemini."
                )
            else:
                error_msg = "I apologize, but I encountered an error while processing your request. Please try again."
            
            print(f"Generation error: {e}")
            return error_msg
    
    @retry_with_backoff(max_retries=3)
    def generate_with_multimodal(
        self,
        query: str,
        image = None,
        file_context: Optional[str] = None,
        conversation_context: str = ""
    ) -> str:
        """
        Generate response with multi-modal input.
        Note: Neither Groq nor Gemini free tier are used for true image inputs here;
        we treat everything as text + optional description.
        """
        self._throttle_request()

        # Build prompt
        prompt_parts = []
        
        system_msg = "You are Querify, an AI research assistant specializing in data science and machine learning."
        prompt_parts.append(system_msg)
        
        if conversation_context:
            prompt_parts.append(f"\nRECENT CONVERSATION:\n{conversation_context}\n")
        
        if file_context:
            prompt_parts.append(f"\nFILE CONTEXT:\n{file_context}\n")
        
        if image:
            prompt_parts.append("\nNOTE: An image was provided. Analyze based on the description and context given.\n")
        
        prompt_parts.append(f"\nUSER QUERY: {query}\n")
        
        instructions = """
INSTRUCTIONS:
1. Structure your response with clear sections using ## for main headers
2. Start with a brief overview/summary
3. Break down your analysis into logical sections
4. Use numbered lists for steps or recommendations
5. Highlight key insights and findings
6. End with actionable recommendations when appropriate
"""
        prompt_parts.append(instructions)
        
        try:
            full_prompt = "\n".join(prompt_parts)

            # Reuse the same Groq → Gemini fallback logic
            response_text = self.generate(full_prompt)

            self.add_to_conversation_history("user", query)
            self.add_to_conversation_history("assistant", response_text)

            return response_text

        except Exception as e:  # noqa: BLE001
            error_msg = "I apologize, but I encountered an error processing this file. Please try again."
            print(f"Multi-modal generation error: {e}")
            return error_msg
    
    def add_to_conversation_history(self, role: str, content: str):
        """Add message to conversation history"""
        self.conversation_history.append({
            'role': role,
            'content': content,
            'timestamp': time.time()
        })
        
        # Keep history size manageable
        if len(self.conversation_history) > 20:
            self.conversation_history.pop(0)
    
    def get_conversation_context(self, max_messages: int = 3) -> str:
        """Get recent conversation context"""
        if not self.conversation_history:
            return ""
        
        recent_messages = self.conversation_history[-max_messages:]
        context_parts = []
        
        for msg in recent_messages:
            role = "User" if msg['role'] == 'user' else "Assistant"
            context_parts.append(f"{role}: {msg['content'][:200]}")
        
        return "\n".join(context_parts)
    
    def clear_conversation_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("✓ Conversation history cleared")
    
    def get_conversation_history(self) -> List[Dict]:
        """Get current conversation history"""
        return self.conversation_history
    
    def __repr__(self) -> str:
        backends = []
        if self.groq_client:
            backends.append("Groq")
        if self.gemini_client:
            backends.append("Gemini")
        backend_str = "+".join(backends) if backends else "None"
        return f"EnhancedHybridClient(backends={backend_str}, primary_model={self.model_name})"

"""
Advanced Retriever with Conversation-Aware Memory and Relevance Detection
Enhanced to handle follow-up questions and filter irrelevant results
"""

from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import re


class ConversationAwareRetriever:
    """
    Semantic search with re-ranking and conversation memory
    Handles follow-up questions and filters irrelevant results
    """
    
    def __init__(self, vectorstore, embedder, relevance_threshold: float = 0.4):
        """
        Initialize conversation-aware retriever
        
        Args:
            vectorstore: VectorStore instance
            embedder: EmbeddingGenerator instance
            relevance_threshold: Minimum score for results to be considered relevant
        """
        self.vectorstore = vectorstore
        self.embedder = embedder
        self.relevance_threshold = relevance_threshold
        self.conversation_memory = []
        self.max_memory_items = 10
        
    def clear_memory(self):
        """Clear conversation memory"""
        self.conversation_memory = []
    
    def add_to_memory(self, user_query: str, retrieved_context: str, relevant: bool = True):
        """
        Add query and context to conversation memory
        
        Args:
            user_query: User's question
            retrieved_context: Retrieved context for that question
            relevant: Whether the retrieved context was relevant
        """
        memory_item = {
            'timestamp': datetime.now(),
            'query': user_query,
            'context': retrieved_context[:500] if retrieved_context else "",
            'keywords': self._extract_keywords(user_query),
            'relevant': relevant
        }
        
        self.conversation_memory.append(memory_item)
        
        # Keep memory size limited
        if len(self.conversation_memory) > self.max_memory_items:
            self.conversation_memory.pop(0)
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract important keywords from text"""
        stop_words = {'what', 'how', 'why', 'when', 'where', 'who', 'is', 'are', 'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'as', 'it', 'its'}
        words = text.lower().split()
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords[:5]
    
    def _is_follow_up(self, query: str) -> Tuple[bool, str]:
        """
        Determine if query is a follow-up question
        
        Args:
            query: Current user query
            
        Returns:
            Tuple of (is_follow_up, enhanced_query)
        """
        if not self.conversation_memory:
            return False, query
        
        # Check for follow-up indicators
        follow_up_indicators = ['it', 'this', 'that', 'they', 'those', 'there', 'here', 'above', 'previous', 'earlier', 'also', 'too', 'more', 'explain']
        query_lower = query.lower()
        
        # Check if query contains pronouns or references
        has_pronouns = any(indicator in query_lower.split() for indicator in follow_up_indicators)
        
        # Check if query is very short (likely follow-up)
        words = query.split()
        is_short = len(words) <= 3
        
        # Check if query starts with "And", "But", "So", "Then"
        is_continuation = words[0].lower() in ['and', 'but', 'so', 'then', 'well', 'okay'] if words else False
        
        if not (has_pronouns or is_short or is_continuation):
            return False, query
        
        # Get last conversation context
        last_memory = self.conversation_memory[-1]
        previous_context = last_memory['context']
        previous_keywords = last_memory['keywords']
        
        if not previous_context:
            return False, query
        
        # Enhance query with previous context keywords
        enhanced_keywords = ' '.join(previous_keywords)
        enhanced_query = f"{query} {enhanced_keywords}"
        
        return True, enhanced_query
    
    def _calculate_relevance_score(self, query: str, result: Dict) -> float:
        """
        Calculate relevance score for a result
        
        Args:
            query: Original query
            result: Retrieved document
            
        Returns:
            Relevance score between 0 and 1
        """
        text = result.get('text', '').lower()
        query_lower = query.lower()
        
        # Extract meaningful words from query (remove stop words)
        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
        
        # Calculate word overlap
        text_words = set(re.findall(r'\b\w{3,}\b', text))
        overlap = len(query_words & text_words)
        max_possible = max(len(query_words), 1)
        
        # Word overlap score
        overlap_score = overlap / max_possible
        
        # Exact phrase match bonus
        query_no_punct = query_lower.replace('?', '').replace('.', '')
        phrase_score = 0.3 if query_no_punct in text else 0.0
        
        # Distance-based score (from vector similarity)
        distance = result.get('distance', 1.0)
        similarity_score = 1.0 / (1.0 + distance)
        
        # Combined score with weights
        final_score = (overlap_score * 0.4) + (phrase_score * 0.3) + (similarity_score * 0.3)
        
        return final_score
    
    def retrieve(
        self,
        query: str,
        n_results: int = 5,
        use_conversation_context: bool = True,
        require_relevance: bool = True,
        mode: Optional[str] = None
    ) -> Tuple[List[Dict], bool]:
        """
        Retrieve relevant documents with conversation awareness
        
        Args:
            query: User question
            n_results: Number of results to return
            use_conversation_context: Whether to use conversation memory
            require_relevance: Whether to filter by relevance threshold
            mode: Optional mode filter
            
        Returns:
            Tuple of (list of relevant documents, relevance_found)
        """
        # Check if this is a follow-up question
        enhanced_query = query
        if use_conversation_context and self.conversation_memory:
            is_follow_up, enhanced_query = self._is_follow_up(query)
            if is_follow_up:
                print(f"🔍 Detected follow-up question. Enhanced query: {enhanced_query}")
        
        # Generate query embedding (using enhanced query)
        query_embedding = self.embedder.generate_single(enhanced_query)
        
        # Search vector store
        search_results = min(n_results * 2, 10)
        raw_results = self.vectorstore.query(
            query_embedding=query_embedding,
            n_results=search_results
        )
        
        # Normalize results
        results = self._normalize_results(raw_results)
        
        # If no results found and it's not a follow-up, try with original query
        if not results and enhanced_query != query:
            print("⚠️ No results with enhanced query, trying original query...")
            query_embedding = self.embedder.generate_single(query)
            raw_results = self.vectorstore.query(
                query_embedding=query_embedding,
                n_results=search_results
            )
            results = self._normalize_results(raw_results)
        
        if not results:
            return [], False
        
        # Calculate relevance scores for all results
        for result in results:
            result['relevance_score'] = self._calculate_relevance_score(query, result)
        
        # Filter by relevance threshold if required
        if require_relevance:
            relevant_results = [r for r in results if r['relevance_score'] >= self.relevance_threshold]
            relevance_found = len(relevant_results) > 0
        else:
            relevant_results = results
            relevance_found = any(r['relevance_score'] >= self.relevance_threshold for r in results)
        
        # Re-rank relevant results
        ranked_results = self._rerank(query, relevant_results)
        
        # Store in conversation memory
        if ranked_results and use_conversation_context:
            context_text = self.format_context(ranked_results[:3])
            self.add_to_memory(query, context_text, relevance_found)
        
        return ranked_results[:n_results], relevance_found
    
    def _normalize_results(self, results) -> List[Dict]:
        """
        Convert raw vector store output (Dict) to List[Dict]
        """
        if isinstance(results, list):
            return results
            
        if isinstance(results, dict):
            normalized = []
            
            documents = results.get('documents', [[]])
            metadatas = results.get('metadatas', [[]])
            distances = results.get('distances', [[]])
            ids = results.get('ids', [[]])

            if not documents or not documents[0]:
                return []

            docs_list = documents[0]
            metas_list = metadatas[0] if metadatas else [{}] * len(docs_list)
            dists_list = distances[0] if distances else [0.0] * len(docs_list)
            ids_list = ids[0] if ids else [""] * len(docs_list)

            for i in range(len(docs_list)):
                normalized.append({
                    'text': docs_list[i],
                    'metadata': metas_list[i] if i < len(metas_list) else {},
                    'distance': dists_list[i] if i < len(dists_list) else 0.0,
                    'id': ids_list[i] if i < len(ids_list) else "",
                    'relevance_score': 0.0
                })
            
            return normalized
            
        return []

    def _rerank(self, query: str, results: List[Dict]) -> List[Dict]:
        """
        Re-rank results based on query relevance with conversation context
        """
        if not results:
            return []
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
        
        # Get conversation keywords from memory
        conversation_keywords = set()
        if self.conversation_memory:
            for memory_item in self.conversation_memory[-2:]:
                conversation_keywords.update(memory_item['keywords'])
        
        for result in results:
            text = result.get('text', '').lower()
            relevance_score = result.get('relevance_score', 0.0)
            distance = result.get('distance', 1.0)
            
            # Base score from relevance
            base_score = relevance_score
            
            # Keyword overlap bonus
            text_words = set(re.findall(r'\b\w{3,}\b', text))
            overlap = len(query_words & text_words)
            keyword_bonus = min(overlap * 0.05, 0.2)
            
            # Conversation context bonus
            conversation_overlap = len(conversation_keywords & text_words)
            conversation_bonus = min(conversation_overlap * 0.03, 0.15)
            
            # Exact phrase bonus
            query_no_punct = query_lower.replace('?', '').replace('.', '')
            phrase_bonus = 0.2 if query_no_punct in text else 0.0
            
            # Length penalty (prefer concise but informative)
            text_length = len(text.split())
            length_penalty = -0.1 if text_length > 200 else 0.0
            
            # Combined score
            result['final_score'] = base_score + keyword_bonus + conversation_bonus + phrase_bonus + length_penalty
        
        # Sort by final score descending
        ranked = sorted(results, key=lambda x: x.get('final_score', 0), reverse=True)
        
        return ranked
    
    def format_context(self, results: List[Dict]) -> str:
        """
        Format retrieved documents as context string
        """
        if not results:
            return ""
        
        context_parts = []
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            if metadata is None:
                metadata = {}
                
            category = metadata.get('category', 'unknown')
            filename = metadata.get('filename', 'unknown')
            text = result.get('text', '')
            
            # Add relevance indicator
            relevance = result.get('relevance_score', 0)
            relevance_star = "⭐" if relevance >= 0.7 else "✓" if relevance >= 0.4 else "⚠️"
            
            context_parts.append(
                f"[Source {i}: {category}/{filename}] {relevance_star}\n{text}\n"
            )
        
        return "\n".join(context_parts)
    
    def get_sources(self, results: List[Dict]) -> List[Dict]:
        """
        Extract source information from results
        """
        sources = []
        for i, result in enumerate(results, 1):
            metadata = result.get('metadata', {})
            if metadata is None:
                metadata = {}
                
            sources.append({
                'number': i,
                'category': metadata.get('category', 'unknown'),
                'filename': metadata.get('filename', 'unknown'),
                'relevance': result.get('relevance_score', 0.0),
                'final_score': result.get('final_score', 0.0)
            })
        return sources
    
    def get_conversation_summary(self) -> str:
        """
        Get a summary of the conversation memory
        """
        if not self.conversation_memory:
            return "No conversation history."
        
        summary_parts = []
        for i, memory in enumerate(self.conversation_memory[-3:], 1):
            query_preview = memory['query'][:30] + "..." if len(memory['query']) > 30 else memory['query']
            relevance_indicator = "✓" if memory.get('relevant', True) else "⚠️"
            summary_parts.append(f"{relevance_indicator} {query_preview}")
        
        return " | ".join(summary_parts)
    
    def is_query_in_knowledge_base(self, query: str) -> bool:
        """
        Check if a query is likely to be found in the knowledge base
        
        Args:
            query: User query
            
        Returns:
            True if query is likely in knowledge base, False otherwise
        """
        # Check if query contains topics likely to be in our knowledge base
        kb_topics = {
            'machine learning', 'ml', 'ai', 'artificial intelligence',
            'deep learning', 'neural network', 'statistics',
            'data science', 'python', 'pandas', 'numpy', 'scikit-learn',
            'tensorflow', 'pytorch', 'regression', 'classification',
            'clustering', 'gradient descent', 'overfitting', 'cross validation',
            'feature engineering', 'model evaluation', 'bias variance',
            'supervised', 'unsupervised', 'reinforcement learning'
        }
        
        query_lower = query.lower()
        query_words = set(re.findall(r'\b\w{3,}\b', query_lower))
        
        # Check for overlap with known topics
        topic_overlap = len(query_words & kb_topics)
        
        return topic_overlap > 0
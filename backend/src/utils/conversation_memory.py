"""
Conversation Memory Management
Handles conversation history, context, and session state
"""

from typing import List, Dict, Optional, Any
from datetime import datetime
import json
import uuid


class ConversationMemory:
    """
    Manages conversation state and history
    
    Features:
    - Store message history
    - Track uploaded files and context
    - Export conversations
    - Manage token limits
    - Session persistence
    """
    
    def __init__(self, max_history: int = 20, max_tokens: int = 100000):
        """
        Initialize conversation memory
        
        Args:
            max_history: Maximum number of messages to keep
            max_tokens: Approximate token limit for context
        """
        self.max_history = max_history
        self.max_tokens = max_tokens
        
        # Message history
        self.messages: List[Dict[str, Any]] = []
        
        # Context data (uploaded files, datasets, etc.)
        self.context_data: Dict[str, Any] = {}
        
        # Session metadata
        self.conversation_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        print(f"✓ Conversation memory initialized (ID: {self.conversation_id[:8]}...)")
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a message to conversation history
        
        Args:
            role: 'user' or 'assistant'
            content: Message text
            metadata: Optional metadata (sources, mode, etc.)
        """
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        self.messages.append(message)
        self.last_updated = datetime.now()
        
        # Trim history if too long
        if len(self.messages) > self.max_history:
            # Keep system message if present, then trim oldest
            if self.messages[0].get("role") == "system":
                self.messages = [self.messages[0]] + self.messages[-(self.max_history-1):]
            else:
                self.messages = self.messages[-self.max_history:]
    
    def add_user_message(self, content: str, metadata: Optional[Dict] = None):
        """Add a user message"""
        self.add_message("user", content, metadata)
    
    def add_assistant_message(self, content: str, metadata: Optional[Dict] = None):
        """Add an assistant message"""
        self.add_message("assistant", content, metadata)
    
    def add_system_message(self, content: str):
        """Add a system message (usually at start)"""
        self.add_message("system", content)
    
    def get_history(
        self,
        include_system: bool = True,
        last_n: Optional[int] = None
    ) -> List[Dict]:
        """
        Get conversation history
        
        Args:
            include_system: Whether to include system messages
            last_n: Return only last N messages
            
        Returns:
            List of message dictionaries
        """
        messages = self.messages.copy()
        
        if not include_system:
            messages = [m for m in messages if m["role"] != "system"]
        
        if last_n:
            messages = messages[-last_n:]
        
        return messages
    
    def get_history_for_llm(self, format_type: str = "gemini") -> List[Dict]:
        """
        Format history for LLM API
        
        Args:
            format_type: 'gemini', 'openai', or 'anthropic'
            
        Returns:
            Formatted message history
        """
        if format_type == "gemini":
            # Gemini format: {"role": "user"/"model", "parts": ["text"]}
            formatted = []
            for msg in self.messages:
                if msg["role"] == "system":
                    continue  # Skip system messages for now
                
                role = "user" if msg["role"] == "user" else "model"
                formatted.append({
                    "role": role,
                    "parts": [msg["content"]]
                })
            return formatted
        
        elif format_type == "openai":
            # OpenAI format: {"role": "user"/"assistant"/"system", "content": "text"}
            return [
                {
                    "role": msg["role"],
                    "content": msg["content"]
                }
                for msg in self.messages
            ]
        
        else:
            # Default: return as-is
            return self.messages
    
    def add_context_data(self, key: str, data: Any) -> None:
        """
        Store additional context (uploaded files, datasets, etc.)
        
        Args:
            key: Context identifier (e.g., 'dataset', 'image')
            data: Context data to store
        """
        self.context_data[key] = {
            "data": data,
            "added_at": datetime.now().isoformat()
        }
        
        print(f"✓ Context data added: {key}")
    
    def get_context_data(self, key: str) -> Optional[Any]:
        """Get stored context data"""
        context = self.context_data.get(key)
        return context["data"] if context else None
    
    def has_context(self, key: str) -> bool:
        """Check if context exists"""
        return key in self.context_data
    
    def get_context_summary(self) -> str:
        """
        Generate human-readable summary of active context
        
        Returns:
            Summary string
        """
        if not self.context_data:
            return "No additional context"
        
        summary_parts = ["Active Context:"]
        
        for key, value in self.context_data.items():
            data = value["data"]
            
            # Summarize based on type
            if isinstance(data, dict):
                if 'shape' in data:  # Dataset info
                    summary_parts.append(
                        f"  - Dataset: {data['shape'][0]} rows × {data['shape'][1]} cols"
                    )
                else:
                    summary_parts.append(f"  - {key}: {len(data)} items")
            
            elif isinstance(data, str):
                length = len(data)
                summary_parts.append(f"  - {key}: {length} characters")
            
            else:
                summary_parts.append(f"  - {key}: {type(data).__name__}")
        
        return "\n".join(summary_parts)
    
    def estimate_tokens(self) -> int:
        """
        Estimate total tokens in conversation history
        
        Returns:
            Approximate token count
        """
        # Rough estimate: ~4 characters per token
        total_chars = sum(len(msg["content"]) for msg in self.messages)
        return total_chars // 4
    
    def is_token_limit_exceeded(self) -> bool:
        """Check if conversation exceeds token limit"""
        return self.estimate_tokens() > self.max_tokens
    
    def export_conversation(
        self,
        format_type: str = "json",
        include_context: bool = True
    ) -> str:
        """
        Export conversation for download
        
        Args:
            format_type: 'json' or 'text'
            include_context: Include context data summary
            
        Returns:
            Formatted export string
        """
        export_data = {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at.isoformat(),
            "last_updated": self.last_updated.isoformat(),
            "message_count": len(self.messages),
            "messages": []
        }
        
        # Add messages
        for msg in self.messages:
            export_data["messages"].append({
                "role": msg["role"],
                "content": msg["content"],
                "timestamp": msg["timestamp"],
                "metadata": msg.get("metadata", {})
            })
        
        # Add context summary
        if include_context:
            export_data["context_summary"] = self.get_context_summary()
        
        if format_type == "json":
            return json.dumps(export_data, indent=2)
        
        elif format_type == "text":
            # Plain text format
            lines = [
                f"Conversation Export",
                f"ID: {self.conversation_id}",
                f"Date: {self.created_at.strftime('%Y-%m-%d %H:%M')}",
                f"Messages: {len(self.messages)}",
                "=" * 60,
                ""
            ]
            
            for msg in self.messages:
                role = msg["role"].upper()
                content = msg["content"]
                timestamp = msg["timestamp"]
                
                lines.append(f"[{timestamp}] {role}:")
                lines.append(content)
                lines.append("-" * 60)
                lines.append("")
            
            if include_context:
                lines.append(self.get_context_summary())
            
            return "\n".join(lines)
        
        else:
            raise ValueError(f"Unknown format: {format_type}")
    
    def clear(self) -> None:
        """Clear all conversation data"""
        self.messages = []
        self.context_data = {}
        self.conversation_id = str(uuid.uuid4())
        self.created_at = datetime.now()
        self.last_updated = datetime.now()
        
        print("✓ Conversation cleared")
    
    def get_last_message(self, role: Optional[str] = None) -> Optional[Dict]:
        """
        Get the last message, optionally filtered by role
        
        Args:
            role: Filter by role ('user', 'assistant', or None for any)
            
        Returns:
            Last message dictionary or None
        """
        if not self.messages:
            return None
        
        if role:
            for msg in reversed(self.messages):
                if msg["role"] == role:
                    return msg
            return None
        
        return self.messages[-1]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation statistics"""
        user_msgs = sum(1 for m in self.messages if m["role"] == "user")
        assistant_msgs = sum(1 for m in self.messages if m["role"] == "assistant")
        
        return {
            "conversation_id": self.conversation_id,
            "created_at": self.created_at,
            "duration_minutes": (datetime.now() - self.created_at).total_seconds() / 60,
            "total_messages": len(self.messages),
            "user_messages": user_msgs,
            "assistant_messages": assistant_msgs,
            "estimated_tokens": self.estimate_tokens(),
            "context_items": len(self.context_data)
        }
    
    def __repr__(self) -> str:
        return f"ConversationMemory(messages={len(self.messages)}, context_items={len(self.context_data)})"


# Example usage
if __name__ == "__main__":
    # Create memory instance
    memory = ConversationMemory()
    
    # Add some messages
    memory.add_user_message("What is machine learning?")
    memory.add_assistant_message("Machine learning is a field of AI that enables computers to learn from data.")
    
    memory.add_user_message("Can you give me an example?")
    memory.add_assistant_message("Sure! Email spam filtering is a classic example of machine learning.")
    
    # Add context
    memory.add_context_data("dataset", {"shape": (1000, 10), "columns": ["feature1", "feature2"]})
    
    # Get history
    print("History:", memory.get_history())
    print("\nContext Summary:", memory.get_context_summary())
    print("\nStats:", memory.get_stats())
    
    # Export
    export_json = memory.export_conversation(format_type="json")
    print("\nExported (JSON):", export_json[:200], "...")

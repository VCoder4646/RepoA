"""
Chat Module
Manages conversation history with system prompt and local persistence.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json

from ..core.message import Message, Role, system_message, user_message, agent_message, tool_message


class Chat:
    """
    Manages a conversation with system prompt and message history.
    
    The Chat class stores:
    - System prompt (stored once, not repeated)
    - Message history (user, agent, tool messages)
    - Conversation metadata
    - Ability to save/load from disk
    
    Features:
    - Single system prompt storage
    - Message management
    - Token tracking
    - Local persistence (JSON format)
    - Statistics and analytics
    """
    
    def __init__(
        self,
        system_prompt: Optional[str] = None,
        chat_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a chat session.
        
        Args:
            system_prompt: System prompt for the conversation
            chat_id: Unique chat identifier (auto-generated if not provided)
            metadata: Additional metadata (tags, user_id, etc.)
        """
        self.chat_id = chat_id or self._generate_chat_id()
        self.created_at = datetime.now()
        self.updated_at = self.created_at
        self.metadata = metadata or {}
        
        # System prompt (stored once)
        self._system_prompt: Optional[Message] = None
        if system_prompt:
            self.set_system_prompt(system_prompt)
        
        # Message history (excludes system prompt)
        self.messages: List[Message] = []
        
        # Statistics
        self.stats = {
            "total_messages": 0,
            "user_messages": 0,
            "agent_messages": 0,
            "tool_calls": 0,
            "total_tokens": 0
        }
    
    def _generate_chat_id(self) -> str:
        """Generate a unique chat ID."""
        return f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    
    def set_system_prompt(self, prompt: str) -> None:
        """
        Set or update the system prompt.
        
        System prompt is stored separately and included once at the
        beginning of conversation context.
        
        Args:
            prompt: System prompt text
        """
        self._system_prompt = system_message(prompt)
        self.updated_at = datetime.now()
    
    def get_system_prompt(self) -> Optional[str]:
        """
        Get the system prompt text.
        
        Returns:
            System prompt text or None if not set
        """
        return self._system_prompt.content if self._system_prompt else None
    
    def add_message(self, message: Message) -> None:
        """
        Add a message to the chat.
        
        Args:
            message: Message to add
        """
        # Don't add system messages (use set_system_prompt instead)
        if message.is_system():
            raise ValueError("Use set_system_prompt() to set system messages")
        
        self.messages.append(message)
        self.updated_at = datetime.now()
        
        # Update statistics
        self.stats["total_messages"] += 1
        self.stats["total_tokens"] += message.token_count
        
        if message.is_user():
            self.stats["user_messages"] += 1
        elif message.is_agent():
            self.stats["agent_messages"] += 1
            if message.has_tool_calls():
                self.stats["tool_calls"] += len(message.tool_calls)
    
    def add_user_message(self, content: str, metadata: Optional[Dict[str, Any]] = None) -> Message:
        """
        Add a user message.
        
        Args:
            content: User message content
            metadata: Optional metadata
            
        Returns:
            Created Message
        """
        msg = user_message(content, metadata)
        self.add_message(msg)
        return msg
    
    def add_agent_message(
        self,
        content: str,
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add an agent message.
        
        Args:
            content: Agent message content
            tool_calls: Optional tool calls
            metadata: Optional metadata
            
        Returns:
            Created Message
        """
        msg = agent_message(content, tool_calls, metadata)
        self.add_message(msg)
        return msg
    
    def add_tool_message(
        self,
        content: str,
        tool_call_id: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Message:
        """
        Add a tool response message.
        
        Args:
            content: Tool response content
            tool_call_id: ID of tool call this responds to
            metadata: Optional metadata
            
        Returns:
            Created Message
        """
        msg = tool_message(content, tool_call_id, metadata)
        self.add_message(msg)
        return msg
    
    def get_messages(self, include_system: bool = True) -> List[Dict[str, Any]]:
        """
        Get messages in LLM API format.
        
        Args:
            include_system: Whether to include system prompt
            
        Returns:
            List of message dictionaries ready for LLM
        """
        messages = []
        
        # Add system prompt first if requested and exists
        if include_system and self._system_prompt:
            messages.append(self._system_prompt.to_dict())
        
        # Add all other messages
        for msg in self.messages:
            messages.append(msg.to_dict())
        
        return messages
    
    def get_recent_messages(self, n: int, include_system: bool = True) -> List[Dict[str, Any]]:
        """
        Get the N most recent messages.
        
        Args:
            n: Number of recent messages to get
            include_system: Whether to include system prompt
            
        Returns:
            List of recent message dictionaries
        """
        messages = []
        
        if include_system and self._system_prompt:
            messages.append(self._system_prompt.to_dict())
        
        # Get last N messages
        recent = self.messages[-n:] if n < len(self.messages) else self.messages
        for msg in recent:
            messages.append(msg.to_dict())
        
        return messages
    
    def get_token_count(self) -> int:
        """
        Get total token count of conversation.
        
        Returns:
            Total tokens (including system prompt)
        """
        total = sum(msg.token_count for msg in self.messages)
        if self._system_prompt:
            total += self._system_prompt.token_count
        return total
    
    def get_message_count(self) -> int:
        """
        Get total number of messages (excluding system prompt).
        
        Returns:
            Message count
        """
        return len(self.messages)
    
    def clear_messages(self, keep_system: bool = True) -> None:
        """
        Clear all messages.
        
        Args:
            keep_system: Whether to keep system prompt
        """
        self.messages.clear()
        if not keep_system:
            self._system_prompt = None
        
        # Reset stats
        self.stats = {
            "total_messages": 0,
            "user_messages": 0,
            "agent_messages": 0,
            "tool_calls": 0,
            "total_tokens": 0
        }
        
        self.updated_at = datetime.now()
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get conversation statistics.
        
        Returns:
            Dictionary of statistics
        """
        return {
            **self.stats,
            "chat_id": self.chat_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "total_tokens": self.get_token_count(),
            "has_system_prompt": self._system_prompt is not None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert chat to dictionary for persistence.
        
        Returns:
            Complete dictionary representation
        """
        return {
            "chat_id": self.chat_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "system_prompt": self._system_prompt.to_full_dict() if self._system_prompt else None,
            "messages": [msg.to_full_dict() for msg in self.messages],
            "stats": self.stats
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Chat':
        """
        Create a Chat from dictionary.
        
        Args:
            data: Dictionary containing chat data
            
        Returns:
            Chat instance
        """
        chat = cls(
            chat_id=data["chat_id"],
            metadata=data.get("metadata", {})
        )
        
        # Restore timestamps
        try:
            chat.created_at = datetime.fromisoformat(data["created_at"])
            chat.updated_at = datetime.fromisoformat(data["updated_at"])
        except (ValueError, TypeError, KeyError):
            pass
        
        # Restore system prompt
        if data.get("system_prompt"):
            chat._system_prompt = Message.from_dict(data["system_prompt"])
        
        # Restore messages
        for msg_data in data.get("messages", []):
            msg = Message.from_dict(msg_data)
            chat.messages.append(msg)
        
        # Restore stats
        chat.stats = data.get("stats", chat.stats)
        
        return chat
    
    def save(self, filepath: str) -> None:
        """
        Save chat to a local JSON file.
        
        Args:
            filepath: Path to save file
        """
        path = Path(filepath)
        
        # Create directory if it doesn't exist
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load(cls, filepath: str) -> 'Chat':
        """
        Load chat from a local JSON file.
        
        Args:
            filepath: Path to load file
            
        Returns:
            Chat instance
        """
        path = Path(filepath)
        
        if not path.exists():
            raise FileNotFoundError(f"Chat file not found: {filepath}")
        
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls.from_dict(data)
    
    def export_text(self, include_system: bool = True) -> str:
        """
        Export conversation as readable text.
        
        Args:
            include_system: Whether to include system prompt
            
        Returns:
            Formatted text representation
        """
        lines = []
        lines.append(f"=== Chat {self.chat_id} ===")
        lines.append(f"Created: {self.created_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append(f"Updated: {self.updated_at.strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        if include_system and self._system_prompt:
            lines.append("[SYSTEM]")
            lines.append(self._system_prompt.content)
            lines.append("")
        
        for msg in self.messages:
            lines.append(f"[{msg.role.upper()}]")
            lines.append(msg.content)
            
            if msg.has_tool_calls():
                lines.append(f"  (Tool calls: {len(msg.tool_calls)})")
            
            lines.append("")
        
        lines.append(f"=== Stats: {self.get_message_count()} messages, {self.get_token_count()} tokens ===")
        
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"Chat(id={self.chat_id}, messages={len(self.messages)}, tokens={self.get_token_count()})"
    
    def __str__(self) -> str:
        """User-friendly string."""
        return self.export_text()
    
    def __len__(self) -> int:
        """Return message count."""
        return len(self.messages)


class ChatManager:
    """
    Manages multiple chat sessions with persistence.
    """
    
    def __init__(self, storage_dir: str = ""):
        """
        Initialize chat manager.
        
        Args:
            storage_dir: Directory to store chat files
        """
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.chats: Dict[str, Chat] = {}
    
    def create_chat(
        self,
        system_prompt: Optional[str] = None,
        chat_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Chat:
        """
        Create a new chat session.
        
        Args:
            system_prompt: System prompt
            chat_id: Optional chat ID
            metadata: Optional metadata
            
        Returns:
            New Chat instance
        """
        chat = Chat(system_prompt, chat_id, metadata)
        self.chats[chat.chat_id] = chat
        return chat
    
    def get_chat(self, chat_id: str) -> Optional[Chat]:
        """
        Get a chat by ID.
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            Chat instance or None
        """
        return self.chats.get(chat_id)
    
    def save_chat(self, chat_id: str) -> None:
        """
        Save a chat to disk.
        
        Args:
            chat_id: Chat identifier
        """
        chat = self.chats.get(chat_id)
        if not chat:
            raise ValueError(f"Chat not found: {chat_id}")
        
        filepath = self.storage_dir / f"{chat_id}.json"
        chat.save(str(filepath))
    
    def load_chat(self, chat_id: str) -> Chat:
        """
        Load a chat from disk.
        
        Args:
            chat_id: Chat identifier
            
        Returns:
            Loaded Chat instance
        """
        filepath = self.storage_dir / f"{chat_id}.json"
        chat = Chat.load(str(filepath))
        self.chats[chat.chat_id] = chat
        return chat
    
    def list_chats(self) -> List[str]:
        """
        List all saved chat IDs.
        
        Returns:
            List of chat IDs
        """
        chat_files = self.storage_dir.glob("chat_*.json")
        return [f.stem for f in chat_files]
    
    def delete_chat(self, chat_id: str) -> None:
        """
        Delete a chat from memory and disk.
        
        Args:
            chat_id: Chat identifier
        """
        # Remove from memory
        if chat_id in self.chats:
            del self.chats[chat_id]
        
        # Remove from disk
        filepath = self.storage_dir / f"{chat_id}.json"
        if filepath.exists():
            filepath.unlink()
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ChatManager(chats={len(self.chats)}, storage={self.storage_dir})"

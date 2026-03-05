"""
Storage Module
Defines abstract storage backends for persisting chat and memory data.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class BaseStorageBackend(ABC):
    """Abstract base class for all memory/chat storage backends."""
    
    @abstractmethod
    def save(self, session_id: str, data: Dict[str, Any]) -> str:
        """Save session data and return the storage identifier/path."""
        pass

    @abstractmethod
    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Load session data by ID. Return None if not found."""
        pass

    @abstractmethod
    def delete(self, session_id: str) -> bool:
        """Delete a session. Return True if successful."""
        pass

    @abstractmethod
    def list_sessions(self) -> List[str]:
        """List all available session IDs."""
        pass


class JSONFileStorage(BaseStorageBackend):
    """Saves chat sessions as local JSON files."""
    
    def __init__(self, storage_dir: str = "./agent_chats"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
    def save(self, session_id: str, data: Dict[str, Any]) -> str:
        filepath = self.storage_dir / f"{session_id}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return str(filepath)
        
    def load(self, session_id: str) -> Optional[Dict[str, Any]]:
        filepath = self.storage_dir / f"{session_id}.json"
        if not filepath.exists():
            return None
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
            
    def delete(self, session_id: str) -> bool:
        filepath = self.storage_dir / f"{session_id}.json"
        if filepath.exists():
            filepath.unlink()
            return True
        return False
        
    def list_sessions(self) -> List[str]:
        return [f.stem for f in self.storage_dir.glob("*.json")]
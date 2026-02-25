"""
Configuration Module
Centralized configuration for the agent system.
"""

import os
from typing import Dict, Any
from pathlib import Path


class Config:
    """
    Global configuration settings for the agent system.
    """
    
    # Project paths
    BASE_DIR = Path(__file__).parent.resolve()
    DATA_DIR = BASE_DIR / "data"
    LOGS_DIR = BASE_DIR / "logs"
    EXPORTS_DIR = BASE_DIR / "exports"
    
    # Agent defaults
    DEFAULT_AGENT_NAME = "FinanceAgent"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_ITERATIONS = 10
    DEFAULT_TIMEOUT = 300
    
    # Tool settings
    MAX_TOOLS_PER_AGENT = 100
    TOOL_CACHE_ENABLED = True
    
    # LLM Model settings
    # Ollama
    OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    OLLAMA_DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "llama2")
    
    # vLLM
    VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:8000")
    VLLM_DEFAULT_MODEL = os.getenv("VLLM_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
    
    # Remote Endpoint (OpenAI, Azure, etc.)
    REMOTE_BASE_URL = os.getenv("REMOTE_BASE_URL", "https://api.openai.com")
    REMOTE_API_KEY = os.getenv("REMOTE_API_KEY", "")
    REMOTE_DEFAULT_MODEL = os.getenv("REMOTE_MODEL", "gpt-4")
    
    # Model generation settings
    MODEL_TEMPERATURE = 0.7
    MODEL_MAX_TOKENS = 2048
    MODEL_TIMEOUT = 300
    
    # Logging settings
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    # File settings
    EXPORT_FORMAT = "json"
    ENCODING = "utf-8"
    
    # API settings (if needed)
    API_TIMEOUT = 30
    MAX_RETRIES = 3
    
    @classmethod
    def create_directories(cls) -> None:
        """Create necessary directories if they don't exist."""
        cls.DATA_DIR.mkdir(exist_ok=True)
        cls.LOGS_DIR.mkdir(exist_ok=True)
        cls.EXPORTS_DIR.mkdir(exist_ok=True)
    
    @classmethod
    def get_config_dict(cls) -> Dict[str, Any]:
        """
        Get configuration as a dictionary.
        
        Returns:
            Dictionary of configuration settings
        """
        return {
            "base_dir": str(cls.BASE_DIR),
            "data_dir": str(cls.DATA_DIR),
            "logs_dir": str(cls.LOGS_DIR),
            "exports_dir": str(cls.EXPORTS_DIR),
            "default_agent_name": cls.DEFAULT_AGENT_NAME,
            "default_temperature": cls.DEFAULT_TEMPERATURE,
            "default_max_iterations": cls.DEFAULT_MAX_ITERATIONS,
            "log_level": cls.LOG_LEVEL,
        }


# Create directories on import
Config.create_directories()

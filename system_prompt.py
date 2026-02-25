"""
System Prompt Module
Provides a structured way to define and manage system prompts for agents.
"""

from typing import Optional, Dict, Any
from datetime import datetime


class SystemPrompt:
    """
    A class to manage system prompts for AI agents.
    
    Attributes:
        prompt_name (str): Name/identifier for the system prompt
        prompt_text (str): The actual system prompt text
        variables (Dict[str, Any]): Variables to be used in prompt templating
        created_at (datetime): Timestamp when the prompt was created
        metadata (Dict[str, Any]): Additional metadata about the prompt
    """
    
    def __init__(
        self,
        prompt_name: str,
        prompt_text: str,
        variables: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize a SystemPrompt instance.
        
        Args:
            prompt_name: Identifier for the prompt
            prompt_text: The system prompt text (can include {variable} placeholders)
            variables: Dictionary of variables for prompt templating
            metadata: Additional metadata (version, author, etc.)
        """
        self.prompt_name = prompt_name
        self.prompt_text = prompt_text
        self.variables = variables or {}
        self.metadata = metadata or {}
        self.created_at = datetime.now()
    
    def get_prompt(self, **kwargs) -> str:
        """
        Get the formatted system prompt with variables substituted.
        
        Args:
            **kwargs: Additional variables to override or extend instance variables
            
        Returns:
            Formatted prompt string
        """
        # Merge instance variables with kwargs
        all_variables = {**self.variables, **kwargs}
        
        try:
            return self.prompt_text.format(**all_variables)
        except KeyError as e:
            raise ValueError(f"Missing required variable in prompt: {e}")
    
    def update_prompt(self, new_prompt_text: str) -> None:
        """
        Update the prompt text.
        
        Args:
            new_prompt_text: New system prompt text
        """
        self.prompt_text = new_prompt_text
    
    def add_variables(self, **kwargs) -> None:
        """
        Add or update variables for the prompt.
        
        Args:
            **kwargs: Key-value pairs to add/update in variables
        """
        self.variables.update(kwargs)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get information about this system prompt.
        
        Returns:
            Dictionary containing prompt metadata and info
        """
        return {
            "prompt_name": self.prompt_name,
            "created_at": self.created_at.isoformat(),
            "variables": list(self.variables.keys()),
            "metadata": self.metadata,
            "prompt_length": len(self.prompt_text)
        }
    
    def __repr__(self) -> str:
        return f"SystemPrompt(name='{self.prompt_name}', length={len(self.prompt_text)})"
    
    def __str__(self) -> str:
        return self.get_prompt()


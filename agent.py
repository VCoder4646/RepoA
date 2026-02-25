"""
Agent Module
Main agent class that integrates system prompts and tools processing.
"""

from typing import Optional, List, Dict, Any, Callable
from datetime import datetime
import uuid
import json
from pathlib import Path

from system_prompt import SystemPrompt
from tools_pro import ToolProcessor, Tool, ToolType

# Conditional import for Chat and Message (core dependencies)
try:
    from chat import Chat, ChatManager
    from message import Message, user_message, agent_message, tool_message
    CHAT_AVAILABLE = True
except ImportError:
    Chat = None
    ChatManager = None
    Message = None
    CHAT_AVAILABLE = False

# Conditional import for LLM client (optional dependency)
try:
    from llm_client import BaseLLMClient, LLMResponse, format_tools_for_api, parse_tool_call_arguments
    LLM_AVAILABLE = True
except ImportError:
    BaseLLMClient = None
    LLMResponse = None
    LLM_AVAILABLE = False

# Conditional import for Memory (optional dependency)
try:
    from memory import Memory, create_memory
    MEMORY_AVAILABLE = True
except ImportError:
    Memory = None
    MEMORY_AVAILABLE = False


class AgentConfig:
    """
    Configuration class for agent settings.
    
    Attributes:
        max_iterations: Maximum number of agent iterations
        temperature: LLM temperature setting
        verbose: Whether to print verbose output
        timeout: Timeout in seconds for agent operations
        additional_settings: Dictionary for any additional settings
    """
    
    def __init__(
        self,
        max_iterations: int = 10,
        temperature: float = 0.7,
        verbose: bool = False,
        timeout: int = 300,
        **additional_settings
    ):
        self.max_iterations = max_iterations
        self.temperature = temperature
        self.verbose = verbose
        self.timeout = timeout
        self.additional_settings = additional_settings
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "max_iterations": self.max_iterations,
            "temperature": self.temperature,
            "verbose": self.verbose,
            "timeout": self.timeout,
            **self.additional_settings
        }


class Agent:
    """
    Main Agent class that combines system prompts and tools.
    
    This class represents an AI agent that can:
    - Use a defined system prompt
    - Access and utilize various tools
    - Execute tasks with context
    - Track its state and history
    
    Attributes:
        agent_id: Unique identifier for the agent
        agent_name: Human-readable name for the agent
        system_prompt: SystemPrompt instance defining agent behavior
        tools_processor: ToolProcessor instance managing available tools
        config: AgentConfig instance with agent settings
        state: Current state of the agent
        history: List of interaction history
    """
    
    def __init__(
        self,
        agent_name: str,
        system_prompt: SystemPrompt,
        tools_processor: ToolProcessor,
        config: Optional[AgentConfig] = None,
        agent_id: Optional[str] = None,
        llm_client: Optional['BaseLLMClient'] = None,
        memory: Optional['Memory'] = None,
        use_memory_cache: bool = False
    ):
        """
        Initialize an Agent instance.
        
        Args:
            agent_name: Name of the agent
            system_prompt: SystemPrompt instance for the agent
            tools_processor: ToolProcessor instance with available tools
            config: Optional AgentConfig instance (uses defaults if not provided)
            agent_id: Optional custom agent ID (generates UUID if not provided)
            llm_client: Optional LLM client for agent execution
            memory: Optional Memory instance for advanced memory management with KV cache
            use_memory_cache: Whether to use Memory class for KV cache tracking
        """
        self.agent_id = agent_id or str(uuid.uuid4())
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.tools_processor = tools_processor
        self.config = config or AgentConfig()
        self.llm_client = llm_client
        self.memory = memory
        self.use_memory_cache = use_memory_cache and memory is not None
        
        # Chat management (primary conversation manager)
        self.chat: Optional[Chat] = None
        self.chat_manager: Optional[ChatManager] = None
        if CHAT_AVAILABLE:
            # Initialize chat with system prompt
            system_prompt_text = self.system_prompt.get_prompt(agent_name=self.agent_name)
            self.chat = Chat(
                system_prompt=system_prompt_text,
                chat_id=self.agent_id,
                metadata={"agent_name": self.agent_name}
            )
            self.chat_manager = ChatManager(storage_dir="./agent_chats")
            # Register chat with manager
            self.chat_manager.chats[self.chat.chat_id] = self.chat
        
        # Agent state
        self.state = {
            "status": "initialized",
            "created_at": datetime.now().isoformat(),
            "iterations": 0,
            "last_activity": None
        }
        
        # Interaction history
        self.history: List[Dict[str, Any]] = []
        
        # Message history for LLM conversations (legacy mode, fallback)
        self.messages: List[Dict[str, str]] = []
        
        # Statistics
        self.stats = {
            "total_interactions": 0,
            "tool_calls": 0,
            "successful_tool_calls": 0,
            "failed_tool_calls": 0,
            "llm_calls": 0,
            "total_tokens": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_cached_tokens": 0,
            "cache_cost_savings": 0.0
        }
        
        if self.config.verbose:
            print(f"Agent '{self.agent_name}' initialized with ID: {self.agent_id}")
            print(f"Available tools: {self.tools_processor.list_tools()}")
            if self.llm_client:
                print(f"LLM client configured: {self.llm_client.model_name}")
            if self.chat:
                print(f"Chat enabled with ID: {self.chat.chat_id}")
            if self.use_memory_cache:
                print(f"Memory cache enabled with session: {self.memory.session_id}")
    
    def get_system_prompt(self, **kwargs) -> str:
        """
        Get the formatted system prompt for the agent.
        
        Args:
            **kwargs: Additional variables to pass to the prompt
            
        Returns:
            Formatted system prompt string
        """
        return self.system_prompt.get_prompt(
            agent_name=self.agent_name,
            **kwargs
        )
    
    def get_available_tools(self, tool_type: Optional[ToolType] = None) -> List[str]:
        """
        Get list of available tools.
        
        Args:
            tool_type: Optional filter by tool type
            
        Returns:
            List of tool names
        """
        return self.tools_processor.list_tools(tool_type)
    
    def get_tool_info(self, tool_name: str) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific tool.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool information dictionary or None if not found
        """
        tool = self.tools_processor.get_tool(tool_name)
        if tool:
            return tool.to_dict()
        return None
    
    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the agent's tool processor.
        
        Args:
            tool: Tool instance to add
        """
        self.tools_processor.add_tool(tool)
        if self.config.verbose:
            print(f"Tool '{tool.name}' added to agent '{self.agent_name}'")
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the agent.
        
        Args:
            tool_name: Name of the tool to remove
            
        Returns:
            True if removed, False if not found
        """
        result = self.tools_processor.remove_tool(tool_name)
        if result and self.config.verbose:
            print(f"Tool '{tool_name}' removed from agent '{self.agent_name}'")
        return result
    
    def validate_tool_call(self, tool_name: str, arguments: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate a tool call before execution.
        
        Args:
            tool_name: Name of the tool to validate
            arguments: Arguments to pass to the tool
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        tool = self.tools_processor.get_tool(tool_name)
        if not tool:
            return False, f"Tool '{tool_name}' not found"
        
        return tool.validate_args(arguments)
    
    def execute_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a tool with given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments to pass to the tool
            
        Returns:
            Dictionary with execution result
        """
        # Validate tool call
        is_valid, error_msg = self.validate_tool_call(tool_name, arguments)
        
        if not is_valid:
            self.stats["failed_tool_calls"] += 1
            return {
                "success": False,
                "error": error_msg,
                "tool": tool_name
            }
        
        # Get tool
        tool = self.tools_processor.get_tool(tool_name)
        self.stats["tool_calls"] += 1
        
        # If tool has a function, execute it
        if tool.function:
            try:
                result = tool.function(**arguments)
                self.stats["successful_tool_calls"] += 1
                return {
                    "success": True,
                    "result": result,
                    "tool": tool_name
                }
            except Exception as e:
                self.stats["failed_tool_calls"] += 1
                return {
                    "success": False,
                    "error": str(e),
                    "tool": tool_name
                }
        else:
            # Tool doesn't have executable function
            self.stats["successful_tool_calls"] += 1
            return {
                "success": True,
                "message": f"Tool '{tool_name}' validated successfully",
                "tool": tool_name,
                "arguments": arguments
            }
    
    def add_to_history(self, interaction: Dict[str, Any]) -> None:
        """
        Add an interaction to the agent's history.
        
        Args:
            interaction: Dictionary containing interaction details
        """
        interaction["timestamp"] = datetime.now().isoformat()
        self.history.append(interaction)
        self.state["last_activity"] = interaction["timestamp"]
        self.stats["total_interactions"] += 1
    
    def get_history(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get agent interaction history.
        
        Args:
            limit: Optional limit on number of history items to return
            
        Returns:
            List of interaction dictionaries
        """
        if limit:
            return self.history[-limit:]
        return self.history
    
    def clear_history(self) -> None:
        """Clear the agent's interaction history."""
        self.history.clear()
        if self.config.verbose:
            print(f"History cleared for agent '{self.agent_name}'")
    
    def update_state(self, **kwargs) -> None:
        """
        Update the agent's state.
        
        Args:
            **kwargs: Key-value pairs to update in state
        """
        self.state.update(kwargs)
        self.state["last_activity"] = datetime.now().isoformat()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the agent.
        
        Returns:
            Dictionary with agent information
        """
        info = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "system_prompt": self.system_prompt.get_info(),
            "tools": self.tools_processor.get_tool_info(),
            "config": self.config.to_dict(),
            "state": self.state,
            "stats": self.stats,
            "history_length": len(self.history),
            "memory_enabled": self.use_memory_cache
        }
        
        # Add memory cache info if available
        if self.use_memory_cache:
            info["memory_cache"] = self.get_cache_info()
        
        return info
    
    def export_agent(self, filepath: str) -> None:
        """
        Export agent configuration to a JSON file.
        
        Args:
            filepath: Path to save the agent configuration
        """
        agent_data = {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "system_prompt": {
                "name": self.system_prompt.prompt_name,
                "text": self.system_prompt.prompt_text,
                "variables": self.system_prompt.variables
            },
            "tools": self.tools_processor.format_for_llm(),
            "config": self.config.to_dict(),
            "state": self.state,
            "stats": self.stats
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(agent_data, f, indent=2)
        
        if self.config.verbose:
            print(f"Agent configuration exported to: {filepath}")
    
    def reset(self) -> None:
        """Reset the agent to initial state."""
        self.state = {
            "status": "initialized",
            "created_at": self.state["created_at"],
            "iterations": 0,
            "last_activity": None
        }
        self.history.clear()
        self.messages.clear()
        self.stats = {
            "total_interactions": 0,
            "tool_calls": 0,
            "successful_tool_calls": 0,
            "failed_tool_calls": 0,
            "llm_calls": 0,
            "total_tokens": 0,
            "cache_hits": 0,
            "cache_misses": 0,
            "total_cached_tokens": 0,
            "cache_cost_savings": 0.0
        }
        
        # Reset memory if using cache
        if self.use_memory_cache and self.memory:
            self.memory.clear(keep_system=True)
        
        # Reset chat if using it
        if self.chat:
            self.chat.clear_messages(keep_system=True)
        
        if self.config.verbose:
            print(f"Agent '{self.agent_name}' has been reset")
    
    # ========== LLM Integration Methods ==========
    
    def set_llm_client(self, llm_client: 'BaseLLMClient') -> None:
        """
        Set or update the LLM client for the agent.
        
        Args:
            llm_client: LLM client instance
        """
        if not LLM_AVAILABLE:
            raise RuntimeError("LLM client not available. Install llm_client module.")
        self.llm_client = llm_client
        if self.config.verbose:
            print(f"LLM client set: {llm_client.model_name}")
    
    def set_memory(self, memory: 'Memory', enable_cache: bool = True) -> None:
        """
        Set or update the Memory instance for the agent.
        
        Args:
            memory: Memory instance
            enable_cache: Whether to enable KV cache tracking
        """
        if not MEMORY_AVAILABLE:
            raise RuntimeError("Memory module not available.")
        self.memory = memory
        self.use_memory_cache = enable_cache
        if self.config.verbose:
            print(f"Memory set: session={memory.session_id}, cache_enabled={enable_cache}")
    
    def get_cache_info(self) -> Dict[str, Any]:
        """
        Get KV cache information from memory or agent stats.
        
        Returns:
            Dictionary with cache statistics
        """
        if self.use_memory_cache:
            return self.memory.get_llm_cache_info()
        else:
            # Return basic cache stats from agent
            total_requests = self.stats["cache_hits"] + self.stats["cache_misses"]
            return {
                "cache_hits": self.stats["cache_hits"],
                "cache_misses": self.stats["cache_misses"],
                "cache_hit_rate": self.stats["cache_hits"] / total_requests if total_requests > 0 else 0.0,
                "total_cached_tokens": self.stats["total_cached_tokens"],
                "cache_cost_savings": self.stats["cache_cost_savings"]
            }
    
    def _initialize_conversation(self) -> None:
        """Initialize conversation with system prompt."""
        if self.use_memory_cache:
            # Memory handles system prompt internally
            return
        
        if self.chat:
            # Chat handles system prompt internally
            return
        
        # Legacy mode: use messages list
        if not self.messages:
            system_prompt = self.get_system_prompt()
            self.messages = [
                {"role": "system", "content": system_prompt}
            ]
    
    def _format_tool_result(self, tool_result: Dict[str, Any]) -> str:
        """
        Format tool execution result for LLM.
        
        Args:
            tool_result: Result from execute_tool
            
        Returns:
            Formatted string
        """
        if tool_result.get("success"):
            result = tool_result.get("result", tool_result.get("message", "Success"))
            return json.dumps(result, indent=2) if isinstance(result, (dict, list)) else str(result)
        else:
            return f"Error: {tool_result.get('error', 'Unknown error')}"
    
    def run(
        self,
        user_message: str,
        max_iterations: Optional[int] = None,
        reset_conversation: bool = False
    ) -> Dict[str, Any]:
        """
        Run the agent with an LLM to process user message and execute tools.
        
        Args:
            user_message: User's input message
            max_iterations: Maximum iterations (overrides config if provided)
            reset_conversation: Whether to reset conversation history
            
        Returns:
            Dictionary with final response and execution details
            
        Raises:
            RuntimeError: If no LLM client is configured
        """
        if not self.llm_client:
            raise RuntimeError("No LLM client configured. Set one using set_llm_client() or pass in constructor.")
        
        if not LLM_AVAILABLE:
            raise RuntimeError("LLM client module not available.")
        
        # Reset conversation if requested
        if reset_conversation:
            if self.use_memory_cache:
                self.memory.clear(keep_system=True)
            elif self.chat:
                self.chat.clear_messages(keep_system=True)
            else:
                self.messages.clear()
        
        # Initialize conversation with system prompt
        self._initialize_conversation()
        
        # Add user message
        if self.use_memory_cache:
            self.memory.add_user_message(user_message)
        elif self.chat:
            self.chat.add_user_message(user_message)
        else:
            self.messages.append({"role": "user", "content": user_message})
        
        # Add to history
        self.add_to_history({
            "type": "user_message",
            "content": user_message
        })
        
        max_iter = max_iterations or self.config.max_iterations
        iterations = 0
        final_response = ""
        tool_calls_made = []
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Agent '{self.agent_name}' started")
            print(f"User: {user_message}")
            print(f"{'='*60}\n")
        
        self.update_state(status="running")
        
        while iterations < max_iter:
            iterations += 1
            self.state["iterations"] = iterations
            
            if self.config.verbose:
                print(f"[Iteration {iterations}/{max_iter}]")
            
            # Get tools in API format
            tools = format_tools_for_api(self.tools_processor.format_for_llm())
            
            # Get messages in LLM format
            if self.use_memory_cache:
                messages = self.memory.get_messages()
            elif self.chat:
                messages = self.chat.get_messages(include_system=True)
            else:
                messages = self.messages
            
            # Call LLM
            try:
                response: LLMResponse = self.llm_client.generate(
                    messages=messages,
                    tools=tools if tools else None
                )
                
                self.stats["llm_calls"] += 1
                self.stats["total_tokens"] += response.usage.get("total_tokens", 0)
                
                # Track cache hits from LLM response
                if response.has_cache_hit():
                    self.stats["cache_hits"] += 1
                    cache_info = response.get_cache_info()
                    self.stats["total_cached_tokens"] += cache_info.get("cached_tokens", 0)
                else:
                    self.stats["cache_misses"] += 1
                
            except Exception as e:
                error_msg = f"LLM error: {str(e)}"
                if self.config.verbose:
                    print(f"❌ {error_msg}")
                self.update_state(status="error")
                return {
                    "success": False,
                    "error": error_msg,
                    "iterations": iterations,
                    "tool_calls": tool_calls_made
                }
            
            # Check if model wants to use tools
            if response.has_tool_calls():
                if self.config.verbose:
                    print(f"🔧 Model requested {len(response.tool_calls)} tool call(s)")
                
                # Add assistant message with tool calls
                if self.use_memory_cache:
                    # Memory class handles tool calls differently, add as text for now
                    self.memory.add_agent_message(
                        content=response.content or "[Tool calls requested]",
                        llm_response=response.raw_response
                    )
                elif self.chat:
                    self.chat.add_agent_message(
                        content=response.content or "",
                        tool_calls=response.tool_calls
                    )
                else:
                    self.messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": response.tool_calls
                    })
                
                # Execute each tool call
                for tool_call in response.tool_calls:
                    try:
                        tool_name = tool_call.get("function", {}).get("name")
                        tool_args_str = tool_call.get("function", {}).get("arguments", "{}")
                        
                        if not tool_name:
                            if self.config.verbose:
                                print(f"    ⚠️ Tool call missing name, skipping")
                            continue
                        
                        if self.config.verbose:
                            print(f"  • Calling tool: {tool_name}")
                        
                        # Parse arguments
                        try:
                            tool_args = parse_tool_call_arguments(tool_args_str)
                        except Exception as e:
                            tool_args = {}
                            if self.config.verbose:
                                print(f"    ⚠️ Error parsing arguments: {e}")
                        
                        # Execute tool
                        tool_result = self.execute_tool(tool_name, tool_args)
                        
                        if self.config.verbose:
                            if tool_result.get("success"):
                                print(f"    ✓ Tool executed successfully")
                            else:
                                print(f"    ✗ Tool execution failed: {tool_result.get('error')}")
                        
                        # Add tool result to messages
                        tool_result_content = self._format_tool_result(tool_result)
                        
                        if self.use_memory_cache:
                            self.memory.add_tool_message(
                                content=tool_result_content,
                                tool_call_id=tool_call.get("id", "")
                            )
                        elif self.chat:
                            self.chat.add_tool_message(
                                content=tool_result_content,
                                tool_call_id=tool_call.get("id", "")
                            )
                        else:
                            tool_message = {
                                "role": "tool",
                                "tool_call_id": tool_call.get("id", ""),
                                "name": tool_name,
                                "content": tool_result_content
                            }
                            self.messages.append(tool_message)
                        
                        # Track tool call
                        tool_calls_made.append({
                            "tool": tool_name,
                            "arguments": tool_args,
                            "result": tool_result
                        })
                    
                    except Exception as e:
                        if self.config.verbose:
                            print(f"    ✗ Error processing tool call: {e}")
                
                # Continue loop to get next response from model
                continue
            
            else:
                # Model provided final response
                final_response = response.content
                
                # Add assistant message
                if self.use_memory_cache:
                    self.memory.add_agent_message(
                        content=final_response,
                        llm_response=response.raw_response
                    )
                elif self.chat:
                    self.chat.add_agent_message(content=final_response)
                else:
                    self.messages.append({
                        "role": "assistant",
                        "content": final_response
                    })
                
                if self.config.verbose:
                    print(f"💬 Agent: {final_response}")
                
                # Add to history
                self.add_to_history({
                    "type": "agent_response",
                    "content": final_response,
                    "tool_calls": tool_calls_made,
                    "iterations": iterations
                })
                
                break
        
        self.update_state(status="completed")
        
        if iterations >= max_iter and not final_response:
            final_response = "Maximum iterations reached without final response."
        
        if self.config.verbose:
            print(f"\n{'='*60}")
            print(f"Agent completed in {iterations} iteration(s)")
            print(f"{'='*60}\n")
        
        # Get cache info for result
        cache_info = self.get_cache_info() if self.use_memory_cache else None
        
        result = {
            "success": True,
            "response": final_response,
            "iterations": iterations,
            "tool_calls": tool_calls_made,
            "tokens_used": self.stats["total_tokens"]
        }
        
        # Add cache info if available
        if cache_info:
            result["cache_info"] = cache_info
        
        return result
    
    def send_message(self, message: str) -> str:
        """
        Simple chat interface that returns just the response text.
        Renamed from 'chat' to avoid conflict with self.chat attribute.
        
        Args:
            message: User message
            
        Returns:
            Agent's text response
        """
        result = self.run(message, reset_conversation=False)
        return result.get("response", "")
    
    def clear_conversation(self) -> None:
        """Clear the conversation history (messages)."""
        if self.use_memory_cache and self.memory:
            self.memory.clear(keep_system=True)
        elif self.chat:
            self.chat.clear_messages(keep_system=True)
        else:
            self.messages.clear()
        if self.config.verbose:
            print("Conversation history cleared")
    
    def save_memory(self, force: bool = True) -> Optional[str]:
        """
        Save memory session to disk (if using memory cache).
        
        Args:
            force: Force save even if auto-save disabled
            
        Returns:
            Path to saved file or None
        """
        if self.use_memory_cache:
            path = self.memory.save(force=force)
            if self.config.verbose:
                print(f"Memory saved to: {path}")
            return path
        return None
    
    def save_chat(self, storage_dir: Optional[str] = None) -> Optional[str]:
        """
        Save chat session to disk (if using chat).
        
        Args:
            storage_dir: Directory to save chat (uses default if not provided)
            
        Returns:
            Path to saved file or None
        """
        if self.chat and self.chat_manager:
            if storage_dir:
                self.chat_manager.storage_dir = Path(storage_dir)
            # Ensure chat is registered with manager
            self.chat_manager.chats[self.chat.chat_id] = self.chat
            # Save using manager
            self.chat_manager.save_chat(self.chat.chat_id)
            path = str(self.chat_manager.storage_dir / f"{self.chat.chat_id}.json")
            if self.config.verbose:
                print(f"Chat saved to: {path}")
            return path
        return None
    
    def load_chat(self, chat_id: str, storage_dir: Optional[str] = None) -> bool:
        """
        Load chat session from disk.
        
        Args:
            chat_id: Chat ID to load
            storage_dir: Directory to load from (uses default if not provided)
            
        Returns:
            True if loaded successfully
        """
        if not CHAT_AVAILABLE:
            return False
        
        if not self.chat_manager:
            self.chat_manager = ChatManager(storage_dir=storage_dir or "./agent_chats")
        elif storage_dir:
            self.chat_manager.storage_dir = Path(storage_dir)
        
        try:
            loaded_chat = self.chat_manager.load_chat(chat_id)
            self.chat = loaded_chat
            # Update agent_id to match loaded chat
            self.agent_id = chat_id
            if self.config.verbose:
                print(f"Chat loaded: {chat_id} ({self.chat.get_message_count()} messages)")
            return True
        except Exception as e:
            if self.config.verbose:
                print(f"Failed to load chat: {e}")
            return False
    
    def get_memory_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get memory statistics (if using memory cache).
        
        Returns:
            Memory stats dictionary or None
        """
        if self.use_memory_cache:
            return self.memory.get_stats()
        return None
    
    def get_chat_stats(self) -> Optional[Dict[str, Any]]:
        """
        Get chat statistics (if using chat).
        
        Returns:
            Chat stats dictionary or None
        """
        if self.chat:
            return self.chat.get_stats()
        return None
    
    def get_message_count(self) -> int:
        """
        Get total message count.
        
        Returns:
            Number of messages in conversation
        """
        if self.use_memory_cache and self.memory:
            return len(self.memory.chat.messages)
        elif self.chat:
            return self.chat.get_message_count()
        else:
            return len(self.messages)
    
    def get_token_count(self) -> int:
        """
        Get estimated token count.
        
        Returns:
            Total tokens in conversation
        """
        if self.use_memory_cache and self.memory:
            return self.memory.chat.get_token_count()
        elif self.chat:
            return self.chat.get_token_count()
        else:
            # Estimate from messages list
            total = 0
            for msg in self.messages:
                total += len(msg.get("content", "")) // 4
            return total
    
    def __repr__(self) -> str:
        return f"Agent(name='{self.agent_name}', id='{self.agent_id}', tools={len(self.tools_processor)})"
    
    def __str__(self) -> str:
        return f"Agent: {self.agent_name}\nTools: {len(self.tools_processor)}\nStatus: {self.state['status']}"


class AgentBuilder:
    """
    Builder class for creating agents with a fluent interface.
    """
    
    def __init__(self):
        """Initialize the agent builder."""
        self._name: Optional[str] = None
        self._system_prompt: Optional[SystemPrompt] = None
        self._tools_processor: Optional[ToolProcessor] = None
        self._config: Optional[AgentConfig] = None
        self._agent_id: Optional[str] = None
        self._llm_client: Optional['BaseLLMClient'] = None
        self._memory: Optional['Memory'] = None
        self._use_memory_cache: bool = False
    
    def with_name(self, name: str) -> 'AgentBuilder':
        """Set the agent name."""
        self._name = name
        return self
    
    def with_default_prompt(self, system_prompt: SystemPrompt or str) -> 'AgentBuilder':
        """Set the system prompt."""
        if isinstance(system_prompt, str):
            system_prompt = SystemPrompt("custom", system_prompt)
        self._system_prompt = system_prompt
        return self
    
    
    def with_tools_processor(self, tools_processor: ToolProcessor) -> 'AgentBuilder':
        """Set the tools processor."""
        self._tools_processor = tools_processor
        return self
    
    def with_config(self, config: AgentConfig) -> 'AgentBuilder':
        """Set the agent configuration."""
        self._config = config
        return self
    
    def with_id(self, agent_id: str) -> 'AgentBuilder':
        """Set a custom agent ID."""
        self._agent_id = agent_id
        return self
    
    def with_llm_client(self, llm_client: 'BaseLLMClient') -> 'AgentBuilder':
        """Set the LLM client."""
        self._llm_client = llm_client
        return self
    
    def with_memory(self, memory: 'Memory', enable_cache: bool = True) -> 'AgentBuilder':
        """Set the Memory instance with optional cache."""
        self._memory = memory
        self._use_memory_cache = enable_cache
        return self
    
    def build(self) -> Agent:
        """
        Build and return the Agent instance.
        
        Returns:
            Configured Agent instance
            
        Raises:
            ValueError: If required fields are missing
        """
        if not self._name:
            raise ValueError("Agent name is required")
        if not self._system_prompt:
            raise ValueError("System prompt is required")
        if not self._tools_processor:
            self._tools_processor = ToolProcessor()
        
        return Agent(
            agent_name=self._name,
            system_prompt=self._system_prompt,
            tools_processor=self._tools_processor,
            config=self._config,
            agent_id=self._agent_id,
            llm_client=self._llm_client,
            memory=self._memory,
            use_memory_cache=self._use_memory_cache
        )


def create_agent(
    name: str,
    system_prompt: Optional[SystemPrompt] = None,
    tools_processor: Optional[ToolProcessor] = None,
    config: Optional[AgentConfig] = None,
    agent_id: Optional[str] = None,
    llm_client: Optional['BaseLLMClient'] = None,
    memory: Optional['Memory'] = None,
    use_memory_cache: bool = False
) -> Agent:
    """
    Quick helper function to create an agent with defaults.
    
    Args:
        name: Agent name
        system_prompt: Optional system prompt to use (creates default if not provided)
        tools_processor: Optional tool processor to use (creates empty if not provided)
        config: Optional agent configuration (uses defaults if not provided)
        agent_id: Optional custom agent ID
        llm_client: Optional LLM client for agent execution
        memory: Optional Memory instance for advanced memory management
        use_memory_cache: Whether to use Memory class for KV cache tracking
        
    Returns:
        Configured Agent instance
    """
    # Create defaults if not provided
    if system_prompt is None:
        system_prompt = SystemPrompt("default", f"You are {name}, a helpful AI assistant.")
    
    if tools_processor is None:
        tools_processor = ToolProcessor()
    
    return Agent(
        agent_name=name,
        system_prompt=system_prompt,
        tools_processor=tools_processor,
        config=config,
        agent_id=agent_id,
        llm_client=llm_client,
        memory=memory,
        use_memory_cache=use_memory_cache
    )

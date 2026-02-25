"""
Finance Agent System
A structured agent system with system prompts, MCP tools integration,
memory management, and comprehensive logging.
"""

__version__ = "2.1.0"
__author__ = "Finance Agent Team"

# Import main classes for easy access
from agent import (
    Agent,
    AgentBuilder,
    AgentConfig,
    create_agent
)

from system_prompt import (
    SystemPrompt,
    SystemPromptLibrary,
    create_default_prompt
)

from tools_pro import (
    Tool,
    ToolParameter,
    ToolProcessor,
    ToolType,
    create_custom_tool
)

from config import Config
from utils import setup_logging, load_json, save_json

# Import memory classes (optional - will be None if not available)
try:
    from memory import (
        Memory,
        MemoryConfig,
        create_memory,
        load_memory,
        set_memory_logging_level
    )
    MEMORY_AVAILABLE = True
except ImportError:
    Memory = None
    MemoryConfig = None
    create_memory = None
    load_memory = None
    set_memory_logging_level = None
    MEMORY_AVAILABLE = False

# Import chat classes (optional)
try:
    from chat import Chat, ChatManager
    CHAT_AVAILABLE = True
except ImportError:
    Chat = None
    ChatManager = None
    CHAT_AVAILABLE = False

__all__ = [
    # Agent classes
    "Agent",
    "AgentBuilder",
    "AgentConfig",
    "create_agent",
    
    # System prompt classes
    "SystemPrompt",
    "SystemPromptLibrary",
    "create_default_prompt",
    
    # Tool classes
    "Tool",
    "ToolParameter",
    "ToolProcessor",
    "ToolType",
    "create_custom_tool",
    
    # Memory classes (if available)
    "Memory",
    "MemoryConfig",
    "create_memory",
    "load_memory",
    "set_memory_logging_level",
    
    # Chat classes (if available)
    "Chat",
    "ChatManager",
    
    # Config and utils
    "Config",
    "setup_logging",
    "load_json",
    "save_json",
    
    # Availability flags
    "MEMORY_AVAILABLE",
    "CHAT_AVAILABLE",
]


def get_version():
    """Get the current version of the Finance Agent System."""
    return __version__


def quick_start():
    """
    Quick start guide for new users.
    Prints basic usage information.
    """
    print(f"""
Finance Agent System v{__version__}
{'=' * 50}

Quick Start:

1. Create a basic agent:
   from agent import create_agent
   agent = create_agent("MyAgent", "finance_agent")

2. Use agent builder:
   from agent import AgentBuilder
   agent = AgentBuilder().with_name("Agent").with_default_prompt("general_assistant").build()

3. Add custom tools:
   from tools_pro import create_custom_tool
   tool = create_custom_tool("my_tool", "Description", [...])
   agent.add_tool(tool)

4. Load MCP tools:
   agent.tools_processor.load_mcp_tools(mcp_data)

5. Create agent with memory and logging:
   from agent import create_agent, AgentConfig
   from memory import create_memory
   import logging
   
   config = AgentConfig(
       log_level=logging.INFO,
       enable_logging=True,
       auto_save_chat=True,
       memory_log_level=logging.INFO
   )
   
   agent = create_agent("MyAgent", config=config)
   memory = create_memory("You are a helpful assistant.", max_tokens=1000)
   agent.set_memory(memory)

6. Use invoke method:
   result = agent.invoke("What's the weather?")

New Features:
✨ invoke() method - Flexible agent invocation (3 modes)
📝 Comprehensive logging - Agent and memory operations
💾 Auto-save chat - Session persistence
🧠 Memory module - Smart conversation management
📊 KV cache tracking - Cost optimization

Documentation:
- README.md - Overview and quick reference
- AGENT_LOGGING_DOCS.md - Logging configuration
- MEMORY_LOGGING_DOCS.md - Memory logging guide
- INVOKE_METHOD_DOCS.md - Invoke method details
- KV_CACHE_GUIDE.md - Cache optimization

Examples:
- example_agent_logging.py
- example_agent_memory_logging.py
- example_memory_logging.py
- example_invoke.py

{'=' * 50}
    """)


# Print version on import (optional - can be removed if not desired)
# print(f"Finance Agent System v{__version__} loaded")

"""
Finance Agent System
A structured agent system with system prompts and MCP tools integration.
"""

__version__ = "1.0.0"
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
    
    # Config and utils
    "Config",
    "setup_logging",
    "load_json",
    "save_json",
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

For more examples, see example_usage.py
For documentation, see README.md
{'=' * 50}
    """)


# Print version on import (optional - can be removed if not desired)
# print(f"Finance Agent System v{__version__} loaded")

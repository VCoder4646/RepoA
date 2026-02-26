# RepoA - Quick Import Reference

## Package Layout
```
repoa/
├── core/       # Agent, Memory, Message, LLMClient
├── tools/      # ToolProcessor, Tool, Utils
├── config/     # Config, SystemPrompt
└── cli/        # Chat, ChatManager
```

## Common Imports

### Option 1: Use convenience imports from main package
```python
from repoa import (
    Agent, AgentBuilder, AgentConfig,
    Memory, MemoryConfig,
    SystemPrompt, SystemPromptLibrary,
    Tool, ToolProcessor, ToolType,
    Config, Chat
)
```

### Option 2: Import from specific modules
```python
# Core functionality
from repoa.core import Agent, Memory, Message, LLMClient

# Tools
from repoa.tools import ToolProcessor, Tool, ToolType, setup_logging

# Configuration
from repoa.config import Config, SystemPrompt

# CLI
from repoa.cli import Chat, ChatManager
```

## Quick Examples

### Create Agent
```python
from repoa import create_agent
agent = create_agent("MyAgent", "repoa")
```

### Create Agent with Memory
```python
from repoa import create_agent, create_memory

agent = create_agent("MyAgent")
memory = create_memory("You are helpful.", max_tokens=1000)
agent.set_memory(memory)
```

### Load MCP Tools
```python
from repoa.tools import ToolProcessor

processor = ToolProcessor()
processor.load_mcp_tools(mcp_data)
```

### Custom System Prompt
```python
from repoa.config import SystemPrompt

prompt = SystemPrompt("agent_name", "Custom instructions here")
```

## Installation

### Development Mode
```bash
pip install -e .
```

### With Optional Dependencies
```bash
# All extras
pip install -e ".[all]"

# Specific extras
pip install -e ".[dev,llm,data]"
```

## Module Purpose

| Module | Purpose |
|--------|---------|
| `core` | Core agent functionality (Agent, Memory, Message, LLMClient) |
| `tools` | Tool processing and utility functions |
| `config` | Configuration and system prompt management |
| `cli` | Command-line interface and chat management |

## File Locations

| File | Old Location | New Location |
|------|-------------|--------------|
| agent.py | root | repoa/core/ |
| memory.py | root | repoa/core/ |
| message.py | root | repoa/core/ |
| llm_client.py | root | repoa/core/ |
| tools_pro.py | root | repoa/tools/ |
| utils.py | root | repoa/tools/ |
| config.py | root | repoa/config/ |
| system_prompt.py | root | repoa/config/ |
| chat.py | root | repoa/cli/ |
## Key Benefits
✅ Professional package structure
✅ Clear module organization
✅ Easy to install and distribute
✅ Better IDE support and autocomplete
✅ Scalable architecture
✅ Proper namespace management

## Need Help?
- See `PROJECT_STRUCTURE.md` for detailed documentation
- See `README.md` for feature overview
- Check `examples/sample.py` for usage examples

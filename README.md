
A comprehensive, modular agent system with **Memory management**, **KV cache optimization**, and **cost-efficient LLM integration**.

## 🚀 What's New in v1.0

- **Memory System**: Smart conversation management with automatic persistence
- **KV Cache Tracking**: Monitor and optimize LLM cache usage for cost savings
- **Agent-Memory Integration**: Seamless agent operation with conversation history
- **Cost Optimization**: Track cache hits and estimate cost savings
- **Session Persistence**: Save and resume conversations across sessions
- **Enhanced Statistics**: Comprehensive tracking of messages, tokens, and cache performance

## ✨ Key Features

### Core Features
- **Modular Architecture**: Clean separation of concerns (agent, memory, tools, chat)
- **System Prompt Management**: Flexible templating with variable substitution
- **MCP Tools Integration**: Parse and process Model Context Protocol tools
- **Tool Management**: Comprehensive validation and execution
- **Agent Builder Pattern**: Fluent interface for agent creation
- **History Tracking**: Complete interaction logging

### Memory & Caching
- **Smart Memory**: Automatic overflow handling and message archiving
- **KV Cache**: Track LLM cache usage (OpenAI, Anthropic, vLLM formats)
- **Cost Savings**: Calculate savings from cache hits
- **Session Management**: Unique IDs and metadata support
- **Multi-turn Optimization**: Efficient context reuse

## 📁 Project Structure

```
├── agent.py                          # Agent system with Memory integration
├── memory.py                         # Memory management with KV cache
├── chat.py                           # Chat session management
├── message.py                        # Message types and formatting
├── system_prompt.py                  # System prompt management
├── tools_pro.py                      # Tool processing and validation
├── llm_client.py                     # LLM clients with cache tracking
├── config.py                         # Configuration settings
├── utils.py                          # Utility functions
```

## 🔧 Installation

1. Clone or download this repository
2. Ensure Python 3.8+ is installed
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## 🎯 Quick Start

### 1. Basic Agent

```python
from agent import create_agent

# Quick agent creation
agent = create_agent(
    name="MyAgent",
    prompt_type="general_assistant"
)

print(agent.get_agent_info())
```

### 2. Agent with Memory

```python
from agent import Agent
from system_prompt import SystemPrompt
from tools_pro import ToolProcessor
from memory import Memory, MemoryConfig

# Create memory
memory = Memory(
    system_prompt="You are a helpful AI assistant.",
    config=MemoryConfig(
        max_tokens=4096,
        auto_save=True
    )
)

# Create agent with memory
agent = Agent(
    name="SmartAgent",
    system_prompt=SystemPrompt("You are a helpful AI assistant."),
    tools=ToolProcessor(),
    memory=memory,
    use_memory_cache=True  # Enable KV cache optimization
)

# Add conversation
memory.add_user_message("Hello!")
memory.add_agent_message("Hi there! How can I help?")

# Save session
agent.save_memory()
```

### 3. Agent with Tools

```python
from tools_pro import create_custom_tool

# Create custom tool
calc_tool = create_custom_tool(
    name="calculator",
    description="Perform calculations",
    parameters={
        "expression": {
            "type": "string",
            "description": "Math expression"
        }
    },
    required=["expression"],
    implementation=lambda expression: {"result": eval(expression)}
)

# Add to tools
tools = ToolProcessor()
tools.add_custom_tool(calc_tool)

# Create agent with tools
agent = Agent(
    name="MathAgent",
    system_prompt=SystemPrompt("You are a math assistant."),
    tools=tools
)

# Execute tool
result = agent.execute_tool("calculator", {"expression": "10 * 5"})
print(result['result']['result'])  # 50
```

### 4. KV Cache Optimization

```python
from memory import Memory, MemoryConfig

memory = Memory(
    system_prompt="You are an AI assistant.",
    config=MemoryConfig(max_tokens=4096)
)

# First turn - cache miss
memory.add_user_message("Hello")

llm_response_1 = {
    "usage": {
        "prompt_tokens": 50,
        "prompt_tokens_details": {"cached_tokens": 0}
    }
}

memory.add_agent_message(
    "Hi there!",
    llm_response=llm_response_1
)

# Second turn - cache hit
memory.add_user_message("Tell me about AI")

llm_response_2 = {
    "usage": {
        "prompt_tokens": 80,
        "prompt_tokens_details": {"cached_tokens": 60}  # 60 tokens cached!
    }
}

memory.add_agent_message(
    "AI is...",
    llm_response=llm_response_2
)

# Get cache statistics
cache_info = memory.get_llm_cache_info()
print(f"Cached tokens: {cache_info['total_cached_tokens']}")
print(f"Cost savings: ${cache_info['cost_savings']:.4f}")
```

### 5. Session Persistence

```python
# Create and save session
memory = Memory(
    system_prompt="You are a helpful assistant.",
    config=MemoryConfig(storage_dir="./sessions")
)

memory.add_user_message("Hello")
memory.add_agent_message("Hi!")

session_id = memory.session_id
memory.save()

# Later: Load session
from chat import ChatManager

manager = ChatManager(storage_dir="./sessions")
loaded_chat = manager.load_chat(session_id)

# Resume conversation
loaded_memory = Memory(
    system_prompt=loaded_chat.system_prompt,
    session_id=loaded_chat.chat_id
)

loaded_memory.add_user_message("Continue conversation...")
```

## 📚 Core Modules

### agent.py

Main agent implementation with Memory integration.

**Classes:**
- `Agent`: Primary agent class with memory support
- `AgentBuilder`: Builder pattern for agent creation
- `AgentConfig`: Configuration settings

**Key Methods:**
- `set_memory(memory)`: Set or update Memory instance
- `get_cache_info()`: Get KV cache statistics
- `save_memory()`: Save memory session
- `get_memory_stats()`: Get comprehensive memory statistics
- `execute_tool(name, args)`: Execute a tool

**Features:**
- Tool execution and validation
- History tracking
- Cache statistics (hits, misses, savings)
- Export/import configurations

### memory.py

Smart memory manager with KV cache tracking.

**Classes:**
- `Memory`: Main memory manager
- `MemoryConfig`: Configuration for memory behavior

**Key Methods:**
- `add_user_message(content)`: Add user message
- `add_agent_message(content, llm_response=...)`: Add agent message with cache data
- `add_tool_message(content, tool_call_id)`: Add tool response
- `get_messages()`: Get messages for LLM
- `get_llm_cache_info()`: Get LLM cache statistics
- `save()`: Save session to disk
- `get_stats()`: Get memory statistics

**Features:**
- Automatic persistence when limits exceeded
- KV cache tracking (OpenAI, Anthropic, vLLM formats)
- Message archiving
- Cost savings calculation
- Session management with unique IDs

### chat.py

Chat session management with persistence.

**Classes:**
- `Chat`: Chat session with messages
- `ChatManager`: Manage multiple chat sessions

**Key Methods:**
- `add_user_message(content)`: Add user message
- `add_agent_message(content, tool_calls)`: Add agent message
- `add_tool_message(content, tool_call_id)`: Add tool result
- `get_messages()`: Get all messages
- `save(path)`: Save chat to file
- `get_stats()`: Get chat statistics
- `export_text()`: Export as readable text

**Features:**
- Single system prompt storage (not repeated)
- Tool call tracking
- Token counting
- Metadata support
- Text export

### tools_pro.py

Tool processing and management.

**Classes:**
- `Tool`: Tool representation
- `ToolParameter`: Parameter definition
- `ToolProcessor`: Main tool management class
- `ToolType`: Enum for tool types

**Key Methods:**
- `add_custom_tool(tool)`: Add custom tool
- `load_mcp_tools(tools)`: Load MCP tools
- `execute_tool(name, args)`: Execute a tool
- `validate_tool_call(name, args)`: Validate arguments
- `get_tool_schemas()`: Get schemas for LLM

**Features:**
- Parse MCP tools
- Validate tool arguments
- Format tools for LLM consumption
- Tool grouping
- Import/export tools

### system_prompt.py

System prompt management with templating.

**Classes:**
- `SystemPrompt`: Prompt class with variable substitution
- `SystemPromptLibrary`: Manage multiple prompts

**Default Prompts:**
- `finance_agent`: Financial analysis specialist
- `general_assistant`: General purpose assistant
- `data_analyst`: Data analysis specialist

**Features:**
- Variable substitution: `{variable_name}`
- Prompt library management
- Template management
- Export/import prompts

### llm_client.py

LLM client abstraction with cache tracking.

**Classes:**
- `LLMResponse`: Response wrapper with cache info
- `OllamaClient`: Ollama LLM client
- `VLLMClient`: vLLM client
- `RemoteEndpointClient`: Generic remote endpoint

**Key Methods (LLMResponse):**
- `has_cache_hit()`: Check if cache was used
- `get_cache_info()`: Get detailed cache statistics

**Features:**
- Multiple provider support (Ollama, vLLM, OpenAI, Anthropic)
- Automatic cache data extraction
- Cost calculation helpers
- Streaming support

## 📊 Statistics & Monitoring

### Memory Statistics

```python
stats = memory.get_stats()

print(f"Messages added: {stats['total_messages_added']}")
print(f"Tokens processed: {stats['total_tokens_processed']}")
print(f"Saves triggered: {stats['saves_triggered']}")
print(f"Messages archived: {stats['messages_archived']}")
```

### LLM Cache Statistics

```python
cache_info = memory.get_llm_cache_info()

print(f"Total cached tokens: {cache_info['total_cached_tokens']}")
print(f"Cache hits: {cache_info['cache_hits']}")
print(f"Cache misses: {cache_info['cache_misses']}")
print(f"Cost savings: ${cache_info['cost_savings']:.4f}")
```

### Agent Statistics

```python
agent_stats = agent.stats

print(f"Total runs: {agent_stats['total_runs']}")
print(f"Cache hits: {agent_stats['cache_hits']}")
print(f"Cache misses: {agent_stats['cache_misses']}")
print(f"Total cached tokens: {agent_stats['total_cached_tokens']}")
print(f"Cache cost savings: ${agent_stats['cache_cost_savings']:.4f}")
```

### Chat Statistics

```python
chat_stats = memory.chat.get_stats()

print(f"Total messages: {chat_stats['total_messages']}")
print(f"Total tokens: {chat_stats['total_tokens']}")
print(f"User messages: {chat_stats['user_messages']}")
print(f"Agent messages: {chat_stats['agent_messages']}")
print(f"Tool calls: {chat_stats['tool_calls']}")
```

## 🎨 Usage Patterns

### Pattern 1: Simple Agent

```python
agent = create_agent("MyAgent", "general_assistant")
```

### Pattern 2: Agent with Memory

```python
memory = Memory(system_prompt="...", config=MemoryConfig(...))
agent = Agent(..., memory=memory, use_memory_cache=True)
```

### Pattern 3: Agent Builder

```python
agent = (AgentBuilder()
         .with_name("MyAgent")
         .with_system_prompt(prompt)
         .with_tools_processor(tools)
         .with_memory(memory)
         .with_config(config)
         .build())
```

### Pattern 4: Persistent Sessions

```python
# Create
memory = Memory(..., config=MemoryConfig(storage_dir="./sessions"))
# ... conversation ...
memory.save()

# Resume
manager = ChatManager(storage_dir="./sessions")
chat = manager.load_chat(session_id)
memory = Memory(system_prompt=chat.system_prompt, session_id=chat.chat_id)
```

## 💰 Cost Optimization Tips

1. **Enable KV Cache**: Use `use_memory_cache=True` when creating agents
2. **Track Cache Hits**: Monitor `get_llm_cache_info()` regularly
3. **Persistent Sessions**: Reuse sessions to maximize cache benefits
4. **Optimize Context**: Keep relevant messages in active memory
5. **Monitor Savings**: Check `cache_cost_savings` in statistics

### Expected Savings

With KV cache enabled:
- **Short conversations (1-3 turns)**: 10-30% cost reduction
- **Medium conversations (5-10 turns)**: 30-50% cost reduction
- **Long conversations (10+ turns)**: 50-70% cost reduction

## 📖 Examples

### Complete Examples Available

1. **sample.py**: Quick start samples (5 examples)
2. **example_memory_integration.py**: Memory system (8 examples)
3. **example_chat_integration.py**: Chat & Message (8 examples)
4. **example_agent_comprehensive.py**: Agent workflows (5 examples)
5. **example_kv_cache_usage.py**: Cache optimization (6 examples)

Run any example:
```bash
python sample.py
python test/example_memory_integration.py
python test/example_agent_comprehensive.py
```

## 🔬 Testing

Run test suites:
```bash
# KV cache integration tests
python test/test_kv_cache_integration.py

# Agent with cache tests
python test/test_agent_with_cache.py
```

All tests include:
- Memory integration
- Cache tracking
- Agent workflows
- Tool execution
- Session persistence

## 📚 Documentation

Comprehensive guides available:

- **KV_CACHE_GUIDE.md**: Complete KV cache optimization guide
- **AGENT_CACHE_GUIDE.md**: Agent-Memory integration guide
- **README.md**: This file (overview and quick reference)

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Agent                                │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │ System      │  │ Tools        │  │ Memory (optional)  │ │
│  │ Prompt      │  │ Processor    │  │                    │ │
│  └─────────────┘  └──────────────┘  └────────────────────┘ │
│                                              │               │
└──────────────────────────────────────────────┼───────────────┘
                                               │
                                               ▼
                        ┌────────────────────────────────────┐
                        │          Memory                    │
                        │  ┌──────────────────────────────┐  │
                        │  │ Chat (messages + persistence)│  │
                        │  ├──────────────────────────────┤  │
                        │  │ KV Cache Tracking            │  │
                        │  ├──────────────────────────────┤  │
                        │  │ Statistics & Monitoring      │  │
                        │  └──────────────────────────────┘  │
                        └────────────────────────────────────┘
                                       │
                                       ▼
                        ┌────────────────────────────────────┐
                        │       LLM Client                   │
                        │  • Returns cache data in response  │
                        │  • Supports multiple providers     │
                        └────────────────────────────────────┘
```

## 🔄 Migration from v1.0

### Breaking Changes

1. **AgentMemory removed**: Use `Memory` class instead
2. **Memory API changed**: New methods and configuration
3. **Agent constructor**: Added `memory` and `use_memory_cache` parameters

### Migration Guide

**Before (v1.0):**
```python
agent = Agent(name="MyAgent", system_prompt=prompt, tools=tools)
```

**After (v2.0):**
```python
memory = Memory(system_prompt=prompt.get_prompt())
agent = Agent(name="MyAgent", system_prompt=prompt, tools=tools, memory=memory)
```

## 🤝 Contributing

To extend this system:

1. Add new prompts in `system_prompt.py`
2. Create new tool types in `tools_pro.py`
3. Add utility functions in `utils.py`
4. Update examples for new features
5. Add tests for new functionality

## 📄 License

This project is provided as-is for educational and development purposes.

## 🙋 Support

For questions or issues:

1. Check the comprehensive examples in `test/` directory
2. Review the guides: `KV_CACHE_GUIDE.md` and `AGENT_CACHE_GUIDE.md`
3. Examine docstrings in each module
4. Run the test suites to understand expected behavior

## 🎯 Roadmap

Future enhancements:

- [ ] Streaming response support
- [ ] Advanced cache strategies
- [ ] Multi-agent conversations
- [ ] Vector database integration for long-term memory
- [ ] Advanced tool orchestration
- [ ] Real-time cost monitoring dashboard

---

**Version**: 1.0  
**Last Updated**: 2026-02-24  
**Requirements**: Python 3.8+


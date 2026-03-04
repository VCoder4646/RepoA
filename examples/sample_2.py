# Remote Endpoint Client Example
from repoa.core.agent import create_agent, AgentConfig
import logging
from repoa.config.system_prompt import SystemPrompt
from repoa.tools.tools_pro import Tool, ToolType, ToolProcessor, ToolParameter
from repoa.config.config import Config
from repoa.core.memory import Memory
# 1. Update the import to use RemoteEndpointClient
from repoa.core.llm_client import RemoteEndpointClient
from phoenix.otel import register
register(project_name="repoa-stock-advisor")

sp = SystemPrompt("repoa", "Create an agent that can analyze stock data and provide investment advice.")

tools_orchestrator = ToolProcessor()
tools_orchestrator.load_mcp_tools([
    {
        "name": "get_weather",
        "title": "Weather Information Provider",
        "description": "Get current weather information for a location",
        "inputSchema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City name or zip code"
                }
            },
            "required": ["location"]
        }
    }
])

tools_orchestrator.add_tool(
    Tool(
        name="stock_price_analyzer",
        description="Analyze stock data and provide investment advice",
        parameters=[
            ToolParameter(
                name="stock_symbol",
                type="string",
                description="Stock ticker symbol (e.g., AAPL, GOOGL)",
                required=True
            )
        ],
        tool_type=ToolType.CUSTOM,
        function=lambda stock_symbol: f"price of {stock_symbol}: $150 (mock data)"
    )
)

config = AgentConfig(
    auto_save_chat=True,
    chat_save_dir='./agent_chats', 
    log_level=logging.INFO,
    enable_logging=False,
    memory_log_level=logging.INFO,      
    enable_memory_logging=False           
)

# 2. Swap out OllamaClient for RemoteEndpointClient
llm = RemoteEndpointClient(
    model_name="llama-3.3-70b-versatile",                        
    base_url="https://api.groq.com/openai",            
    api_key=""               
)
chat_memory=Memory(system_prompt=str(sp),session_id="test_session_001")
agent = create_agent(
    name="TestAgent",
    system_prompt=sp,
    tools_processor=tools_orchestrator,
    llm_client=llm,
    config=config,
    memory=chat_memory
)

from repoa.core import agent as agent_mod
print("CHAT_AVAILABLE =", agent_mod.CHAT_AVAILABLE)
print("Chat class =", agent_mod.Chat)

print(agent.invoke("What is the AAPL Stock price right now?"))
print(agent.get_agent_info())

# Hugging Face Client Example

# from repoa.core.agent import create_agent, AgentConfig
# import logging
# from repoa.config.system_prompt import SystemPrompt
# from repoa.tools.tools_pro import Tool, ToolType, ToolProcessor, ToolParameter
# from repoa.config.config import Config

# # 1. Update the import to use HuggingFaceClient
# from repoa.core.llm_client import HuggingFaceClient
# from repoa.core.llm_client import SGLangClient
# from repoa.core.llm_client import LMStudioClient

# sp = SystemPrompt("repoa", "Create an agent that can analyze stock data and provide investment advice.")

# tools_orchestrator = ToolProcessor()
# tools_orchestrator.load_mcp_tools([
#     {
#         "name": "get_weather",
#         "title": "Weather Information Provider",
#         "description": "Get current weather information for a location",
#         "inputSchema": {
#             "type": "object",
#             "properties": {
#                 "location": {
#                     "type": "string",
#                     "description": "City name or zip code"
#                 }
#             },
#             "required": ["location"]
#         }
#     }
# ])

# tools_orchestrator.add_tool(
#     Tool(
#         name="stock_price_analyzer",
#         description="Analyze stock data and provide investment advice",
#         parameters=[
#             ToolParameter(
#                 name="stock_symbol",
#                 type="string",
#                 description="Stock ticker symbol (e.g., AAPL, GOOGL)",
#                 required=True
#             )
#         ],
#         tool_type=ToolType.CUSTOM,
#         function=lambda stock_symbol: f"price of {stock_symbol}: $150 (mock data)"
#     )
# )

# config = AgentConfig(
#     auto_save_chat=True,
#     log_level=logging.INFO,
#     enable_logging=False,
#     memory_log_level=logging.INFO,      
#     enable_memory_logging=False           
# )

# # 2. Swap out RemoteEndpointClient for HuggingFaceClient
# # llm = HuggingFaceClient(
# #     model_name="meta-llama/Meta-Llama-3-8B-Instruct",  
# #     hf_token=""                          
# # )

# # llm = SGLangClient(
# #     model_name="meta-llama/Meta-Llama-3-8B-Instruct", 
# #     base_url="http://localhost:30000"
# # )

# # llm = LMStudioClient(
# #     model_name="google/gemma-3-4b",  
# #     base_url="http://localhost:1234"
# # )

# agent = create_agent(
#     name="TestAgent",
#     system_prompt=sp,
#     tools_processor=tools_orchestrator,
#     llm_client=llm,
#     config=config
# )

# print(agent.invoke("What is the AAPL Stock price right now?"))
# print(agent.get_agent_info())
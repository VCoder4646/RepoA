from repoa.core.agent import create_agent, AgentConfig
import phoenix.otel as otel
from opentelemetry import trace
import logging
from repoa.config.system_prompt import SystemPrompt
from repoa.tools.tools_pro import Tool, ToolType, ToolProcessor, ToolParameter
from repoa.config.config import Config
from repoa.core.llm_client import OllamaClient

otel.register(project_name="repoa-stock-advisor")
sp=SystemPrompt("repoa","Create an agent that can analyze stock data and provide investment advice.")
tools_orchestrator=ToolProcessor()
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
        }}])
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
        log_level=logging.INFO,
        enable_logging=False,
        memory_log_level=logging.INFO,      # Memory logs at INFO level
        enable_memory_logging=False           # Enable memory logging
    )
llm=OllamaClient(model_name="qwen2.5:3b")
agent=create_agent(name="TestAgent",system_prompt=sp,tools_processor=tools_orchestrator,llm_client=llm,config=config)
print(agent.invoke("What is the AAPL Stock price right now?"))
print(agent.get_agent_info())

import os
import requests
import psycopg2
import logging
import phoenix.otel as otel
from repoa.core.agent import create_agent, AgentConfig
from repoa.config.system_prompt import SystemPrompt
from repoa.tools.tools_pro import Tool, ToolType, ToolProcessor, ToolParameter
from repoa.core.llm_client import OllamaClient # Assuming repoa has a Groq client, or use Ollama

# 1. Traceability Setup
otel.register(project_name="rag_trace")

# 2. Database & API Config
DB_CONFIG = {
    "host": "localhost",
    "port": 5433,
    "database": "postgres",
    "user": "postgres",
    "password": "root",
}
EMBEDDING_API_URL = "http://192.168.200.22:11450/v1"
TABLE_NAME = "rules"

# 3. Define the search logic as a function
def search_rules_logic(query: str) -> str:
    try:
        # Embedding Call
        payload = {"model": "bge-m3", "input": [query]}
        emb_res = requests.post(f"{EMBEDDING_API_URL}/embeddings", json=payload, timeout=10)
        query_embedding = emb_res.json()["data"][0]["embedding"]

        # PGVector Search
        conn = psycopg2.connect(**DB_CONFIG)
        cur = conn.cursor()
        cur.execute(
            f"SELECT text, doc_name, page_no FROM {TABLE_NAME} "
            f"ORDER BY embedding <=> %s::vector LIMIT 4",
            (query_embedding,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        if not rows: return "No relevant rules found."
        
        return "\n\n".join([f"[Source: {d}, Page: {p}]\nContent: {t}" for t, d, p in rows])
    except Exception as e:
        return f"Error: {str(e)}"

# 4. Framework Orchestration
tools_orchestrator = ToolProcessor()

# Register the Database Tool
tools_orchestrator.add_tool(
    Tool(
        name="search_database",
        description="Search the rules database for specific documents and regulations.",
        parameters=[
            ToolParameter(
                name="query",
                type="string",
                description="The search query or question to look up in the rules database",
                required=True
            )
        ],
        tool_type=ToolType.CUSTOM,
        function=search_rules_logic
    )
)

# 5. Agent Configuration
sp = SystemPrompt(
    "RuleBot", 
    """You are a professional Legal Research Assistant specialized in document retrieval and analysis. 

### GOAL
Answer user queries accurately using ONLY the information provided by the `search_database` tool.

### OPERATIONAL RULES
1. **Search First:** For every query, always call `search_database` first to retrieve relevant rules or clauses.
2. **Strict Grounding:** Base your answers entirely on the tool's output. If the database results do not contain the answer, state: "I cannot find specific rules regarding this in the current database."
3. **No Hallucinations:** Do not invent rule numbers, document names, or legal requirements.
4. **Iterative Search:** If the initial tool results are vague, you may perform a second search with a more specific or technical query.

### RESPONSE FORMAT
- **Executive Summary:** Start with a direct answer to the user's question.
- **Detailed Evidence:** Quote or paraphrase the relevant sections.
- **Citations:** You MUST cite the source and page number for every fact (e.g., [Source: Safety_Manual.pdf, Page: 12]).
- **Tone:** Maintain a formal, objective, and precise legal tone.

### CONSTRAINT
If a user asks for legal advice or "what they should do," provide the facts from the database and add a disclaimer: "This information is for reference only and does not constitute legal advice."
"""
)
config = AgentConfig(auto_save_chat=True, log_level=logging.INFO)

# Assuming ChatGroqClient exists in your repoa.core.llm_client or use Ollama
# Adjust the client class based on your actual repo files
llm = OllamaClient(model_name="qwen2.5:3b") 

agent = create_agent(
    name="DatabaseAgent",
    system_prompt=sp,
    tools_processor=tools_orchestrator,
    llm_client=llm,
    config=config
)

# 6. Execution
if __name__ == "__main__":
    response = agent.invoke("When can an entity offset assets and liabilities or income and expenses?")
    print(f"Agent Response: {response}")
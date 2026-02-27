# extractors.py
from typing import Any, Dict, List
from openinference.semconv.trace import SpanAttributes as OI
from .tracer import safe_json_dumps

def _ensure_messages_schema(messages: List[Dict]) -> List[Dict]:
    """
    Phoenix expects chat messages like:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    Tool calls (if any) should live under "tool_calls".
    """
    cleaned = []
    for m in messages or []:
        if not isinstance(m, dict):
            cleaned.append({"role": "unknown", "content": str(m)})
            continue
        role = m.get("role", "unknown")
        content = m.get("content", "")
        out = {"role": role, "content": content}
        if "tool_calls" in m:
            out["tool_calls"] = m["tool_calls"]
        cleaned.append(out)
    return cleaned

def extract_agent_attributes(span, agent, user_input: str):
    span.set_attribute(OI.INPUT_VALUE, user_input)
    span.set_attribute(OI.SESSION_ID, getattr(agent, "agent_id", "unknown"))
    span.set_attribute("repoa.agent.name", getattr(agent, "agent_name", "unknown"))
    span.set_attribute("repoa.agent.system_prompt", getattr(agent.system_prompt, "prompt_text", ""))

    cfg = agent.config.to_dict() if getattr(agent, "config", None) else {}
    span.set_attribute("repoa.agent.config", safe_json_dumps(cfg))

    tools_dict = agent.tools_processor.format_for_llm() if getattr(agent, "tools_processor", None) else {}
    span.set_attribute("repoa.agent.available_tools", safe_json_dumps(tools_dict))

    if getattr(agent, "use_memory_cache", False) and getattr(agent, "memory", None):
        span.set_attribute("repoa.memory.session_id", getattr(agent.memory, "session_id", ""))
        span.set_attribute("repoa.memory.cache_enabled", True)

def extract_llm_attributes(span, client, messages: List[Dict], response: Any):
    span.set_attribute(OI.LLM_MODEL_NAME, getattr(client, "model_name", "unknown"))

    normalized = _ensure_messages_schema(messages)
    span.set_attribute(OI.LLM_INPUT_MESSAGES, safe_json_dumps(normalized))

    if not response:
        return

    # Output text
    content = getattr(response, "content", None)
    if content:
        span.set_attribute(OI.OUTPUT_VALUE, content)

    # Tool calls (OpenInference format)
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        # Phoenix usually expects output messages containing tool_calls
        out_messages = [{"role": "assistant", "content": "", "tool_calls": tool_calls}]
        span.set_attribute(OI.LLM_OUTPUT_MESSAGES, safe_json_dumps(out_messages))

    # Usage
    usage = getattr(response, "usage", None)
    if isinstance(usage, dict):
        span.set_attribute(OI.LLM_TOKEN_COUNT_PROMPT, int(usage.get("prompt_tokens", 0)))
        span.set_attribute(OI.LLM_TOKEN_COUNT_COMPLETION, int(usage.get("completion_tokens", 0)))
        span.set_attribute(OI.LLM_TOKEN_COUNT_TOTAL, int(usage.get("total_tokens", 0)))

    # Cache (custom, fine)
    if hasattr(response, "get_cache_info"):
        cache_info = response.get_cache_info() or {}
        if cache_info:
            span.set_attribute("repoa.cache.cached_tokens", int(cache_info.get("cached_tokens", 0)))
            span.set_attribute("repoa.cache.hit_rate", float(cache_info.get("cache_hit_rate", 0.0)))

def extract_tool_attributes(span, tool_name: str, arguments: Dict, result: Any):
    span.set_attribute(OI.TOOL_NAME, tool_name)
    span.set_attribute(OI.INPUT_VALUE, safe_json_dumps(arguments))
    span.set_attribute(OI.OUTPUT_VALUE, safe_json_dumps(result))
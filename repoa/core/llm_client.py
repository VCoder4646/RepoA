"""
LLM Client Module
Handles integration with various LLM backends: Ollama, vLLM, and remote endpoints.
Supports tool calling and streaming responses.
"""

import json
import requests
from typing import List, Dict, Any, Optional, Callable, Generator
from enum import Enum
from abc import ABC, abstractmethod
# from repoa.instrumentation.tracing import traced_span, OpenInferenceSpanKindValues
from openinference.semconv.trace import OpenInferenceSpanKindValues
from repoa.instrumentation.decorators import trace_repoa

class ModelType(Enum):
    """Enumeration of supported model types."""
    OLLAMA = "ollama"
    VLLM = "vllm"
    REMOTE = "remote"
    OPENAI_COMPATIBLE = "openai_compatible"


class LLMResponse:
    """
    Represents a response from an LLM.
    
    Attributes:
        content: The text content of the response
        tool_calls: List of tool calls requested by the model
        finish_reason: Reason for completion (stop, tool_calls, length, etc.)
        usage: Token usage information (including cache data)
        raw_response: Raw response from the API
        cached_tokens: Number of tokens served from KV cache
    """
    
    def __init__(
        self,
        content: str = "",
        tool_calls: Optional[List[Dict[str, Any]]] = None,
        finish_reason: str = "stop",
        usage: Optional[Dict[str, int]] = None,
        raw_response: Optional[Dict[str, Any]] = None,
        cached_tokens: int = 0
    ):
        self.content = content
        self.tool_calls = tool_calls or []
        self.finish_reason = finish_reason
        self.usage = usage or {}
        self.raw_response = raw_response or {}
        self.cached_tokens = cached_tokens
    
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return len(self.tool_calls) > 0
    
    def has_cache_hit(self) -> bool:
        """Check if response used KV cache."""
        return self.cached_tokens > 0 or self._extract_cached_tokens() > 0
    
    def _extract_cached_tokens(self) -> int:
        """Extract cached token count from usage data."""
        # OpenAI format
        if "prompt_tokens_details" in self.usage:
            return self.usage["prompt_tokens_details"].get("cached_tokens", 0)
        
        # Anthropic format
        if "cache_read_input_tokens" in self.usage:
            return self.usage.get("cache_read_input_tokens", 0)
        
        # vLLM/generic format
        if "cached_tokens" in self.usage:
            return self.usage.get("cached_tokens", 0)
        
        return self.cached_tokens
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get cache information from response."""
        cached = self._extract_cached_tokens()
        total_prompt = self.usage.get("prompt_tokens", 0)
        
        return {
            "cached_tokens": cached,
            "cache_hit": cached > 0,
            "cache_hit_rate": cached / total_prompt if total_prompt > 0 else 0.0,
            "prompt_tokens": total_prompt,
            "uncached_tokens": total_prompt - cached
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert response to dictionary."""
        return {
            "content": self.content,
            "tool_calls": self.tool_calls,
            "finish_reason": self.finish_reason,
            "usage": self.usage,
            "cached_tokens": self.cached_tokens,
            "cache_info": self.get_cache_info()
        }


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str,
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        **kwargs
    ):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Name of the model to use
            base_url: Base URL for the API
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional configuration
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip('/')
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.config = kwargs
    
    @abstractmethod
    @trace_repoa(kind=OpenInferenceSpanKindValues.LLM)
    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> LLMResponse:
        """
        Generate a response from the LLM.
        
        Args:
            messages: List of message dictionaries
            tools: Optional list of tools the model can use
            stream: Whether to stream the response
            
        Returns:
            LLMResponse object
        """
        pass
    @trace_repoa(kind=OpenInferenceSpanKindValues.LLM)
    def _make_request(
        self,
        endpoint: str,
        payload: Dict[str, Any],
        stream: bool = False
    ) -> requests.Response:
        """
        Make an HTTP request to the API.
        
        Args:
            endpoint: API endpoint
            payload: Request payload
            stream: Whether to stream the response
            
        Returns:
            Response object
        """
        url = f"{self.base_url}{endpoint}"
        headers = {
            "Content-Type": "application/json"
        }
        
        # Add API key if provided
        if "api_key" in self.config:
            headers["Authorization"] = f"Bearer {self.config['api_key']}"
        
        response = requests.post(
            url,
            json=payload,
            headers=headers,
            stream=stream,
            timeout=self.config.get("timeout", 300)
        )
        response.raise_for_status()
        return response


class OllamaClient(BaseLLMClient):
    """
    Client for Ollama API.
    Ollama runs models locally on your machine.
    """
    
    def __init__(
        self,
        model_name: str = "llama2",
        base_url: str = "http://localhost:11434",
        **kwargs
    ):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Ollama model name (e.g., 'llama2', 'mistral', 'codellama')
            base_url: Ollama server URL
            **kwargs: Additional configuration
        """
        super().__init__(model_name, base_url, **kwargs)
    
    @trace_repoa(kind=OpenInferenceSpanKindValues.LLM)
    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> LLMResponse:
        """
        Generate response using Ollama.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            tools: Optional tools for function calling
            stream: Whether to stream response
            
        Returns:
            LLMResponse object
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
            }
        }
        
        if self.max_tokens:
            payload["options"]["num_predict"] = self.max_tokens
        
        # Add tools if provided (Ollama supports function calling in newer versions)
        if tools:
            payload["tools"] = tools
        
        try:
            response = self._make_request("/api/chat", payload, stream=stream)
            data = response.json()
            
            # Parse response
            content = ""
            tool_calls = []
            
            if "message" in data:
                content = data["message"].get("content", "")
                
                # Check for tool calls
                if "tool_calls" in data["message"]:
                    tool_calls = data["message"]["tool_calls"]
            
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=data.get("done_reason", "stop"),
                usage={
                    "prompt_tokens": data.get("prompt_eval_count", 0),
                    "completion_tokens": data.get("eval_count", 0),
                    "total_tokens": data.get("prompt_eval_count", 0) + data.get("eval_count", 0),
                    # Ollama doesn't directly expose cache info, but we can track it
                    "cached_tokens": 0  # Would need Ollama API enhancement
                },
                raw_response=data,
                cached_tokens=0
            )
        
        except Exception as e:
            raise RuntimeError(f"Ollama API error: {str(e)}")
    
    def list_models(self) -> List[str]:
        """
        List available Ollama models.
        
        Returns:
            List of model names
        """
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
        except Exception as e:
            raise RuntimeError(f"Error listing Ollama models: {str(e)}")


class VLLMClient(BaseLLMClient):
    """
    Client for vLLM OpenAI-compatible API.
    vLLM provides high-throughput serving for LLMs.
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str = "http://localhost:8000",
        **kwargs
    ):
        """
        Initialize vLLM client.
        
        Args:
            model_name: Model name (must be loaded in vLLM server)
            base_url: vLLM server URL
            **kwargs: Additional configuration
        """
        super().__init__(model_name, base_url, **kwargs)
    
    @trace_repoa(kind=OpenInferenceSpanKindValues.LLM)
    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> LLMResponse:
        """
        Generate response using vLLM.
        
        Args:
            messages: List of message dictionaries
            tools: Optional tools for function calling
            stream: Whether to stream response
            
        Returns:
            LLMResponse object
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            response = self._make_request("/v1/chat/completions", payload, stream=stream)
            data = response.json()
            
            # Parse OpenAI-compatible response
            choice = data["choices"][0]
            message = choice["message"]
            
            content = message.get("content", "") or ""
            tool_calls = []
            
            # Parse tool calls if present
            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    tool_calls.append({
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    })
            
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=choice.get("finish_reason", "stop"),
                usage=data.get("usage", {}),
                raw_response=data,
                cached_tokens=data.get("usage", {}).get("cached_tokens", 0)
            )
        
        except Exception as e:
            raise RuntimeError(f"vLLM API error: {str(e)}")


class RemoteEndpointClient(BaseLLMClient):
    """
    Generic client for remote OpenAI-compatible endpoints.
    Works with OpenAI API, Azure OpenAI, and other compatible services.
    """
    
    def __init__(
        self,
        model_name: str,
        base_url: str,
        api_key: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize remote endpoint client.
        
        Args:
            model_name: Model name
            base_url: API base URL
            api_key: API key for authentication
            **kwargs: Additional configuration
        """
        if api_key:
            kwargs["api_key"] = api_key
        super().__init__(model_name, base_url, **kwargs)
    
    @trace_repoa(kind=OpenInferenceSpanKindValues.LLM)
    def generate(
        self,
        messages: List[Dict[str, str]],
        tools: Optional[List[Dict[str, Any]]] = None,
        stream: bool = False
    ) -> LLMResponse:
        """
        Generate response using remote endpoint.
        
        Args:
            messages: List of message dictionaries
            tools: Optional tools for function calling
            stream: Whether to stream response
            
        Returns:
            LLMResponse object
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": self.temperature,
            "stream": False
        }
        
        if self.max_tokens:
            payload["max_tokens"] = self.max_tokens
        
        # Add tools if provided
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        try:
            # Determine endpoint
            endpoint = self.config.get("endpoint", "/v1/chat/completions")
            response = self._make_request(endpoint, payload, stream=stream)
            data = response.json()
            
            # Parse OpenAI-compatible response
            choice = data["choices"][0]
            message = choice["message"]
            
            content = message.get("content", "") or ""
            tool_calls = []
            
            # Parse tool calls if present
            if "tool_calls" in message and message["tool_calls"]:
                for tc in message["tool_calls"]:
                    tool_calls.append({
                        "id": tc.get("id", ""),
                        "type": tc.get("type", "function"),
                        "function": {
                            "name": tc["function"]["name"],
                            "arguments": tc["function"]["arguments"]
                        }
                    })
            
            return LLMResponse(
                content=content,
                tool_calls=tool_calls,
                finish_reason=choice.get("finish_reason", "stop"),
                usage=data.get("usage", {}),
                raw_response=data,
                cached_tokens=data.get("usage", {}).get("cached_tokens", 0)
            )
        
        except Exception as e:
            raise RuntimeError(f"Remote endpoint API error: {str(e)}")


class LLMClientFactory:
    """
    Factory class for creating LLM clients.
    """
    
    @staticmethod
    def create_client(
        model_type: ModelType,
        model_name: str,
        base_url: Optional[str] = None,
        **kwargs
    ) -> BaseLLMClient:
        """
        Create an LLM client based on the model type.
        
        Args:
            model_type: Type of model (Ollama, vLLM, Remote)
            model_name: Name of the model
            base_url: Base URL for the API (uses defaults if not provided)
            **kwargs: Additional configuration
            
        Returns:
            LLM client instance
        """
        if model_type == ModelType.OLLAMA:
            url = base_url or "http://localhost:11434"
            return OllamaClient(model_name, url, **kwargs)
        
        elif model_type == ModelType.VLLM:
            url = base_url or "http://localhost:8000"
            return VLLMClient(model_name, url, **kwargs)
        
        elif model_type in [ModelType.REMOTE, ModelType.OPENAI_COMPATIBLE]:
            if not base_url:
                raise ValueError("base_url is required for remote endpoints")
            return RemoteEndpointClient(model_name, base_url, **kwargs)
        
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    @staticmethod
    def create_ollama(model_name: str = "llama2", **kwargs) -> OllamaClient:
        """Quick helper to create Ollama client."""
        return OllamaClient(model_name, **kwargs)
    
    @staticmethod
    def create_vllm(model_name: str, base_url: str = "http://localhost:8000", **kwargs) -> VLLMClient:
        """Quick helper to create vLLM client."""
        return VLLMClient(model_name, base_url, **kwargs)
    
    @staticmethod
    def create_remote(model_name: str, base_url: str, api_key: Optional[str] = None, **kwargs) -> RemoteEndpointClient:
        """Quick helper to create remote endpoint client."""
        return RemoteEndpointClient(model_name, base_url, api_key, **kwargs)


def format_tools_for_api(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Format tools for OpenAI-compatible API.
    
    Args:
        tools: List of tool dictionaries from ToolProcessor
        
    Returns:
        Formatted tools list
    """
    formatted = []
    for tool in tools:
        formatted.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool["description"],
                "parameters": tool["parameters"]
            }
        })
    return formatted


def parse_tool_call_arguments(arguments_str: str) -> Dict[str, Any]:
    """
    Parse tool call arguments from JSON string.
    
    Args:
        arguments_str: JSON string of arguments
        
    Returns:
        Parsed arguments dictionary
    """
    try:
        return json.loads(arguments_str)
    except json.JSONDecodeError:
        return {}

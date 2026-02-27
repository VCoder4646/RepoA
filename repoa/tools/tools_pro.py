"""
Tools Processing Module
Handles MCP (Model Context Protocol) tools processing and formatting.
"""

from typing import List, Dict, Any, Optional, Callable
import json
from dataclasses import dataclass, field
from enum import Enum
from openinference.semconv.trace import OpenInferenceSpanKindValues


class ToolType(Enum):
    """Enumeration of tool types."""
    MCP = "mcp"
    CUSTOM = "custom"
    BUILTIN = "builtin"


@dataclass
class ToolParameter:
    """
    Represents a single tool parameter.
    
    Attributes:
        name: Parameter name
        type: Parameter type (string, integer, boolean, object, array)
        description: Parameter description
        required: Whether this parameter is required
        default: Default value if any
        enum: List of allowed values if restricted
    """
    name: str
    type: str
    description: str
    required: bool = False
    default: Any = None
    enum: Optional[List[Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert parameter to dictionary format."""
        param_dict = {
            "type": self.type,
            "description": self.description
        }
        if self.default is not None:
            param_dict["default"] = self.default
        if self.enum:
            param_dict["enum"] = self.enum
        return param_dict


@dataclass
class Tool:
    """
    Represents a tool that can be used by an agent.
    
    Attributes:
        name: Tool name/identifier
        description: What the tool does
        parameters: List of tool parameters
        tool_type: Type of tool (MCP, custom, builtin)
        function: Optional callable function for custom tools
        metadata: Additional metadata about the tool
    """
    name: str
    description: str
    parameters: List[ToolParameter] = field(default_factory=list)
    tool_type: ToolType = ToolType.MCP
    function: Optional[Callable] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_required_params(self) -> List[str]:
        """Get list of required parameter names."""
        return [param.name for param in self.parameters if param.required]
    
    def get_optional_params(self) -> List[str]:
        """Get list of optional parameter names."""
        return [param.name for param in self.parameters if not param.required]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert tool to dictionary format."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    param.name: param.to_dict() 
                    for param in self.parameters
                },
                "required": self.get_required_params()
            },
            "tool_type": self.tool_type.value,
            "metadata": self.metadata
        }
    

    def validate_args(self, args: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """
        Validate arguments against tool parameters.
        
        Args:
            args: Dictionary of arguments to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required parameters
        required = self.get_required_params()
        for param_name in required:
            if param_name not in args:
                return False, f"Missing required parameter: {param_name}"
        
        # Check for unknown parameters
        valid_params = {param.name for param in self.parameters}
        for arg_name in args:
            if arg_name not in valid_params:
                return False, f"Unknown parameter: {arg_name}"
        
        return True, None


class ToolProcessor:
    """
    Processes and manages tools for agent use.
    Handles MCP tools formatting and custom tool registration.
    """
    
    def __init__(self):
        """Initialize the tool processor."""
        self._tools: Dict[str, Tool] = {}
        self._tool_groups: Dict[str, List[str]] = {}
    
    def add_tool(self, tool: Tool) -> None:
        """
        Add a tool to the processor.
        
        Args:
            tool: Tool instance to add
        """
        self._tools[tool.name] = tool
    
    def remove_tool(self, tool_name: str) -> bool:
        """
        Remove a tool from the processor.
        
        Args:
            tool_name: Name of the tool to remove
            
        Returns:
            True if removed, False if not found
        """
        if tool_name in self._tools:
            del self._tools[tool_name]
            return True
        return False
    
    def get_tool(self, tool_name: str) -> Optional[Tool]:
        """
        Get a tool by name.
        
        Args:
            tool_name: Name of the tool
            
        Returns:
            Tool instance or None if not found
        """
        return self._tools.get(tool_name)
    
    def list_tools(self, tool_type: Optional[ToolType] = None) -> List[str]:
        """
        List all available tools, optionally filtered by type.
        
        Args:
            tool_type: Optional filter by tool type
            
        Returns:
            List of tool names
        """
        if tool_type is None:
            return list(self._tools.keys())
        return [
            name for name, tool in self._tools.items() 
            if tool.tool_type == tool_type
        ]
    
    def create_tool_group(self, group_name: str, tool_names: List[str]) -> None:
        """
        Create a named group of tools.
        
        Args:
            group_name: Name for the tool group
            tool_names: List of tool names to include
        """
        self._tool_groups[group_name] = tool_names
    
    def get_tool_group(self, group_name: str) -> List[Tool]:
        """
        Get all tools in a named group.
        
        Args:
            group_name: Name of the tool group
            
        Returns:
            List of Tool instances
        """
        tool_names = self._tool_groups.get(group_name, [])
        return [self._tools[name] for name in tool_names if name in self._tools]
    

    def parse_mcp_tool(self, mcp_tool_data: Dict[str, Any]) -> Tool:
        """
        Parse MCP tool data into a Tool instance.
        
        Args:
            mcp_tool_data: MCP tool data in dictionary format
            
        Returns:
            Tool instance
        """
        name = mcp_tool_data.get("name", "")
        description = mcp_tool_data.get("description", "")
        
        # Parse parameters
        parameters = []
        input_schema = mcp_tool_data.get("inputSchema", {})
        properties = input_schema.get("properties", {})
        required = input_schema.get("required", [])
        
        for param_name, param_info in properties.items():
            param = ToolParameter(
                name=param_name,
                type=param_info.get("type", "string"),
                description=param_info.get("description", ""),
                required=param_name in required,
                default=param_info.get("default"),
                enum=param_info.get("enum")
            )
            parameters.append(param)
        
        return Tool(
            name=name,
            description=description,
            parameters=parameters,
            tool_type=ToolType.MCP,
            metadata={"original_schema": mcp_tool_data}
        )
    
    def load_mcp_tools(self, mcp_tools: List[Dict[str, Any]]) -> int:
        """
        Load multiple MCP tools at once.
        
        Args:
            mcp_tools: List of MCP tool data dictionaries
            
        Returns:
            Number of tools successfully loaded
        """
        count = 0
        for tool_data in mcp_tools:
            try:
                tool = self.parse_mcp_tool(tool_data)
                self.add_tool(tool)
                count += 1
            except Exception as e:
                print(f"Error loading tool {tool_data.get('name', 'unknown')}: {e}")
        return count
    

    def format_for_llm(self, tool_names: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Format tools for LLM consumption.
        
        Args:
            tool_names: Optional list of specific tools to format. If None, formats all.
            
        Returns:
            List of formatted tool dictionaries
        """
        if tool_names is None:
            tools_to_format = self._tools.values()
        else:
            tools_to_format = [self._tools[name] for name in tool_names if name in self._tools]
        
        return [tool.to_dict() for tool in tools_to_format]
    

    def export_tools_json(self, filepath: str, tool_names: Optional[List[str]] = None) -> None:
        """
        Export tools to a JSON file.
        
        Args:
            filepath: Path to save JSON file
            tool_names: Optional list of specific tools to export
        """
        formatted_tools = self.format_for_llm(tool_names)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(formatted_tools, f, indent=2)
    

    def import_tools_json(self, filepath: str) -> int:
        """
        Import tools from a JSON file.
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            Number of tools imported
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            tools_data = json.load(f)
        
        return self.load_mcp_tools(tools_data)
    
    def get_tool_info(self) -> Dict[str, Any]:
        """
        Get information about all loaded tools.
        
        Returns:
            Dictionary with tool statistics and information
        """
        return {
            "total_tools": len(self._tools),
            "by_type": {
                tool_type.value: len(self.list_tools(tool_type))
                for tool_type in ToolType
            },
            "tool_groups": list(self._tool_groups.keys()),
            "tools": list(self._tools.keys())
        }
    
    def __len__(self) -> int:
        """Return the number of tools."""
        return len(self._tools)
    
    def __repr__(self) -> str:
        return f"ToolProcessor(tools={len(self._tools)})"


def create_custom_tool(
    name: str,
    description: str,
    parameters: List[Dict[str, Any]],
    function: Optional[Callable] = None
) -> Tool:
    """
    Helper function to create a custom tool.
    
    Args:
        name: Tool name
        description: Tool description
        parameters: List of parameter dictionaries with keys:
                   - name (str): Parameter name
                   - type (str): Parameter type (string, number, integer, boolean, object, array)
                   - description (str): Parameter description
                   - required (bool, optional): Whether parameter is required (default: False)
                   - default (Any, optional): Default value if not required
                   - enum (List[Any], optional): List of allowed values
        function: Optional function to execute when tool is called
        
    Returns:
        Tool instance
        
    Example:
        tool = create_custom_tool(
            name="calculator",
            description="Perform basic arithmetic",
            parameters=[
                {"name": "a", "type": "number", "description": "First number", "required": True},
                {"name": "b", "type": "number", "description": "Second number", "required": True}
            ],
            function=lambda a, b: {"result": a + b}
        )
    """
    tool_params = []
    for param in parameters:
        tool_params.append(ToolParameter(
            name=param["name"],
            type=param.get("type", "string"),
            description=param.get("description", ""),
            required=param.get("required", False),
            default=param.get("default"),
            enum=param.get("enum")
        ))
    
    return Tool(
        name=name,
        description=description,
        parameters=tool_params,
        tool_type=ToolType.CUSTOM,
        function=function
    )

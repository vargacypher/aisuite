from typing import Callable, Dict, Any, Type, Optional
from pydantic import BaseModel, create_model, Field, ValidationError
import inspect


class ToolManager:
    def __init__(self):
        self._tools = {}

    # Add a tool function with or without a Pydantic model.
    def add_tool(self, func: Callable, param_model: Optional[Type[BaseModel]] = None):
        """Register a tool function with metadata. If no param_model is provided, infer from function signature."""
        if param_model:
            tool_spec = self._convert_to_tool_spec(func, param_model)
        else:
            tool_spec, param_model = self._infer_from_signature(func)

        self._tools[func.__name__] = {
            "function": func,
            "param_model": param_model,
            "spec": tool_spec,
        }

    def tools(self, format="openai") -> list:
        """Return tools in the specified format (default OpenAI)."""
        if format == "openai":
            return self._convert_to_openai_format()
        return [tool["spec"] for tool in self._tools.values()]

    def _convert_to_tool_spec(
        self, func: Callable, param_model: Type[BaseModel]
    ) -> Dict[str, Any]:
        """Convert the function and its Pydantic model to a unified tool specification."""
        properties = {
            field.alias: {
                "type": str(field.type_),
                "description": field.field_info.description or "",
                "default": field.default if field.default is not None else None,
            }
            for field in param_model.model_fields.values()
        }

        required_fields = [
            field.alias
            for field in param_model.model_fields.values()
            if field.default is None
        ]

        return {
            "name": func.__name__,
            "description": func.__doc__ or "No description provided.",
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required_fields,
            },
        }

    def _infer_from_signature(
        self, func: Callable
    ) -> tuple[Dict[str, Any], Type[BaseModel]]:
        """Infer parameters(required and optional) and requirements directly from the function signature."""
        signature = inspect.signature(func)
        fields = {}
        required_fields = []

        for param_name, param in signature.parameters.items():
            # Check if a type annotation is missing
            if param.annotation == inspect._empty:
                raise TypeError(
                    f"Parameter '{param_name}' in function '{func.__name__}' must have a type annotation."
                )

            # Determine field type and optionality
            param_type = param.annotation
            if param.default == inspect._empty:
                fields[param_name] = (param_type, ...)
                required_fields.append(param_name)
            else:
                fields[param_name] = (param_type, Field(default=param.default))

        # Dynamically create a Pydantic model based on inferred fields
        param_model = create_model(f"{func.__name__.capitalize()}Params", **fields)

        # Convert inferred model to a tool spec format
        tool_spec = self._convert_to_tool_spec(func, param_model)
        return tool_spec, param_model

    def _convert_to_openai_format(self) -> list:
        """Convert tools to OpenAI's format."""
        return [
            {"type": "function", "function": tool["spec"]}
            for tool in self._tools.values()
        ]

    def execute_tool(self, tool_call: Dict[str, Any]) -> tuple:
        """Executes a registered tool based on the tool call from the model."""
        tool_name = tool_call.get("name")
        if tool_name not in self._tools:
            raise ValueError(f"Tool '{tool_name}' not registered.")

        tool = self._tools[tool_name]
        tool_func = tool["function"]
        param_model = tool["param_model"]
        arguments = tool_call.get("arguments", {})

        # Validate and parse the arguments with Pydantic if a model exists
        try:
            validated_args = param_model(**arguments)
            result = tool_func(
                **validated_args.dict()
            )  # Execute the tool with validated args
            result_message = {
                "role": "assistant",
                "content": f"Result from {tool_name}: {result}",
            }
            return result, result_message
        except ValidationError as e:
            error_message = f"Error in tool '{tool_name}' parameters: {e}"
            return {"error": error_message}, {
                "role": "assistant",
                "content": error_message,
            }


# Example tool function with all parameters having type annotations
def get_current_temperature(location: str, unit: str = "Celsius"):
    """Gets the current temperature for a specific location and unit."""
    # Simulate fetching temperature from an API
    return {"location": location, "unit": unit, "temperature": 72}

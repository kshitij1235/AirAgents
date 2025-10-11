import json
import inspect
from typing import Any, Dict, Optional
from abc import ABC, abstractmethod


class Tool(ABC):
    """
    Base class for all Air tools.

    Just inherit, set name and description, override run() with your logic.
    Schema is automatically generated from run() function signature.

    Example:
        class MyTool(Tool):
            def __init__(self):
                super().__init__(
                    name="my_tool",
                    description="Does something useful"
                )

            def run(self, query: str, limit: int = 10) -> str:
                # your code
                return json.dumps({"result": "..."})
    """

    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description

    @abstractmethod
    def run(self, **kwargs) -> str:
        """
        Override this with your tool logic.
        Must return a string (usually JSON).
        """
        pass

    def _get_run_signature(self) -> inspect.Signature:
        """Get the signature of the run() method."""
        return inspect.signature(self.run)

    def _type_to_json_type(self, type_hint: Any) -> str:
        """Convert Python type hints to JSON schema types."""
        if type_hint == str:
            return "string"
        elif type_hint == int:
            return "integer"
        elif type_hint == float:
            return "number"
        elif type_hint == bool:
            return "boolean"
        elif type_hint == list:
            return "array"
        elif type_hint == dict:
            return "object"
        else:
            return "string"  # default

    def get_schema(self) -> Dict[str, Any]:
        """
        Automatically generate JSON schema from run() signature.
        """
        sig = self._get_run_signature()
        properties = {}
        required = []

        for param_name, param in sig.parameters.items():
            # Skip 'self' and 'kwargs'
            if param_name in ["self", "kwargs"]:
                continue

            # Get type annotation
            type_hint = (
                param.annotation if param.annotation != inspect.Parameter.empty else str
            )
            json_type = self._type_to_json_type(type_hint)

            # Check if parameter has a default value
            has_default = param.default != inspect.Parameter.empty

            prop = {
                "type": json_type,
                "description": f"{param_name} parameter",
            }

            # Add default if present
            if has_default:
                prop["default"] = param.default

            properties[param_name] = prop

            # Add to required if no default
            if not has_default:
                required.append(param_name)

        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        }

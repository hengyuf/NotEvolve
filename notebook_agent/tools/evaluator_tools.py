"""Protected evaluator tools.

Evaluator logic is injected externally. The LLM can call these tools
but cannot inspect or edit evaluator source code via notebook-agent tools.
"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import logging
from pathlib import Path
from typing import Any, Callable

from notebook_agent.tools.base import BaseTool, ToolResult

logger = logging.getLogger(__name__)


class EvaluatorTool(BaseTool):
    """A tool that runs an evaluator function."""

    def __init__(
        self,
        tool_name: str,
        tool_description: str,
        tool_parameters: dict,
        eval_fn: Callable[..., Any],
    ):
        self._name = tool_name
        self._description = tool_description
        self._parameters = tool_parameters
        self._eval_fn = eval_fn

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters(self) -> dict:
        return self._parameters

    async def execute(self, **kwargs: Any) -> ToolResult:
        try:
            result = self._eval_fn(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return ToolResult(content=str(result))
        except Exception as e:
            logger.error("Evaluator %s failed: %s", self._name, e, exc_info=True)
            return ToolResult(content=f"Evaluator error: {e}", is_error=True)


def _load_module(module_spec: str):
    """Load evaluator module from either import path or Python file path."""
    maybe_path = Path(module_spec)
    if maybe_path.suffix == ".py" or maybe_path.exists():
        path = maybe_path.resolve()
        module_name = f"notebook_agent_evaluator_{abs(hash(str(path)))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load evaluator module from file: {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    return importlib.import_module(module_spec)


def _schema_from_annotation(annotation: Any) -> dict:
    """Map simple Python annotations to JSON Schema."""
    if annotation is inspect.Parameter.empty:
        return {"type": "string"}
    origin = getattr(annotation, "__origin__", None)
    if annotation is list or origin is list:
        return {"type": "array", "items": {"type": "number"}}
    if annotation is tuple or origin is tuple:
        return {"type": "array", "items": {"type": "number"}}
    if annotation is float:
        return {"type": "number"}
    if annotation is int:
        return {"type": "integer"}
    if annotation is bool:
        return {"type": "boolean"}
    if annotation is str:
        return {"type": "string"}
    return {"type": "string"}


def _autodiscovered_specs(mod: Any) -> list[dict]:
    """Auto-discover evaluator tools from public evaluate/check functions."""
    specs: list[dict] = []
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if name.startswith("_"):
            continue
        if not (name.startswith("evaluate") or name.startswith("check")):
            continue

        sig = inspect.signature(fn)
        properties: dict[str, dict] = {}
        required: list[str] = []
        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue
            properties[param_name] = _schema_from_annotation(param.annotation)
            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        schema: dict[str, Any] = {"type": "object", "properties": properties}
        if required:
            schema["required"] = required

        doc = (fn.__doc__ or "").strip()
        description = doc.splitlines()[0] if doc else f"Run {name}"

        specs.append(
            {
                "name": name,
                "function": name,
                "description": description,
                "parameters": schema,
            }
        )
    return specs


def load_evaluator_tools(evaluator_config: dict) -> list[EvaluatorTool]:
    """Load evaluator tools.

    Supported config:
    {
      "module": "path.to.module" or "/abs/path/to/file.py",
      "tools": [ ... optional explicit tool specs ... ]
    }

    If `tools` is omitted/empty, tools are auto-discovered from public
    functions starting with `evaluate` or `check`.
    """
    module_spec = evaluator_config.get("module")
    if not module_spec:
        return []

    try:
        mod = _load_module(module_spec)
    except ImportError as e:
        logger.error("Failed to load evaluator module %s: %s", module_spec, e)
        return []

    tool_specs = evaluator_config.get("tools") or _autodiscovered_specs(mod)
    tools: list[EvaluatorTool] = []
    for tool_spec in tool_specs:
        fn_name = tool_spec.get("function")
        fn = getattr(mod, fn_name, None)
        if fn is None:
            logger.warning("Evaluator function %s not found in %s", fn_name, module_spec)
            continue

        tool = EvaluatorTool(
            tool_name=tool_spec["name"],
            tool_description=tool_spec.get("description", f"Run {fn_name}"),
            tool_parameters=tool_spec.get("parameters", {"type": "object", "properties": {}}),
            eval_fn=fn,
        )
        tools.append(tool)
    return tools


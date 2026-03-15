"""Configuration for notebook-agent."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentConfig(BaseModel):
    """Top-level configuration for the notebook-agent system."""

    # LLM
    llm_provider: str = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    llm_api_key: str = ""
    llm_temperature: float = 0.0
    llm_max_tokens: int = 16384
    llm_thinking_budget: int = 10000
    llm_base_url: str | None = None
    llm_extra_headers: dict[str, str] = Field(default_factory=dict)

    # Notebook
    notebook_path: str = "notebook.ipynb"
    kernel_name: str = "python3"
    working_dir: str = "."

    # Context rendering
    unfold_last_n: int = 3
    max_output_lines: int = 30
    max_source_lines_folded: int = 1

    # Execution
    default_timeout: int = 120

    # Security
    forbidden_code_patterns: list[str] = Field(default_factory=lambda: [
        r"pip\s+install",
        r"pip3\s+install",
        r"!pip",
        r"%pip",
        r"subprocess\.",
        r"os\.system\s*\(",
    ])
    evaluator_paths: list[str] = Field(default_factory=list)
    evaluator_module: str = ""
    evaluator_tools: list[dict] = Field(default_factory=list)

    # Persistence
    checkpoint_dir: str = ".checkpoints"
    auto_save: bool = True

    # Engine
    max_rounds: int = 50
    max_tool_calls_per_round: int = 30
    max_sessions: int = 1
    rounds_per_session: int = 0  # 0 = use max_rounds for the single session

    # Task
    task: str = ""
    system_prompt_extra: str = ""

"""Tests for security policy."""

import pytest

from notebook_agent.security.policy import SecurityError, SecurityPolicy


class TestSecurityPolicy:
    def test_blocks_pip_install(self):
        policy = SecurityPolicy(forbidden_code_patterns=[r"pip\s+install", r"!pip", r"%pip"])
        with pytest.raises(SecurityError, match="pip"):
            policy.check_code("!pip install numpy")

    def test_blocks_pip_install_variations(self):
        policy = SecurityPolicy(forbidden_code_patterns=[r"pip\s+install", r"!pip", r"%pip", r"subprocess\."])
        with pytest.raises(SecurityError):
            policy.check_code("pip install numpy")
        with pytest.raises(SecurityError):
            policy.check_code("%pip install numpy")
        with pytest.raises(SecurityError):
            policy.check_code("import subprocess; subprocess.run(['pip', 'install', 'x'])")

    def test_allows_normal_code(self):
        policy = SecurityPolicy(forbidden_code_patterns=[r"pip\s+install"])
        policy.check_code("import numpy as np")  # Should not raise
        policy.check_code("x = 42")

    def test_blocks_subprocess(self):
        policy = SecurityPolicy(forbidden_code_patterns=[r"subprocess\."])
        with pytest.raises(SecurityError):
            policy.check_code("import subprocess\nsubprocess.run(['ls'])")

    def test_blocks_code_referencing_evaluator_file(self, tmp_path):
        eval_file = tmp_path / "evaluator_adapter.py"
        eval_file.write_text("x = 1")
        policy = SecurityPolicy(evaluator_paths=[str(eval_file)], working_dir=str(tmp_path))
        with pytest.raises(SecurityError):
            policy.check_code("open('evaluator_adapter.py').read()")

    def test_blocks_evaluator_path(self, tmp_path):
        eval_dir = tmp_path / "evaluator"
        eval_dir.mkdir()
        secret_file = eval_dir / "secret.py"
        secret_file.write_text("score = 42")

        policy = SecurityPolicy(evaluator_paths=[str(eval_dir)], working_dir=str(tmp_path))
        with pytest.raises(SecurityError, match="protected"):
            policy.check_path(str(secret_file))

    def test_allows_non_evaluator_path(self, tmp_path):
        eval_dir = tmp_path / "evaluator"
        eval_dir.mkdir()
        other_file = tmp_path / "code.py"
        other_file.write_text("x = 1")

        policy = SecurityPolicy(evaluator_paths=[str(eval_dir)], working_dir=str(tmp_path))
        policy.check_path(str(other_file))  # Should not raise

    def test_is_path_allowed(self, tmp_path):
        eval_dir = tmp_path / "evaluator"
        eval_dir.mkdir()
        secret = eval_dir / "secret.py"
        secret.write_text("")

        policy = SecurityPolicy(evaluator_paths=[str(eval_dir)], working_dir=str(tmp_path))
        assert not policy.is_path_allowed(str(secret))
        assert policy.is_path_allowed(str(tmp_path / "code.py"))

    def test_check_tool_call_path_validation(self, tmp_path):
        eval_dir = tmp_path / "evaluator"
        eval_dir.mkdir()
        secret = eval_dir / "secret.py"
        secret.write_text("")

        policy = SecurityPolicy(evaluator_paths=[str(eval_dir)], working_dir=str(tmp_path))
        with pytest.raises(SecurityError):
            policy.check_tool_call("read_file", {"path": str(secret)})

    def test_blocks_paths_outside_working_dir(self, tmp_path):
        workdir = tmp_path / "workspace"
        workdir.mkdir()
        outside_file = tmp_path / "outside.txt"
        outside_file.write_text("x")

        policy = SecurityPolicy(working_dir=str(workdir))
        with pytest.raises(SecurityError, match="outside the working directory"):
            policy.check_path(str(outside_file))

    def test_relative_paths_are_resolved_from_working_dir(self, tmp_path):
        workdir = tmp_path / "workspace"
        workdir.mkdir()
        inside_file = workdir / "inside.txt"
        inside_file.write_text("ok")

        policy = SecurityPolicy(working_dir=str(workdir))
        policy.check_path("inside.txt")  # Should not raise

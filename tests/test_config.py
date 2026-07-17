"""Tests for agent.config module."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.config import (
    AgentConfig,
    EmbeddingConfig,
    LLMConfig,
    LLMProvider,
    MemoryConfig,
    SecurityConfig,
    load_config,
)


class TestLLMConfig:
    """Tests for LLM configuration."""

    def test_default_provider_is_deepseek(self):
        cfg = LLMConfig()
        assert cfg.provider == LLMProvider.DEEPSEEK
        assert "deepseek" in cfg.base_url

    def test_ollama_placeholder_key(self, monkeypatch):
        monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
        cfg = LLMConfig(provider="ollama")
        assert cfg.api_key == "ollama"
        assert "localhost" in cfg.base_url

    def test_api_key_from_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test-123")
        cfg = LLMConfig(provider="deepseek")
        assert cfg.api_key == "sk-test-123"

    def test_explicit_overrides_env(self, monkeypatch):
        monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-env")
        cfg = LLMConfig(provider="deepseek", api_key="sk-explicit")
        assert cfg.api_key == "sk-explicit"

    def test_openai_provider(self):
        cfg = LLMConfig(provider="openai", api_key="sk-test")
        assert cfg.provider == LLMProvider.OPENAI
        assert "openai" in cfg.base_url


class TestSecurityConfig:
    """Tests for security configuration."""

    def test_defaults(self):
        cfg = SecurityConfig()
        assert cfg.command_timeout == 120
        assert cfg.max_output_chars == 50000
        assert cfg.require_approval is True

    def test_custom_values(self):
        cfg = SecurityConfig(command_timeout=300, require_approval=False)
        assert cfg.command_timeout == 300
        assert cfg.require_approval is False


class TestAgentConfig:
    """Tests for agent configuration."""

    def test_defaults(self):
        cfg = AgentConfig()
        assert cfg.max_iterations == 15
        assert cfg.max_consecutive_failures == 3


def test_memory_auto_capture_defaults_and_limits():
    config = MemoryConfig()
    assert config.auto_capture_decisions is True
    assert config.auto_capture_max_chars == 2000
    with pytest.raises(ValueError):
        MemoryConfig(auto_capture_max_chars=100)


def test_embedding_offline_defaults_and_timeout_limits():
    config = EmbeddingConfig()
    assert config.local_files_only is True
    assert config.hub_timeout_seconds == 10
    with pytest.raises(ValueError):
        EmbeddingConfig(hub_timeout_seconds=0)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_load_with_valid_repo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / "demo-repo").mkdir()
        monkeypatch.setenv("AGENT_REPO_ROOT", "./demo-repo")
        monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
        llm, security, agent = load_config()
        assert security.repo_root.exists()
        assert security.repo_root.is_dir()

    def test_load_fails_with_missing_repo(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("AGENT_REPO_ROOT", "./nonexistent")
        with pytest.raises(ValueError, match="does not exist"):
            load_config()

"""Agent configuration using Pydantic Settings.

All secrets are loaded from environment variables or .env file.
No secrets are hardcoded in source code.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings


class LLMProvider(str, Enum):
    DEEPSEEK = "deepseek"
    OPENAI = "openai"
    OLLAMA = "ollama"


# Default provider configurations
_PROVIDER_DEFAULTS: dict[LLMProvider, dict[str, str]] = {
    LLMProvider.DEEPSEEK: {
        "base_url": "https://api.deepseek.com",
        "model": "deepseek-v4-flash",
    },
    LLMProvider.OPENAI: {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o",
    },
    LLMProvider.OLLAMA: {
        "base_url": "http://localhost:11434/v1",
        "model": "qwen2.5-coder:7b",
    },
}


class LLMConfig(BaseSettings):
    """LLM provider configuration.

    API keys are read from environment variables:
    - DEEPSEEK_API_KEY for DeepSeek
    - OPENAI_API_KEY for OpenAI
    - Ollama requires no API key (set to "ollama" as placeholder)
    """

    model_config = {"env_prefix": ""}

    provider: LLMProvider = Field(default=LLMProvider.DEEPSEEK)
    api_key: str = Field(default="")
    base_url: str = Field(default="")
    model: str = Field(default="")
    temperature: float = Field(default=0.0)
    max_tokens: int = Field(default=4096)

    @model_validator(mode="after")
    def resolve_defaults(self) -> "LLMConfig":
        """Fill in provider-specific defaults when not explicitly set."""
        defaults = _PROVIDER_DEFAULTS[self.provider]
        import os

        provider_prefix = self.provider.value.upper()
        provider_base_url = os.environ.get(f"{provider_prefix}_BASE_URL", "")
        provider_model = os.environ.get(f"{provider_prefix}_MODEL", "")
        if not self.base_url:
            self.base_url = provider_base_url or defaults["base_url"]
        if not self.model:
            self.model = provider_model or defaults["model"]
        # Auto-fill API key from provider-specific env var
        if not self.api_key:
            env_map = {
                LLMProvider.DEEPSEEK: "DEEPSEEK_API_KEY",
                LLMProvider.OPENAI: "OPENAI_API_KEY",
                LLMProvider.OLLAMA: "OLLAMA_API_KEY",
            }
            env_var = env_map[self.provider]
            self.api_key = os.environ.get(env_var, "")
        # Ollama uses a placeholder key
        if self.provider == LLMProvider.OLLAMA and not self.api_key:
            self.api_key = "ollama"
        return self


class SecurityConfig(BaseSettings):
    """Security configuration for the agent.

    Controls path protection, command execution limits, and approval flow.
    """

    model_config = {"env_prefix": "AGENT_"}

    repo_root: Path = Field(default=Path("./demo-repo"))
    command_timeout: int = Field(default=120)
    max_output_chars: int = Field(default=50000)
    require_approval: bool = Field(default=True)


class AgentConfig(BaseSettings):
    """Agent loop configuration.

    Controls iteration limits and failure handling.
    """

    model_config = {"env_prefix": "AGENT_"}

    max_iterations: int = Field(default=15)
    max_consecutive_failures: int = Field(default=3)


class WorkflowConfig(BaseSettings):
    """Multi-Agent workflow limits."""

    model_config = {"env_prefix": "WORKFLOW_"}

    max_rework: int = Field(default=2, ge=0, le=5)


# ============================================================
# Phase 2: RAG / Embedding / Memory / MCP Configs
# ============================================================


class EmbeddingProvider(str, Enum):
    LOCAL = "local"  # sentence-transformers (BAAI/bge-small-zh-v1.5)
    OPENAI = "openai"  # OpenAI text-embedding-3-small


class EmbeddingConfig(BaseSettings):
    """Embedding service configuration.

    LOCAL uses sentence-transformers with BAAI/bge-small-zh-v1.5.
    OPENAI uses text-embedding-3-small via OpenAI API.
    """

    model_config = {"env_prefix": "EMBEDDING_"}

    provider: EmbeddingProvider = Field(default=EmbeddingProvider.LOCAL)
    model_name: str = Field(default="BAAI/bge-small-zh-v1.5")
    openai_model: str = Field(default="text-embedding-3-small")
    dimension: int = Field(default=512, description="Embedding dimension")
    batch_size: int = Field(default=32, description="Batch size for embedding")


class RAGConfig(BaseSettings):
    """RAG retrieval configuration.

    Controls hybrid search parameters, reranking, and agentic RAG behavior.
    """

    model_config = {"env_prefix": "RAG_"}

    # Vector search
    vector_top_k: int = Field(default=20, description="Top-K from vector search")
    # BM25 search
    bm25_top_k: int = Field(default=20, description="Top-K from BM25 search")
    # Fusion
    rrf_k: int = Field(default=60, description="RRF constant k (score = 1/(k+rank))")
    fusion_top_k: int = Field(default=15, description="Top-K after fusion")
    # Rerank
    rerank_enabled: bool = Field(default=False)
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    rerank_top_k: int = Field(default=10, description="Top-K after reranking")
    # Agentic RAG
    max_retrieval_rounds: int = Field(default=3, description="Max retrieval rounds")
    evidence_threshold: float = Field(default=0.6, description="Evidence sufficiency threshold")
    # ChromaDB
    chroma_collection: str = Field(default="java_code_chunks")
    chroma_persist_dir: str = Field(default=".chroma")
    # Index lifecycle
    enable_vector: bool = Field(
        default=True,
        description="Whether to build/query the vector index; false uses BM25 only",
    )
    index_dir: str = Field(
        default=".agent-index",
        description="Persistent incremental index directory",
    )
    force_reindex: bool = Field(
        default=False,
        description="Ignore cached file hashes and rebuild the index",
    )


class MemoryConfig(BaseSettings):
    """Memory system configuration."""

    model_config = {"env_prefix": "MEMORY_"}

    short_term_window: int = Field(default=20, description="Max messages in short-term window")
    summary_trigger: int = Field(default=30, description="Trigger summary after N messages")
    max_summary_tokens: int = Field(default=1000, description="Max tokens for summary")
    long_term_persist_dir: str = Field(default=".memory", description="Long-term memory storage dir")
    checkpoint_dir: str = Field(default=".checkpoints", description="LangGraph checkpointer dir")


class MCPConfig(BaseSettings):
    """MCP server configuration."""

    model_config = {"env_prefix": "MCP_"}

    enabled: bool = Field(default=True)
    transport: str = Field(default="stdio", description="Phase 3A only supports stdio")
    server_name: str = Field(default="java-coding-agent")
    server_version: str = Field(default="0.2.0")


class ObservabilityConfig(BaseSettings):
    """Trace persistence and optional user-supplied model pricing."""

    model_config = {"env_prefix": "OBSERVABILITY_"}

    enabled: bool = True
    trace_dir: str = ".observability/traces"
    input_cost_per_million: float | None = Field(default=None, ge=0)
    output_cost_per_million: float | None = Field(default=None, ge=0)


def load_config() -> tuple[LLMConfig, SecurityConfig, AgentConfig]:
    """Load all configurations from environment / .env file.

    Returns (llm_config, security_config, agent_config).
    """
    from dotenv import load_dotenv

    load_dotenv()
    llm = LLMConfig()
    security = SecurityConfig()
    agent = AgentConfig()

    # Resolve repo_root to absolute path and validate
    repo_root = security.repo_root.resolve()
    if not repo_root.exists():
        raise ValueError(f"Repository root does not exist: {repo_root}")
    if not repo_root.is_dir():
        raise ValueError(f"Repository root is not a directory: {repo_root}")
    security.repo_root = repo_root

    return llm, security, agent


def load_rag_config() -> RAGConfig:
    """Load RAG configuration from environment / .env file."""
    from dotenv import load_dotenv

    load_dotenv()
    return RAGConfig()


def load_embedding_config() -> EmbeddingConfig:
    """Load embedding configuration from environment / .env file."""
    from dotenv import load_dotenv

    load_dotenv()
    return EmbeddingConfig()


def load_memory_config() -> MemoryConfig:
    """Load memory configuration from environment / .env file."""
    from dotenv import load_dotenv

    load_dotenv()
    return MemoryConfig()


def load_mcp_config() -> MCPConfig:
    """Load MCP configuration from environment / .env file."""
    from dotenv import load_dotenv

    load_dotenv()
    return MCPConfig()


def load_workflow_config() -> WorkflowConfig:
    """Load Multi-Agent workflow configuration."""
    from dotenv import load_dotenv

    load_dotenv()
    return WorkflowConfig()


def load_observability_config() -> ObservabilityConfig:
    """Load trace and optional pricing configuration."""
    from dotenv import load_dotenv

    load_dotenv()
    return ObservabilityConfig()

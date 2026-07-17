"""End-to-end workflow smoke test with real Patch, Maven and Git tools."""

import shutil
import subprocess
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from agent.config import AgentConfig, LLMConfig, MemoryConfig, RAGConfig, WorkflowConfig
from agent.session import SessionManager
from agent.tools.factory import create_tool_registry
from agent.workflow import create_workflow, initial_workflow_state


class RealToolScriptedLLM:
    def bind_tools(self, tools):
        return self

    def invoke(self, messages):
        text = "\n".join(str(message.content) for message in messages)
        if "任务路由专家" in text:
            return AIMessage(content='{"agent":"coder","reason":"修复代码"}')
        if "检索查询改写专家" in text:
            return AIMessage(content='["calculateTotal", "OrderItem getSubtotal"]')
        if "证据评估专家" in text:
            return AIMessage(content='{"sufficient":false,"confidence":0.2,"reason":"继续由Coder定位"}')
        if "代码审查专家" in text:
            return AIMessage(content='{"approved":true,"issues":[],"suggestions":[],"summary":"Diff正确且测试通过"}')
        if any(isinstance(message, ToolMessage) for message in messages):
            return AIMessage(content="已完成calculateTotal修复")
        return AIMessage(
            content="",
            tool_calls=[{
                "id": "real-patch",
                "name": "apply_patch",
                "args": {
                    "path": "src/main/java/com/example/order/OrderService.java",
                    "unified_diff": (
                        "@@ -41,2 +41,2 @@\n"
                        "-                .map(OrderItem::getUnitPrice)\n"
                        "+                .map(OrderItem::getSubtotal)\n"
                    ),
                },
            }],
        )


class EmptySearch:
    def search(self, query, top_k=10):
        return []


def test_real_patch_maven_and_verifier_flow(tmp_path):
    source = Path(__file__).parent.parent / "demo-repo"
    repo = tmp_path / "order-service"
    shutil.copytree(source, repo, ignore=shutil.ignore_patterns("target"))
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.name=Eval", "-c", "user.email=eval@example.invalid", "commit", "-qm", "baseline"],
        cwd=repo,
        check=True,
    )
    target_file = repo / "src/main/java/com/example/order/OrderService.java"
    # Keep this integration fixture deterministic even if demo-repo contains
    # the already-fixed implementation.
    target_file.write_text(
        target_file.read_text(encoding="utf-8").replace(
            "                .map(OrderItem::getSubtotal)",
            "                .map(OrderItem::getUnitPrice)",
            1,
        ),
        encoding="utf-8",
    )
    assert "OrderItem::getUnitPrice" in target_file.read_text(encoding="utf-8")
    baseline_content = target_file.read_bytes()

    llm = RealToolScriptedLLM()
    manager = SessionManager(MemoryConfig(
        checkpoint_dir=str(tmp_path / "checkpoints"),
        long_term_persist_dir=str(tmp_path / "memory"),
    ), llm=llm)
    session_id = manager.create_session()
    graph = create_workflow(
        llm=llm,
        llm_config=LLMConfig(provider="ollama"),
        agent_config=AgentConfig(),
        workflow_config=WorkflowConfig(max_rework=1),
        rag_config=RAGConfig(),
        tool_registry=create_tool_registry(repo),
        search_engine=EmptySearch(),
        session_manager=manager,
        repo_root=repo,
    )
    config = manager.get_thread_config(session_id)
    interrupted = graph.invoke(initial_workflow_state("修复calculateTotal忽略数量的Bug", session_id), config)
    assert interrupted["__interrupt__"]
    result = graph.invoke(Command(resume={"approved": True}), config)

    assert result["test_result"].success is True
    assert result["review_result"].approved is True
    assert "OrderItem::getSubtotal" in target_file.read_text(encoding="utf-8")
    # Verify the patch itself, independently of Git's platform-specific
    # line-ending normalization (which can make `git diff --name-only`
    # empty even when the file was changed and then normalized).
    assert target_file.read_bytes() != baseline_content
    manager.close()

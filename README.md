# Java Coding Agent

一个面向 Java 仓库的本地 Coding Agent。它既能回答代码问题，也能在审批后修改文件、运行测试、审查结果并在失败时返工。主流程由 LangGraph 编排，RAG、Memory、MCP、Multi-Agent、可观测性、CLI 和 Streamlit 操作台共用同一套 `AppService`。

## 功能

- 代码问答与文件/行号引用。
- `git diff`、`git status`、`git log` 等明确只读命令直接返回真实仓库输出，不经过 RAG 改写；操作台根据结构化渲染提示使用 Diff/Text 代码块，保留 `$`、`~`、缩进和换行。
- Java 方法级索引、BM25 与可选向量混合检索。
- Supervisor、Researcher、Coder、Tester、Verifier 协作。
- Patch 前人工审批，仓库路径和命令安全限制。
- SQLite Checkpoint、短期/摘要/长期记忆。
- MCP stdio 工具调用。
- Trace、Token、工具成功率与可选成本统计。
- CLI、Streamlit UI、隔离式评估和 Docker 部署。

## 本地启动

要求 Python 3.11+、JDK 17、Maven 或目标仓库自带 Maven Wrapper。

```powershell
cd "E:\code agent\java-coding-agent"
py -3.14 -m pip install -e ".[dev,ui]"
Copy-Item .env.example .env
```

在 `.env` 中选择 DeepSeek、OpenAI 或 Ollama，并设置对应模型和密钥。首次体验建议先关闭向量下载：

```dotenv
RAG_ENABLE_VECTOR=false
AGENT_REPO_ROOT=./demo-repo
```

CLI：

```powershell
coding-agent
```

Streamlit 操作台：

```powershell
py -3.14 -m streamlit run src/agent/ui/app.py
```

打开 `http://localhost:8501`。页面可切换仓库，新建、切换或删除 Session，查看 Agent 时间线、Diff、测试、Token、Trace，并处理 Patch 审批。删除操作需要二次确认，并同步清理消息状态、事件、Checkpoint 和 Trace。

左侧运行指标可以选择“当前会话”“当前项目”或“全部”。项目范围使用仓库绝对路径的稳定标识筛选；升级前没有项目标识的旧 Trace 仍计入全部和对应会话，但不会被猜测归入某个项目。

## 配置

完整配置见 [.env.example](.env.example)。常用项：

| 变量 | 作用 |
|---|---|
| `PROVIDER` | `deepseek`、`openai` 或 `ollama` |
| `AGENT_REPO_ROOT` | Agent 可访问的仓库根目录 |
| `AGENT_REQUIRE_APPROVAL` | 写操作是否必须审批 |
| `RAG_ENABLE_VECTOR` | 是否启用向量检索；关闭后使用 BM25 |
| `RAG_FORCE_REINDEX` | 是否强制重建索引 |
| `MCP_ENABLED` | 是否启用 Researcher MCP stdio 路径 |
| `OBSERVABILITY_*` | Trace 目录和可选 Token 单价 |

密钥只放在本地环境或 `.env`，不要提交。修改过真实密钥后应在供应商控制台轮换。

## 评估

每个任务会复制 `demo-repo` 到独立临时 Git 仓库，执行固定 Bug 注入、Agent 流程、编译/测试和确定性断言，结束后清理副本。LLM Judge 默认关闭，只有显式注入 Judge LLM 才会产生评分。

```powershell
py -3.14 -m agent.eval.runner --fixture demo-repo --output reports --runs 1
```

该命令会调用配置的真实 Agent 模型并可能产生费用。当前仓库不附带伪造的在线评估数字，状态见 [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md)。

## Docker

Docker Compose 默认启动 Streamlit，工作区可写，其余根文件系统只读；容器以非 root 用户运行，移除 Linux capabilities，不挂载 Docker Socket，并设置 CPU、内存和进程限制。

```powershell
$env:AGENT_WORKSPACE = "E:/my-java-repo"
docker compose up --build
```

Windows CLI 也可直接挂载：

```powershell
docker run --rm -it -v D:/my-java-repo:/home/agent/workspace java-coding-agent coding-agent
```

Ollama 在宿主机运行时，Compose 使用 `host.docker.internal` 访问。Linux 上需要按 Docker 环境补充 host gateway 或改为可访问的 Ollama 地址。

## 验证

```powershell
py -3.14 -m pytest -q
py -3.14 -m compileall -q src tests
py -3.14 -m pip check
```

架构、Demo 和简历表述分别见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)、[DEMO.md](docs/DEMO.md) 和 [RESUME.md](docs/RESUME.md)。

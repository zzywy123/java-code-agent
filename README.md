# Java Coding Agent

一个面向 Java 仓库的本地 Coding Agent。它既能回答代码问题，也能在审批后修改文件、运行测试、审查结果并在失败时返工。主流程由 LangGraph 编排，RAG、Memory、MCP、Multi-Agent、可观测性、CLI 和 Streamlit 操作台共用同一套 `AppService`。

## 功能

- 代码问答与文件/行号引用。
- `git diff`、`git status`、`git log` 等明确只读命令直接返回真实仓库输出，不经过 RAG 改写；操作台根据结构化渲染提示使用 Diff/Text 代码块，保留 `$`、`~`、缩进和换行。
- Java 方法级索引、BM25 与可选向量混合检索。
- Supervisor、Researcher、Coder、Tester、Verifier 协作。
- Patch 前人工审批；批准后由 Tester 自动运行白名单测试，不重复审批。
- Patch 使用 `git apply --check/apply/reverse`，构建目标走 Maven/Gradle 白名单并按构建文件自动识别。
- SQLite Checkpoint、短期/摘要记忆；工作流成功结束后自动提取稳定决策并去重写入长期记忆，也支持显式写入。
- MCP stdio 是 Researcher 的首选只读工具通道：启动时完成能力发现，搜索/Git 结果作为证据进入后续 Agent，失败时回退本地 ToolRegistry。
- Trace、Token、工具成功率与可选成本统计。
- CLI、Streamlit UI、隔离式评估和 Docker 部署。

## 本地启动

要求 Python 3.11+、JDK 17，以及 Maven/Gradle 或目标仓库自带对应 Wrapper。

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

工作流成功结束后会额外调用一次模型，只在识别到稳定的 preference、convention 或 decision 时写入长期记忆，并产生 `memory_saved` 事件。CLI 也可显式保存，例如：

```text
/remember convention java-style Java Service 使用构造器注入
```

自动提取器禁止保存代码事实、Bug 状态、测试结果和临时过程；这些信息每次都从当前仓库重新验证。可通过 `MEMORY_AUTO_CAPTURE_DECISIONS=false` 关闭自动提取。

Streamlit 操作台：

```powershell
py -3.14 -m streamlit run src/agent/ui/app.py
```

打开 `http://localhost:8501`。页面支持三种仓库导入方式：

- “本地文件夹”会打开浏览器的目录选择器并上传目录内容。浏览器不会把客户端真实路径交给服务器，上传时会忽略 `.git`、构建产物和密钥文件，再由服务器创建一份 Git 基线。
- “Git 地址”会从允许的 HTTPS 代码托管平台浅克隆公开仓库，也可指定分支。
- “服务器路径”保留原有路径输入，只允许访问 `AGENT_SERVER_PATH_ROOTS` 配置的目录。

导入后可新建、切换或删除 Session，查看 Agent 时间线、Diff、测试、Token、Trace，并处理 Patch 审批。不同浏览器会话使用隔离的上传目录、Checkpoint 和长期记忆。删除操作需要二次确认，并同步清理消息状态、事件、Checkpoint 和 Trace。

左侧运行指标可以选择“当前会话”“当前项目”或“全部”。项目范围使用仓库绝对路径的稳定标识筛选；升级前没有项目标识的旧 Trace 仍计入全部和对应会话，但不会被猜测归入某个项目。

## 配置

完整配置见 [.env.example](.env.example)。常用项：

| 变量 | 作用 |
|---|---|
| `PROVIDER` | `deepseek`、`openai` 或 `ollama` |
| `AGENT_REPO_ROOT` | Agent 可访问的仓库根目录 |
| `AGENT_WORKSPACE_ROOT` | 浏览器上传和 Git 克隆仓库的服务端持久化目录 |
| `AGENT_SERVER_PATH_ROOTS` | “服务器路径”允许访问的根目录，按操作系统路径分隔符分隔 |
| `AGENT_GIT_ALLOWED_HOSTS` | Git HTTPS 克隆主机白名单，逗号分隔 |
| `AGENT_UPLOAD_MAX_FILES` / `AGENT_UPLOAD_MAX_MB` | 单次导入的有效文件数与总容量限制 |
| `AGENT_REQUIRE_APPROVAL` | Patch 等写操作是否必须审批；主工作流批准 Patch 后自动测试 |
| `RAG_ENABLE_VECTOR` | 是否启用向量检索；关闭后使用 BM25 |
| `EMBEDDING_LOCAL_FILES_ONLY` | 本地 Embedding 是否只使用缓存；默认开启，避免首问访问 Hugging Face |
| `EMBEDDING_HUB_TIMEOUT_SECONDS` | 允许在线下载模型时的 Hugging Face 超时秒数 |
| `RAG_FORCE_REINDEX` | 是否强制重建索引 |
| `MEMORY_AUTO_CAPTURE_DECISIONS` | 是否在工作流成功结束后自动沉淀稳定决策 |
| `MEMORY_AUTO_CAPTURE_MAX_CHARS` | 决策提取时任务和结果各自的最大字符数 |
| `MCP_ENABLED` | 是否让 Researcher 优先使用 MCP stdio 只读工具通道 |
| `OBSERVABILITY_*` | Trace 目录和可选 Token 单价 |

密钥只放在本地环境或 `.env`，不要提交。修改过真实密钥后应在供应商控制台轮换。

## 评估

每个任务会复制 `demo-repo` 到独立临时 Git 仓库，执行固定 Bug 注入、Agent 流程、编译/测试和确定性断言，结束后清理副本。修改文件统计来自 `git status --porcelain`，包含未跟踪的新文件；报告代码版本取主 Agent 项目而非 Demo fixture。LLM Judge 默认关闭，只有显式注入 Judge LLM 才会产生评分。

```powershell
py -3.14 -m agent.eval.runner --fixture demo-repo --output reports --runs 1
```

该命令会调用配置的真实 Agent 模型并可能产生费用。当前仓库不附带伪造的在线评估数字，状态见 [docs/EVALUATION_REPORT.md](docs/EVALUATION_REPORT.md)。

## Docker

Docker Compose 默认启动 Streamlit，初始仓库和上传工作区可写，其余根文件系统只读；上传仓库保存在独立的 `agent-workspaces` 卷。容器以非 root 用户运行，移除 Linux capabilities，不挂载 Docker Socket，并设置 CPU、内存和进程限制。

```powershell
$env:AGENT_WORKSPACE = "E:/my-java-repo"
docker compose up --build
```

Windows CLI 也可直接挂载：

```powershell
docker run --rm -it -v D:/my-java-repo:/home/agent/workspace java-coding-agent coding-agent
```

Ollama 在宿主机运行时，Compose 使用 `host.docker.internal` 访问。Linux 上需要按 Docker 环境补充 host gateway 或改为可访问的 Ollama 地址。

公网部署仍应放在登录鉴权之后。Maven/Gradle 构建脚本属于仓库代码，执行测试时虽然会清除 API Key、Token、Secret 和 Password 类环境变量，并受到容器资源限制，但单个共享容器不等同于强多租户沙箱；不要把该部署方式直接开放给不受信任的匿名用户。

## 验证

```powershell
py -3.14 -m pytest -q
py -3.14 -m compileall -q src tests
py -3.14 -m pip check
```

架构、Demo 和简历表述分别见 [ARCHITECTURE.md](docs/ARCHITECTURE.md)、[DEMO.md](docs/DEMO.md) 和 [RESUME.md](docs/RESUME.md)。

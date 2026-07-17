# 实施状态

## Phase 1

状态：已完成。

已具备 OpenAI-compatible LLM、LangGraph 单 Agent Loop、受控文件/Git/构建工具、Patch 审批、CLI 和 Spring Boot Demo。

## Phase 2

状态：已完成模块实现；当前项目全量 379 个测试通过，2 个可选场景跳过。

索引、Hybrid Search、Agentic RAG、Memory、角色权限、Artifact 和官方 MCP SDK 已存在。本阶段此前主要以独立模块测试为主。

## Phase 3A：真实集成门

状态：已完成本地确定性集成，待真实 LLM API 冒烟。

本次接入内容：

- 父级 LangGraph 工作流：Supervisor → Researcher → Coder → Tester → Verifier。
- Verifier 驳回后的 Coder 返工循环和最大返工次数。
- SQLite 持久化 Checkpointer 和 SessionManager。
- ShortTerm/Summary/LongTerm Memory 上下文注入；成功工作流自动提取并去重沉淀稳定决策，AppService/CLI 显式入口继续保留。
- Researcher 通过官方 MCP Python SDK 的 stdio Client 完成能力发现，搜索和 Git 只读工具优先经 MCP 调用，证据进入 Agentic RAG 后续上下文。
- 角色受限 ToolRegistry，Coder、Tester、Verifier 不共享越权工具。
- Demo Maven Wrapper 和已修复的 calculateTotal 示例。

验证：

- Python 全量测试：`379 passed, 2 skipped`。
- Python静态编译：通过。
- 真实 Patch + Maven Wrapper + Git Diff + Verifier 集成测试通过。
- Patch 使用 `git apply --check/apply/reverse`，支持带上下文和零上下文 hunk，并保留内容哈希校验。
- Maven/Gradle goal/task 使用结构化白名单；Tester 根据标准构建文件自动选择工具。
- `search_code` 与 `list_files` 对每个结果执行统一路径保护，敏感文件不会返回给 Agent。
- 本地 Embedding 启动时从缓存单次预热；并发请求不会重复加载，缓存缺失或初始化失败会立即降级 BM25。
- 同一 Session 的新请求会清空任务级 Search/Test/Review Artifact 并创建新 Trace，避免复用旧审查结论。
- Verifier 可在没有 CodeChangeArtifact 时直接审查真实工作区 Diff。
- 真实 stdio MCP `initialize → tools/list → tools/call` 测试通过。
- MCP 调用记录独立 Trace/指标，协议异常和未声明工具会回退本地只读 ToolRegistry。
- SQLite Checkpoint 重启恢复测试通过。
- Session 消息历史会保留 AI tool call 与 ToolMessage 的相邻关系；孤立工具结果降级为 SystemMessage，避免 OpenAI `role=tool` 协议错误。
- 真实工具端到端测试固定注入故意 Bug，并直接校验补丁改变目标文件，避免 Git 行尾归一化导致假失败或假阳性。
- 共享 Runtime 已接入持久化增量索引：缓存代码块、文件哈希和 Chroma 向量；未修改文件不会重复生成 Embedding。
- 支持 `RAG_ENABLE_VECTOR=false` 的 BM25-only 模式，以及 `RAG_FORCE_REINDEX=true` 的显式全量重建。
- 明确写操作使用确定性意图保护，不能被 LLM 错误路由为只读问答。
- Agentic RAG 会补充检索 `Class::method` 依赖，EvidenceJudge 使用标识符覆盖和排名，不再错误比较 RRF 绝对分数。
- Demo Maven 测试：8 个测试全部通过；故意 Bug 只在隔离的端到端测试副本中注入。
- 用户已运行真实在线 CLI；本轮路由与 RAG 修复仍需再次执行同一场景做在线回归。

## Phase 3B：AppService 与统一事件模型

状态：已完成。

- `AppService` 提供 `create_session`、`list_sessions`、`submit`、`resume`、`get_session` 和 `stream_events`。
- `AppService.remember_project_fact` 提供受类型限制的显式长期记忆写入闭环。
- 自动沉淀产生持久化 `memory_saved` 事件，CLI 与 Streamlit 时间线均可查看，重复内容不会重复写入。
- CLI 只调用 AppService，不再直接执行 LangGraph invoke/resume。
- CLI 中重复的旧索引构建、旧单 Agent Runner 和旧 Multi-Agent 拼装代码已删除。
- 父图与 Coder 子图的真实 LangGraph 更新映射为统一 `StreamEvent`。
- 覆盖 Agent 切换、RAG、工具调用/结果、审批、Patch、测试、审查、返工、错误和完成事件。
- 事件以 JSONL 持久化到 Checkpoint 目录，审批恢复按消息、tool call 和 interrupt ID 去重。
- Session 列表、事件历史和 LangGraph Checkpoint 均支持进程重启恢复。

## Phase 3C：可观测性

状态：已完成。

- Trace 使用 ContextVar 按 Session 隔离，不写入 LangGraph State。
- 覆盖 workflow、Supervisor、Researcher、RAG round/search、Coder、LLM、工具、Tester 和 Verifier Span。
- 审批中断保存为 `interrupted`，进程重启后 `resume` 继续同一个 Trace ID。
- LLM 优先读取 `usage_metadata`；缺失时使用 tiktoken 估算并标记 `estimated=true`。
- 成本费率必须通过环境变量显式配置；未配置或 Ollama 场景返回 `cost=None`。
- ToolRegistry 统一记录调用状态、耗时、成功、失败和超时指标，不记录参数值或工具输出。
- structlog 和标准 logging 均执行敏感字段/密钥文本脱敏。
- AppService 已实现 `get_trace(session_id)` 和 `get_metrics()`，并生成 `token_usage` 事件。
- Trace 持久化到 `.observability/traces/<trace_id>.json`。

## Phase 3D：Streamlit 工程操作台

状态：已完成实现、应用边界测试、Mock UI 测试和真实运行时冒烟。

- 使用共享运行时工厂和 `AppService`，UI 不直接依赖 LangGraph、ToolRegistry 或 RAG。
- 支持仓库加载、Session 新建/切换、聊天、Patch 批准/拒绝。
- 支持 Session 二次确认删除；同步清理元数据、SQLite Checkpoint、内存缓存、事件和 Trace，删除最后一个会话后自动新建。
- Researcher 对明确的 `git diff/status/log` 请求直接调用真实只读工具，跳过 RAG 回答生成；`git diff` 现在会检查进程退出码，非 Git 仓库不再误报“没有变更”。
- Git 直通结果携带结构化 `render_hint`，经 LangGraph Message 和 AppService 保留到 UI；聊天中的真实 Git Diff 使用 `st.code(language="diff")`，不再被 Markdown/LaTeX 破坏。
- Trace 持久化仓库路径生成的稳定 `project_id`；左侧指标可选择当前会话、当前项目或全部。旧 Trace 保留在全部/会话范围，不会被错误归入项目。
- 支持 Agent 状态、事件时间线、Diff、测试、Token 和 Trace 面板。
- Streamlit 官方 AppTest 使用真实共享运行时启动通过，服务健康检查通过。
- 应用内 Browser 插件因当前运行时的只读 `process` 属性冲突无法初始化，因此桌面/窄屏截图验收未完成；未用独立浏览器工具替代。

## Phase 3E：隔离式评估

状态：框架与确定性测试已完成，真实付费模型评估未执行。

- 固定 8 个问答、修复、测试生成、返工和安全任务。
- 每个任务复制到独立临时 Git 仓库，setup hook 只修改副本。
- 支持编译、测试、引用、修改范围、符号、返工与安全断言。
- 修改范围通过 `git status --porcelain` 获取，包含未跟踪的新文件；代码版本记录主 Agent 项目。
- Windows Git 对象只读属性会在清理时恢复，不静默遗留目录。
- LLM Judge 默认关闭，报告不伪造模型能力数据。

## Phase 3F：Docker 与交付文档

状态：已完成文件生成与 Compose YAML 安全断言；当前机器未安装 Docker，镜像未实际构建。

- 多阶段镜像包含 Python 3.11、JDK 17、Maven 和 Streamlit。
- 容器使用非 root 用户、只读根文件系统、最小 capabilities 和资源限制。
- 工作区、索引、Checkpoint、Memory、Trace、模型和 Maven 缓存分离挂载。
- README、架构、评估、Demo、简历和自动化脚本已补齐。

## 最终验证

- `py -3.14 -m pytest -q`：379 passed，2 skipped。
- `py -3.14 -m compileall -q src tests`：通过。
- `py -3.14 -m pip check`：无依赖冲突。
- Compose YAML 解析、只读根文件系统、PID、capabilities 和 Docker Socket 安全断言：通过。
- Streamlit Mock UI、真实共享运行时冒烟和 `/_stcore/health`：通过。
- 未执行：真实付费 LLM 评估、Docker build、应用内浏览器桌面/窄屏截图。

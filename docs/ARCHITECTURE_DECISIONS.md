# 架构决策

## ADR-011：使用父级工作流整合 Phase 2

Phase 2 的独立 Agent 类不能代替真实执行链路，因此新增父级 LangGraph。它负责路由、Artifact 交接、Tester、Verifier 和返工；现有单 Agent 图作为受限 Coder 子图复用。

## ADR-012：Session 使用共享 SQLite Checkpointer

同一会话复用一个 Thread ID；所有 Thread 使用同一个 SQLite Checkpointer，通过配置中的 Thread ID 隔离状态。Session 元数据单独保存，不把每个 Thread 建成独立 Checkpointer。

## ADR-013：MCP 只作为协议适配层

MCP Server 使用官方 Python SDK 和 stdio transport，复用现有 ToolRegistry。Researcher 使用只读 MCP Server；权限在 Client Adapter 和 Server 两侧都校验。MCP 不复制文件、Git、Patch 或构建逻辑。

## ADR-014：长期记忆只保存可复用决策

长期记忆允许保存用户偏好、项目约定和架构决策。代码事实、Bug 状态和测试结果必须从当前工作区或工具重新验证，不能直接相信旧记忆。

## ADR-015：显式写意图优先于 LLM 路由

LLM 只负责模糊请求的语义路由。包含明确修改、修复或创建意图的请求必须进入 Coder 工作流；复合“修改并测试”同样先执行 Coder，再由工作流进入 Tester 和 Verifier，避免把有副作用的任务降级为只读问答。

## ADR-016：AppService 消费真实 LangGraph 更新流

CLI 和后续 Streamlit 只依赖 AppService。AppService 使用 `stream_mode=updates` 和 `subgraphs=True` 消费父图及 Coder 子图的真实更新，映射为统一事件；事件以 JSONL 持久化，并使用消息 ID、tool call ID 和 interrupt ID 处理审批恢复重放。

## ADR-017：Trace 独立于 LangGraph State

TraceCollector 通过 ContextVar 绑定当前 Session 和 Span，不把 Trace、Token 或工具指标写入 LangGraph Checkpoint。Trace 使用独立 JSON 文件持久化；审批中断关闭当前活动区间，恢复后继续相同 Trace，避免把等待用户审批的时间计入执行耗时。

## ADR-018：模型成本必须显式配置

内置价格表会随供应商调整而失效，因此系统只保证 Token 和延迟统计。输入/输出每百万 Token 费率由部署环境显式提供；未配置费率、部分费率缺失或本地 Ollama 模型均返回 `cost=None`，不输出伪精确费用。

## ADR-019：所有交互入口共享运行时与 AppService

CLI 和 Streamlit 通过同一个运行时工厂装配索引、LLM、MCP、Workflow、SessionManager 与 AppService。UI 只消费 AppService 的状态、事件、审批、Trace 和指标接口，避免入口之间出现不同的安全与恢复语义。

## ADR-020：评估任务使用临时 Git 仓库隔离

固定任务不能直接改写 Demo fixture。每次运行先复制仓库、应用明确的 setup hook，再初始化 Git 基线并创建独立 AppService。确定性断言基于最终文件、Diff、构建、事件和 Trace；结束后恢复 Windows 只读 Git 对象并删除副本。

## ADR-021：交付镜像采用非 root 与只读根文件系统

容器只把用户工作区和运行时缓存挂载为可写目录，不挂载 Docker Socket。镜像移除 capabilities、启用 `no-new-privileges` 并限制 CPU、内存和 PID；宿主仓库是否可写由显式 bind mount 决定。

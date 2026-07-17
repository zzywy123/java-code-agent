# 项目规格

## 产品目标

构建一个类似 Claude Code 的本地 Java Repository Coding Agent，支持代码问答、代码修改、测试执行、失败修正、真实引用、审批和可追溯工作流。

## 当前交付边界

Phase 3A 至 3F 已覆盖真实主链路、应用服务、可观测性、操作台、评估框架与容器交付：

```text
自然语言请求
→ Supervisor
→ Researcher/RAG
→ Coder Patch
→ Tester 构建
→ Verifier 审查
→ 通过或返工
→ 稳定决策提取 / 长期记忆去重
→ AppService 结构化事件
→ Trace / Token / Tool Metrics
→ CLI / Streamlit
→ 隔离式评估 / Docker 交付
```

真实在线模型评估属于显式付费验收，默认不执行，也不写入伪造指标。

## 安全边界

- 文件和构建工具限制在目标仓库内。
- 明确的只读 Git 查询必须调用真实 Git 工具；不得用 RAG 代码快照代替命令输出。
- Patch 写操作通过 LangGraph interrupt 审批；批准后自动运行受限 Tester，不重复审批。
- Unified Diff 使用 Git 原生 check/apply/reverse，构建目标和参数使用 Maven/Gradle 白名单。
- Agent 使用角色受限 ToolRegistry。
- API Key、源码和敏感工具参数不进入日志。
- 不自动执行 Git commit 或 push。

## 3A 完成标准

- 同一 Session 复用 Thread ID，SQLite Checkpoint 可重启恢复。
- Coder 实际写入 Patch，Tester 按构建文件实际执行 Maven/Gradle，Verifier 可脱离 CodeChangeArtifact 直接读取真实 Diff 和测试结果。
- Verifier拒绝后能回到Coder，且返工次数受限。
- Researcher 的搜索和直接 Git 读取优先使用真实 MCP stdio 工具调用，结果必须进入后续证据上下文，失败时回退本地只读工具。
- 成功工作流自动提取并去重保存稳定决策，失败、未批准和临时代码事实不得沉淀。
- 没有真实LLM时必须明确降级，不能伪造在线验收结果。

## 3B 完成标准

- CLI 通过 AppService 提交和恢复工作流，不直接依赖 LangGraph 执行细节。
- 审批中断返回结构化 `SubmitResult`，可使用同一 Session 恢复。
- LangGraph 父图和 Coder 子图产生统一、持久、可轮询的事件。
- 审批恢复不会重复发送旧工具调用事件。
- Session、Checkpoint 和事件历史支持进程重启恢复。

## 3C 完成标准

- 每个 Session 的工作流、Agent、RAG、LLM 和工具调用形成可持久化 Trace 树。
- 审批中断与进程重启不改变当前任务的 Trace ID。
- Token 优先使用 Provider 返回值，缺失时明确标记估算。
- 未配置价格时不生成伪精确成本。
- 工具指标覆盖成功、失败和超时，日志不包含密钥、完整 Prompt、源码或 Patch。
- AppService 提供 Trace 与全局指标查询，CLI 可消费 `token_usage` 事件。

## 3D 完成标准

- Streamlit 只通过共享运行时和 AppService 操作 Agent。
- 支持 Session 新建、切换、二次确认删除，以及聊天、审批、事件、Diff、测试、Token 和 Trace。
- 运行指标支持当前会话、当前项目和全部三个统计范围，项目归属必须来自 Trace 中持久化的稳定项目标识。
- 删除 Session 必须同步移除元数据、Checkpoint、事件与 Trace；删除最后一个 Session 后自动创建替代会话。
- 桌面与窄屏无关键控件遮挡或文本溢出。

## 3E 完成标准

- 8 个固定任务在独立临时 Git 仓库运行，原始 fixture 不变。
- setup hook、编译、测试、引用、安全和返工断言可独立测试。
- 报告记录主项目代码版本和可复现配置，修改范围包含未跟踪新文件；Judge 默认关闭，未知成本标记为 N/A。

## 3F 完成标准

- Docker 使用非 root 用户、受控可写挂载、独立缓存和资源限制。
- README、架构、评估、Demo、简历、环境变量与脚本保持一致。
- 未实际执行的 Docker 或真实 LLM 验收必须明确标记。

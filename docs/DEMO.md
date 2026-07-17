# Demo 手册

## 准备

```powershell
py -3.14 -m pip install -e ".[dev,ui]"
$env:AGENT_REPO_ROOT = "./demo-repo"
$env:RAG_ENABLE_VECTOR = "false"
```

启动 CLI 使用 `coding-agent`，启动操作台使用：

```powershell
py -3.14 -m streamlit run src/agent/ui/app.py
```

## 场景一：代码问答与引用

输入：

```text
解释 OrderService.calculateTotal 的作用，并引用对应文件和行号。
```

预期：Researcher 检索目标方法及依赖；回答包含真实 `.java:行号` 引用，不产生 Patch，也不请求审批。

## 场景二：审批后修改和测试

不要直接破坏 `demo-repo`。先复制一份演示仓库，在副本中把 `OrderService.calculateTotal` 的 `getSubtotal` 改为 `getUnitPrice`，然后让 Agent 修复：

```text
修复 OrderService.calculateTotal 忽略订单项数量的 Bug，并运行测试。
```

预期：Coder 提出 Patch 并暂停；批准后通过 Git 原生补丁应用变更；Tester 根据构建文件自动运行 Maven/Gradle 白名单测试，不再次审批；Verifier 读取实际 Diff 和测试结果；UI 的 Diff、测试和 Trace 标签出现对应数据。

## 场景三：安全拦截

输入：

```text
修改 .git/config，把 remote origin 改成其他地址。
```

预期：路径保护或角色权限拒绝修改，`.git/config` 哈希保持不变，事件时间线记录拒绝结果。不要为了演示关闭安全层。

## 演示后检查

- `git diff` 只包含审批过的业务改动。
- 测试结果来自实际 Maven 命令。
- `.observability/traces` 中存在对应 Trace。
- `.env`、API Key 和完整源码未出现在日志中。

## 场景四：自动长期记忆闭环

输入一个明确的长期决策，例如：

```text
将项目依赖注入约定统一为构造器注入，并完成必要修改。
```

预期：工作流通过测试和 Verifier 后，时间线先出现“长期记忆”，最后出现“完成”；新建 Session 后询问依赖注入约定，系统能注入该决策。重复执行不会新增相同内容。

显式写入仍可用于不需要代码修改的约定：

CLI 输入：

```text
/remember convention java-style Java Service 使用构造器注入
```

自动和显式入口都不会保存代码状态、Bug 状态或测试结果，这些事实仍从仓库重新读取。

## 场景五：MCP 实际调用

保持 `MCP_ENABLED=true`，输入 `git status` 或代码搜索问题。预期：启动日志显示 MCP 只读工具能力数量；Researcher analysis 标记 `MCP`，Trace 中出现 `mcp.git_status` 或 `mcp.search_code`。临时停止 MCP Server 时，同一请求应回退 `Local ToolRegistry` 而不中断工作流。

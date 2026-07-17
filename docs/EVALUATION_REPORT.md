# 评估报告

## 当前状态

固定评估框架和 8 个任务已经实现，但尚未运行真实在线 LLM 的完整评估。因此本文件不提供任务成功率、Token、成本或平均迭代次数，避免把单元测试结果误写成模型能力指标。

## 已验证的框架能力

- 每个任务使用独立临时 Git 仓库，原始 `demo-repo` 不变。
- 单文件 Bug 和参数校验缺失由 setup hook 在副本中注入。
- 支持引用、编译、测试、修改文件数量、符号、安全拒绝和返工断言。
- 修改文件通过 `git status --porcelain` 统计，包含 Agent 新建但尚未跟踪的测试文件。
- 保存模型、Prompt 版本、主 Agent 项目代码版本、运行次数、Trace、Token、耗时和可选成本。
- Windows 下清理 Git 只读对象，不静默遗留评估工作区。
- LLM-as-a-Judge 默认关闭；未配置 Judge 时评分为 `N/A`。

框架相关测试已包含在项目全量回归中；当前结果为 379 passed、2 skipped。该数字只表示代码测试结果，不代表 8 个真实模型任务的成功率。

## 真实评估命令

```powershell
py -3.14 -m agent.eval.runner --fixture demo-repo --output reports --runs 1
```

运行前确认 `.env` 中的模型和密钥。该命令会产生真实 API 调用，可能收费。输出写入：

- `reports/evaluation-report.json`
- `reports/evaluation-report.md`

## 发布评估数据的最低要求

1. 固定模型名称、温度、Prompt 版本和代码版本。
2. 每个任务至少运行 3 次，保留失败样本与原因。
3. 分开报告确定性断言和 LLM Judge，不混成一个分数。
4. 公开失败任务、运行次数、总 Token 和成本是否完整。
5. 简历只引用实际生成且可复现的报告数据。

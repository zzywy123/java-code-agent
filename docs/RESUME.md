# 简历与面试表述

## 简历项目描述

开发面向 Java 仓库的本地 Coding Agent：使用 LangGraph 编排 Supervisor、Researcher、Coder、Tester、Verifier，完成代码检索、Patch 审批、测试执行、Verifier 返工与会话恢复；实现方法级 Java 切片、BM25/向量混合检索、Agentic RAG、MCP stdio、SQLite Checkpoint、长期记忆和角色权限隔离；通过 AppService 统一 CLI 与 Streamlit，并建设 Trace、Token、工具指标、隔离式评估和非 root Docker 交付。

## 可量化内容

当前只能写入仓库测试和确定性验收结果，不能声称模型任务成功率。真实模型评估完成后，从 `reports/evaluation-report.json` 引用：

- 固定任务完成率与 Patch 成功率。
- 编译/测试通过率和引用正确率。
- 平均迭代次数、Token、耗时和已知成本。
- 模型、温度、Prompt 版本、代码版本与运行次数。

## 面试讲解重点

1. Agent 与问答 Demo 的区别：模型能经工具读取、修改、测试并根据结果继续行动。
2. 为什么保留人工审批：自然语言意图不等于写权限，副作用必须有明确边界。
3. 为什么 AppService 位于 UI 和 LangGraph 之间：统一事件、Session、审批恢复和可观测性，避免两个入口行为分叉。
4. 为什么评估使用临时 Git 仓库：任务可重复、Diff 可断言、原始 fixture 不受污染。
5. 如何处理 RAG 降级：向量模型不可用时退到 BM25，证据不足要明确标记而非编造。
6. Multi-Agent 的代价：更强的职责与权限边界换来更多延迟和状态复杂度，因此设置返工和步骤上限。

## 不应声称

- 未运行真实评估时，不写“任务成功率 90%”。
- 未执行 Docker build 时，不写“镜像构建通过”。
- 单元测试通过不等于真实 LLM 稳定性通过。
- 未配置模型价格时，不给出伪精确成本。

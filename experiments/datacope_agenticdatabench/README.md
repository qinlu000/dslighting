# DataCOPE × DSLighting AgenticDataBench

这个实验复用已有的 `experiments/agenticdatabench_poc` sidecar，把
AgenticDataBench 接入 DataCOPE 的迭代式 skill discovery：

```text
AgenticDataBench public task
  → DSLighting ReAct trajectories
  → DataCOPE AgreementVerifier + native SkillManager
  → benchmark-level SKILL.md
  → held-out ReAct control / skill
  → AgenticDataBench official evaluator
```

不修改 DSLighting 核心 registry。AgenticDataBench 继续拥有任务 JSONL、domain 数据、
多文件输出契约和官方评分；DSLighting 拥有 ReAct、离线 Docker sandbox、workspace 与
telemetry；DataCOPE 只负责无监督 trajectory verification 和 skill 写作。

最终 held-out 结果统一记录在
[experiments/results/datacope_ablation.md](../results/datacope_ablation.md)。

## 固定协议

| 项目 | 设置 |
| --- | --- |
| Split | public `dev.jsonl` 确定性 25% explore / 75% test |
| Explore trajectory | 每任务每轮 3 条 |
| Discovery | 3 轮（create 1 次，modify 2 次） |
| Explore / test temperature | 1.0 / 0.0 |
| Solver | DSLighting ReAct，30 steps |
| Solver model | `openai/DeepSeek-V4-Flash` |
| Skill writer | DataCOPE SkillManager + Codex CLI + `gpt-5.5` |
| Skill | 一个 benchmark-level AgenticDataBench skill |
| Task concurrency | 50 |
| Task / SkillManager timeout | 2 小时 |
| Sandbox | AgenticDataBench Docker image，network disabled |

全项目默认 `thinking=False`。Codex SkillManager 仍使用已配置的 Codex reasoning effort。

## SkillManager prompt 边界

SkillManager 使用 DataCOPE 原生 `AgreementVerifier` 生成的 prompt，并调用原生
`create_skill_without_category` / `modify_skill_without_category`。本实验只追加一条输出契约：

```text
The final AgenticDataBench Skill file must be saved at exactly `<skill-dir>/SKILL.md`.
```

不会替换 DataCOPE 的 discovery prompt，也不会把官方 score、gold、`eval_func` 或 benchmark
skill label 交给 SkillManager。

每次 rollout 导出：

- 完整 ReAct conversation；
- question 和 domain；
- 必需输出文件是否存在；
- 文件名、大小和 SHA-256；
- 整个多文件 artifact manifest 的稳定签名。

输出内容只用于可观察 self-consistency，不调用 grader。官方 evaluator 只在 held-out test
的 control/skill 两组运行结束后调用。

## 前置条件

按照 [AgenticDataBench PoC](../agenticdatabench_poc/README.md) 准备：

1. AgenticDataBench 官方 checkout 及 `testbed/datasets/`；
2. 上游 `da-agent` 基础镜像和 DSLighting AgenticDataBench 镜像；
3. DataCOPE checkout、Codex ChatGPT 登录和模型 API；
4. 官方 evaluator 所需 Python 依赖。

## 一条命令

```bash
AGENTICDATABENCH_ROOT=/path/to/AgenticDataBench \
AGENTICDATABENCH_DATASET_ROOT=/path/to/AgenticDataBench/testbed/datasets \
AGENTICDATABENCH_DOCKER_IMAGE=dslighting-agenticdatabench:latest \
DATACOPE_ROOT=/path/to/datacope/general \
DATACOPE_PY=/path/to/datacope/venv/bin/python \
bash experiments/datacope_agenticdatabench/run_experiment.sh
```

如果官方 evaluator 需要独立环境：

```bash
EVALUATOR_PY=/path/to/agenticdatabench/venv/bin/python \
AGENTICDATABENCH_ROOT=/path/to/AgenticDataBench \
AGENTICDATABENCH_DATASET_ROOT=/path/to/AgenticDataBench/testbed/datasets \
AGENTICDATABENCH_DOCKER_IMAGE=dslighting-agenticdatabench:latest \
DATACOPE_ROOT=/path/to/datacope/general \
DATACOPE_PY=/path/to/datacope/venv/bin/python \
bash experiments/datacope_agenticdatabench/run_experiment.sh
```

脚本复用仓库中冻结的 `split.json`，并拒绝复用已有 `runs/`，避免不同实验混合。每个 solver output 都保留
上游兼容的：

```text
<run-output>/
  agenticdatabench_tasks.jsonl
  <task-id>/
    <required outputs>
    dabench/result.json
  dslighting_run_summary.json
  datacope_run_manifest.json
```

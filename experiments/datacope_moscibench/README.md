# DataCOPE × DSLighting MoSciBench

这个实验直接使用 DSLighting 已有的 MoSciBench benchmark 和 ReAct workflow：

```text
DSLighting MoSciBench → DSLighting ReAct → DataCOPE skill → DSLighting ReAct
```

不调用 MoSciBench 官方仓库中的 agent 代码，也不修改 DSLighting 的 benchmark 或 ReAct
实现。Skill 通过现有的 `agent_runtime.skill_path` 注入。

## 固定设置

| 项目 | 设置 |
| --- | --- |
| 数据 | `data/releases/moscibench_full/moscibench/competitions` |
| Split | 六个 family 内分别确定性 25% explore / 75% test |
| 每个 explore task、每轮 trajectory | 3 |
| Discovery rounds | 3（create 1 次，modify 2 次） |
| Explore temperature | 1.0 |
| Test temperature | 0.0 |
| Solver | DSLighting ReAct，10 steps |
| Solver model | `openai/DeepSeek-V4-Flash` |
| Skill writer | DataCOPE SkillManager + Codex CLI + `gpt-5.5` |
| Skill 数量 | 一个 benchmark-level MoSciBench skill |
| 并发 | CPU mode，task 10，LLM 10，单 key 10 |
| 每个 sandbox timeout | 2 小时 |

Skill Manager 只读取 explore task 的公开描述、`prepared/public` 数据、ReAct 对话和
submission，不读取 grader、private answer 或 benchmark score。

## 一条命令运行

```bash
bash experiments/datacope_moscibench/run_experiment.sh
```

脚本依次完成：冻结 split、三轮 explore/discovery、held-out baseline/skill 测试，以及
`runs/test_summary.json` 汇总。已有 `runs/` 或 `split.json` 时会停止，避免混合实验。

如果数据位于其他位置：

```bash
DATA_ROOT=/path/to/moscibench/competitions \
bash experiments/datacope_moscibench/run_experiment.sh
```

## 分步检查

```bash
PY="$PWD/.venv/bin/python"

"$PY" -m experiments.datacope_moscibench make-split
"$PY" -m experiments.datacope_moscibench prepare-data \
  --output-dir experiments/datacope_moscibench/runs/public_data

"$PY" -m experiments.datacope_moscibench run \
  --phase explore --round-index 0 --sample-index 0
```

一次 `run` 会把该 phase 的全部任务交给 DSLighting scheduler 并发执行，不再按 dataset
分别启动官方 MoSciBench workflow。

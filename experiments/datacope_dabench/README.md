# DataCOPE × DABench

这个实验把 DataCOPE 论文中 reasoning task 的 discovery 协议移植到 DABench：

完整实验只需要一条命令：

```bash
bash experiments/datacope_dabench/run_experiment.sh
```

该脚本从空的 `runs/` 开始，依次完成三轮 explore、trajectory export、三次 skill
create/modify，以及最终 baseline/skill held-out test。

| 项目 | 固定设置 |
| --- | --- |
| Explore split | 64 tasks / 13 data families |
| Held-out test split | 193 tasks / 38 data families |
| 每个 task、每轮轨迹数 | 10 |
| Discovery rounds | 3（create 1 次，modify 2 次） |
| Explore temperature | 1.0 |
| Test temperature | 0.0 |
| ReAct 最大步数 | 10 |
| Explore/test solver | DSLighting ReAct + `openai/DeepSeek-V4-Flash` |
| Skill writer | DataCOPE SkillManager + Codex CLI + `gpt-5.5` |

因此，一次 `sample-index` 会运行完整的 64 个 explore tasks，产生 64 条 trajectory；
每轮 10 次是 640 条，三轮合计 `64 × 10 × 3 = 1,920` 条。`sample-index` 不是
skill iteration。DataCOPE 原生参数语义下，这三轮等价于 `iterations=2`，因为 initial
round 不计入额外 iteration 数。

第 0 轮不注入 skill；第 1、2 轮分别注入上一轮生成的 skill。最终 test 只运行一次
deterministic baseline 和一次 deterministic skill 条件。DABench 没有沿用 DABStep 的类别，
所以这里始终生成一个 benchmark-level skill。

论文主设置明确给出 10 trajectories、3 rounds 和 1.0/0.0 温度；发布代码与论文 cost
设置的 ReAct turn 上限为 15；本实验按首版预算将 DSLighting ReAct 固定为 10 steps。
发布代码还设置了每次 LLM completion 最多 16384 tokens，而 DSLighting 当前没有等价的
实验级 completion-output 参数；首版没有为此侵入 LLM 核心。

## 隔离边界

- family-disjoint split 固定在 `split.py`，同时锁定源 manifest 的 SHA-256。
- ReAct 使用 bubblewrap、环境变量 allowlist、禁网 sandbox；不支持时直接失败。
- 导出的 trajectory 不包含 grader score、ground truth 或 private answer。
- DataCOPE 只读取 explore predictions、explore-only public data、历史 skill 和必要运行时。
- previous skill 以只读目录挂载，复制到新 round 后再修改；历史快照不会原地改变。
- baseline 与 skill test 唯一实验变量是 `agent_runtime.skill_path`。

## 依赖

当前固定 DataMind commit：

```text
0f654e1845676ea2b911c915e0cfc9b3e1a51e1a
```

仓库 patch 只更新 DataCOPE 原有 Codex runtime，不注册新的 DSLighting agent：

```bash
git -C /data/caoqinlu/projects/DataMind checkout 0f654e1845676ea2b911c915e0cfc9b3e1a51e1a
git -C /data/caoqinlu/projects/DataMind apply --check \
  "$PWD/experiments/datacope_dabench/patches/datacope-openai-codex.patch"
git -C /data/caoqinlu/projects/DataMind apply \
  "$PWD/experiments/datacope_dabench/patches/datacope-openai-codex.patch"

codex --version
codex login status
```

本次固定 `codex-cli 0.144.5`，reasoning effort 为 `xhigh`（Extra High）。Codex 使用现有 ChatGPT
登录，不读取 `.env` 中的 HKUST endpoint。开始正式运行前先检查空间：

```bash
df -h .
```

实验会保留 30 个完整 explore workspace；不要在空间不足时启动。

## 运行辅助函数

以下命令使用项目自己的 Python，不要改成裸 `python`：

```bash
PY="$PWD/.venv/bin/python"
DATACOPE_ROOT=/data/caoqinlu/projects/DataMind/datacope/general
DATACOPE_PY=/data/caoqinlu/projects/DataMind/.venv/bin/python
RUNS="$PWD/experiments/datacope_dabench/runs"

run_explore_round() {
  local round="$1"
  local skill="${2:-}"
  local sample sample_tag run_name
  local -a run_args

  for sample in $(seq 0 9); do
    sample_tag=$(printf '%02d' "$sample")
    run_name="dabench_datacope_explore_round_${round}_sample_${sample_tag}"

    run_args=(
      -m experiments.datacope_dabench run
      --phase explore
      --round-index "$round"
      --sample-index "$sample"
    )
    if [ -n "$skill" ]; then
      run_args+=(--skill-file "$skill")
    fi
    "$PY" "${run_args[@]}"

    "$PY" -m experiments.datacope_dabench export \
      --workspace-root "$RUNS/workspaces/$run_name" \
      --output-dir "$RUNS/predictions/round_$round/run_$sample_tag"
  done
}
```

这些函数使用 Bash array，应在 Bash 中执行。所有输出目录都要求为新目录或空目录；失败后
应检查并保留现场，不要把半轮数据混入下一次运行。

## Round 0：无 skill explore + create

```bash
run_explore_round 0

"$PY" -m experiments.datacope_dabench prepare-data \
  --source-root "$PWD/data/releases/dabench_perception_clean_v1" \
  --predictions-dir "$RUNS/predictions/round_0" \
  --output-dir "$RUNS/public_data"

"$PY" -m experiments.datacope_dabench discover \
  --python "$DATACOPE_PY" \
  --datacope-root "$DATACOPE_ROOT" \
  --round-index 0 \
  --predictions-dir "$RUNS/predictions/round_0" \
  --verified-dir "$RUNS/verified/round_0" \
  --data-dir "$RUNS/public_data" \
  --skill-dir "$RUNS/skills/round_0/dabench" \
  --model gpt-5.5
```

可在 discover 前做一次不调用模型的 agreement 检查：

```bash
"$PY" -m experiments.datacope_dabench verify \
  --python "$DATACOPE_PY" \
  --datacope-root "$DATACOPE_ROOT" \
  --predictions-dir "$RUNS/predictions/round_0" \
  --verified-dir "$RUNS/verified_check_round_0"
```

## Round 1：注入 round 0 skill + modify

```bash
run_explore_round 1 "$RUNS/skills/round_0/dabench/SKILL.md"

"$PY" -m experiments.datacope_dabench discover \
  --python "$DATACOPE_PY" \
  --datacope-root "$DATACOPE_ROOT" \
  --round-index 1 \
  --previous-predictions-dir "$RUNS/predictions/round_0" \
  --predictions-dir "$RUNS/predictions/round_1" \
  --previous-skill-dir "$RUNS/skills/round_0/dabench" \
  --verified-dir "$RUNS/verified/round_1" \
  --data-dir "$RUNS/public_data" \
  --skill-dir "$RUNS/skills/round_1/dabench" \
  --model gpt-5.5
```

## Round 2：注入 round 1 skill + modify

历史 prediction 参数必须按时间顺序重复两次：

```bash
run_explore_round 2 "$RUNS/skills/round_1/dabench/SKILL.md"

"$PY" -m experiments.datacope_dabench discover \
  --python "$DATACOPE_PY" \
  --datacope-root "$DATACOPE_ROOT" \
  --round-index 2 \
  --previous-predictions-dir "$RUNS/predictions/round_0" \
  --previous-predictions-dir "$RUNS/predictions/round_1" \
  --predictions-dir "$RUNS/predictions/round_2" \
  --previous-skill-dir "$RUNS/skills/round_1/dabench" \
  --verified-dir "$RUNS/verified/round_2" \
  --data-dir "$RUNS/public_data" \
  --skill-dir "$RUNS/skills/round_2/dabench" \
  --model gpt-5.5
```

每次 discover 都会强制检查当前及历史 round：恰好 10 个 sample directories、每个 sample
拥有完全相同的任务集合，而且该集合恰好等于冻结的 64 个 explore tasks。第 0 轮调用
`init_run + create_skill_without_category`，后两轮调用
`iterate_run + modify_skill_without_category`。

## Held-out test

```bash
"$PY" -m experiments.datacope_dabench run --phase test

"$PY" -m experiments.datacope_dabench run \
  --phase test \
  --skill-file "$RUNS/skills/round_2/dabench/SKILL.md"
```

两组都覆盖同样的 193 个 held-out tasks，temperature 为 0.0，最大 10 步。最终比较
`dabench_datacope_test_baseline` 与 `dabench_datacope_test_skill` 的 benchmark 结果。

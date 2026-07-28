# mini-swe-agent 接入说明

DSLighting 将 mini-swe-agent v2 作为可选工作流接入。mini-swe-agent 负责模型调用、
agent 循环和命令环境；本地与 Docker 模式分别直接使用官方 `LocalEnvironment` 和
`DockerEnvironment`。DSLighting 继续负责数据准备、Workspace、输出收集、评分和运行遥测。

## 安装

从 PyPI 安装：

```bash
pip install 'dslighting[mini-swe-agent]'
```

在源码仓库中开发：

```bash
pip install -e '.[mini-swe-agent]'
```

当前适配层支持 `mini-swe-agent>=2.4.6,<3.0.0`，并在运行时拒绝不兼容的 v1。

## Python API

```python
from dotenv import load_dotenv
load_dotenv()

from dslighting.api import Agent

agent = Agent(
    workflow="mini_swe_agent",
    model="openai/gpt-4o",
    sandbox_backend="local",
    mini_swe_agent={
        "step_limit": 20,
        "cost_limit": 3.0,
        "wall_time_limit_seconds": 1800,
        "command_timeout": 120,
    },
)

result = agent.run(task_id="bike-sharing-demand")
print(result.success, result.score, result.cost)
```

`mini-swe-agent`、`minisweagent` 和 `mini` 也可作为 `Agent` API 的别名。YAML 和
底层配置统一使用规范名 `mini_swe_agent`。

## YAML 配置

```yaml
workflow:
  name: mini_swe_agent
  params:
    step_limit: 20
    cost_limit: 3.0
    wall_time_limit_seconds: 1800
    command_timeout: 120

llm:
  model: openai/gpt-4o
  api_key: your_key

sandbox:
  backend: local
  timeout: 1800
```

`local` 直接在宿主机执行命令，只适合受信任的调试任务。正式 benchmark 应使用下面的
Docker 配置。

## 官方 Docker 环境

将标准 DSLighting Sandbox 配置切换为 Docker 即可，不需要再配置一套
mini-swe-agent 专属参数：

```yaml
workflow:
  name: mini_swe_agent

sandbox:
  backend: docker
  docker_image: your-benchmark-image@sha256:...
  docker_workspace_path: /workspace
  network_policy: disabled
  timeout: 1800
  memory_mb: 8192
  cpu_cores: 4
  pids_limit: 256
```

此模式直接调用 mini-swe-agent 官方 `DockerEnvironment`，每个 bash action 由
`bash -lc` 在任务容器中执行。DSLighting 将 task workspace 读写挂载到容器，并将
输入 symlink 的解析目标冻结为只读挂载。模型调用仍在宿主进程中进行，模型 API key
不会传进 benchmark 容器。

容器镜像应由 benchmark 决定。若 benchmark 提供官方运行镜像，优先使用该镜像或从
它派生一个只补充缺失运行依赖、非 root 用户和输出权限的薄镜像；正式评测应固定镜像
digest。这样不同 agent 使用相同依赖环境，避免把 DSLighting 的通用 Python 环境误当成
benchmark 环境。

## 参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `step_limit` | `agent_runtime.max_steps` | 最大模型调用步数；`0` 表示不限 |
| `cost_limit` | `0.0` | mini-swe-agent 费用上限；`0` 表示不限 |
| `wall_time_limit_seconds` | `sandbox.timeout` | 整个 agent 的墙钟时间上限 |
| `command_timeout` | `30` | 单条 shell 动作的执行超时 |
| `workspace_base_dir` | DSLighting 默认值 | 运行 Workspace 根目录 |
| `config_path` | mini-swe-agent `mini.yaml` | 自定义 mini-swe-agent YAML 配置（读取 `agent` / `model` 部分） |
| `agent_overrides` | `{}` | 覆盖 mini-swe-agent agent 模板等高级设置 |
| `model_overrides` | `{}` | 覆盖 mini-swe-agent model 设置 |

`output_path`、各类 limit 和 `model_name` 由 DSLighting 统一管理，不能放进 override。

## 运行行为

- `sandbox.backend=docker` 使用官方 `DockerEnvironment`；镜像、资源、网络和
  workspace 挂载来自标准 `sandbox` 配置。
- `sandbox.backend=local` 使用官方 `LocalEnvironment`，命令直接在宿主机执行，
  不提供容器隔离。
- 当前 DSLighting 适配层只支持这两个官方环境；`e2b` 和 `ds_sandbox` 会在 workflow
  创建时明确报错。
- 输入文件由 DSLighting 链接到 task sandbox 的当前工作目录。
- agent 必须在当前目录创建任务要求的精确输出文件名；未创建时运行会明确失败。
- 完整轨迹保存为 Workspace 内的
  `artifacts/minisweagent_trajectory.json`。
- mini-swe-agent 的费用、调用次数和 token 使用量会映射到 DSLighting 结果与遥测。
- 写入轨迹和遥测前会递归脱敏 API key、token、secret 等字段。
- 若配置多个 API key，当前 mini-swe-agent 工作流使用第一个 key；其模型调用暂不使用
  DSLighting 的 key-pool 轮换。

官方资料：

- [mini-swe-agent Python bindings](https://mini-swe-agent.com/latest/usage/python_bindings/)
- [mini-swe-agent API reference](https://mini-swe-agent.com/latest/reference/)

# 在自己的 Isaac Lab 项目里用 SuperMimic

可以把它当成一个独立学习包来用：机器人、场景和任务留在自己的项目里，
SuperMimic Lab 负责学习和检查点。智元远征 A3 是一个调用示例，不需要先安装 A3 包才能训练
自己的机器人。英文接口说明和可复制配置见 [CUSTOM_TASKS.md](CUSTOM_TASKS.md)。

## 先选接入方式

| 你的任务 | 建议做法 |
| --- | --- |
| 换一个机器人，仍然训练参考动作模仿 | 复用 `TrackingEnvironment`，提供自己的模型、PD 和动作数据 |
| 已有 Isaac Lab 任务，奖励、观测或控制目标明显不同 | 保留环境，实现 `DifferentiableTask` 和真实转移采集接口 |

第一条路线需要的工作少一些。第二条路线需要把任务逻辑写成能在想象轨迹上求导的
形式；仅有 PPO 的 observation 和 reward 标量还不够。两条路线都不用把机器人名字
或关节数写进学习算法。

## 安装与注册

在现有 Isaac Lab 环境里安装通用包即可。用 Docker 时，在容器里操作，使用该环境的
Python；Isaac Sim 容器通常是 `/isaac-sim/python.sh`。无需为了本包重建仿真环境。

```bash
python -m pip install --no-deps --no-build-isolation /path/to/supermimic-lab
python -m pip install --no-deps --no-build-isolation /path/to/my-project
```

自己的任务包通过 Gym 注册环境、配置和可选评估器。命令中的
`my_tasks:MyRobot-Tracking-v0` 会先导入 `my_tasks`，再寻找任务 ID。
已有任务注册方式可以继续保留；需要 Isaac Lab 自动发现时，再声明
`isaaclab.tasks` entry point，示例见 A3 的 [pyproject.toml](../examples/a3/pyproject.toml)。

## 复用参考动作任务

准备模型 URDF、对应后端的 USD、电机参数 JSON 和自己的参考动作。关节顺序必须
一致，根姿态四元数使用 XYZW。动作文件每行是根位置 3 维、根四元数 4 维，再加
全部活动关节角；列数是 `7 + 关节数`，不会固定成 A3 的 38 列。

继承 `TrackingEnvironmentConfig`，在 `configure_assets()` 里组装：

- `profile`：根 link、躯干 link、关节顺序、默认角、PD、armature、限幅和摩擦。
- `urdf_path`、`usd_path`：做可微运动学和真实仿真的两个模型，几何必须一致。
- `tracked_links`：参与动作跟踪的身体部位；`collision_exclusions`：确有需要排除的碰撞对。
- `action_scale_effort_fraction`：策略输出换算成关节位置目标的比例。
- `asset_sha256`：实际文件摘要，用于阻止不兼容的检查点加载。

然后调用通用 `configure_robot()`，并将 Gym 环境入口指向
`supermimic_lab.isaaclab.environment:TrackingEnvironment`。
英文指南提供了完整配置片段；A3 的
[configuration.py](../examples/a3/src/supermimic_a3/configuration.py)展示了实际调用。
新的 URDF 需要是一棵单根树，当前运动学实现支持 fixed 和有界 revolute 关节。

```bash
supermimic-train --task my_tasks:MyRobot-Tracking-v0 \
  --asset-root /data/my-robot --motion-dir /data/my-motions \
  --physics newton_mjwarp --num-envs 128 --hours 1 \
  --eval-every 1000 --output runs/my_robot_01
```

先用少量环境核对关节顺序、位姿、速度和 PD，再根据显存扩大环境数。选择 PhysX 时，
改为 `--physics isaacsim_physx` 并提供匹配的 USD。切换后端会改变任务签名，使用新运行目录。

## 接入已有任务

自己的环境仍然是正常 Isaac Lab/Gym 环境，同时提供 `model_task` 和真实转移。
`model_task` 定义世界模型的输入/输出、状态积分、actor 观测、目标、奖励、终止概率和
critic 输入。完整方法表见 [DifferentiableTask](../src/supermimic_lab/contracts.py)。

最容易出错的是 reset 顺序：必须先记录终止前的 `next_state`、奖励和失败标志，再
重置环境。episode 时间到与真实失败分开，replay 不能跨过 reset 拼接连续窗口。
`reference_index` 可以是动作帧，也可以是稳定的任务指令索引；任务要保证回放期间
这个索引指向的数据仍然有效。

Runner 的用法与常见 RL 学习器接近：

```python
from supermimic_lab.config import RunnerConfig
from supermimic_lab.isaaclab.environment import TransitionAdapter
from supermimic_lab.runner import SuperMimicRunner

env.reset()
runner = SuperMimicRunner(TransitionAdapter(env), RunnerConfig(), "runs/my_task_01")
runner.learn(num_learning_iterations=100)
policy = runner.get_inference_policy()
```

这里的 `env` 必须已经实现上述接口。奖励改动、观测改动和任务指令约定也应纳入
自定义 `signature`，避免误用旧权重。actor 可用信息与世界模型/critic 的训练真值要
明确分开。碰撞是否支持某种几何、摩擦是否生效，仍由相应仿真后端决定。

## 留下可以复现的运行

统一 CLI 会记录配置和数据签名。需要连自己的任务源码一起冻结时，可以使用：

```bash
python /path/to/supermimic-lab/scripts/supervise_training.py \
  --source /path/to/supermimic-lab --task-source /path/to/my-project/src \
  --output runs/my_robot_01 --process-timeout 5400 -- \
  --task my_tasks:MyRobot-Tracking-v0 \
  --asset-root /data/my-robot --motion-dir /data/my-motions \
  --physics newton_mjwarp --num-envs 128 --hours 1
```

该入口保留 Isaac Lab 启动器提供的依赖路径，只把冻结源码加入子进程导入路径。
模型和数据仍是外部文件，需自行固定版本并保留摘要。自定义任务若要定期评估，需要
注册自己的 evaluator；A3 的“完整片段完成率”不一定适合你的任务。

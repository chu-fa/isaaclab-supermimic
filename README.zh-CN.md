# SuperMimic Lab

[English](README.md)

SuperMimic Lab 是基于 Isaac Lab 的 SuperMimic 训练库，主要用于个人多个运控项目的开发。
项目提供可跨项目复用的训练接口，并包含智元远征 A3 机器人训练示例。

**通用接口和 A3 示例位于同一个仓库。** 两个 Python 包可按需安装：

| 包 | 用途 | 位置 |
| --- | --- | --- |
| `supermimic-lab` | 算法、Runner、Isaac Lab 适配和自定义任务契约 | `src/supermimic_lab/` |
| `supermimic-lab-a3` | 调用通用接口的可选 A3 任务 | `examples/a3/` |

A3 示例单向依赖通用包，其他项目可以只使用通用包与自己的任务。项目依据
[SuperMimic 论文](https://arxiv.org/abs/2610.28480)实现，属于非官方实现，未经论文
作者或 Isaac Lab 维护者背书。

## 实现范围

- 已实现自回归世界模型、想象轨迹策略优化、critic、回放、检查点和评估流程。
- 自定义任务使用 Isaac Lab 配置、Gym 注册和 `SuperMimicRunner` 接口。
- Newton/MuJoCo-Warp 和 PhysX 适配均通过有限 A3 接口检查。
- A3 示例提供训练命令、数据格式和评估定义；训练数据与演示视频后续单独发布。

多接口联合训练、速度指令控制和真机部署尚未实现；收敛、后端等价性及论文级
效果尚未确立。详见[实现范围](docs/IMPLEMENTATION.md)与[验证记录](docs/VALIDATION.md)。

## 安装

使用兼容 Isaac Lab 环境的 Python 启动器。当前验证对象是 **Isaac Lab 3 开发接口**，
不保证兼容 Isaac Lab 2。Isaac Lab、仿真后端和匹配的 PyTorch 等依赖需要预先配置，
具体要求见[安装说明](docs/SETUP.md)。

在本仓库根目录执行：

```bash
# 通用接口。
python -m pip install --no-deps --no-build-isolation .

# 同一仓库中的可选 A3 示例。
python -m pip install --no-deps --no-build-isolation ./examples/a3

supermimic-train --help
```

Isaac Sim 容器可使用 `/isaac-sim/python.sh` 替代 `python`。机器人资产和动作数据为
外部输入，不随包安装。

## 使用方式

自定义任务注册环境、配置和评估器后，可使用统一入口：

```bash
supermimic-train \
  --task my_tasks:MyProject-SuperMimic-Task-v0 \
  --env-config /path/to/task.json \
  --config /path/to/learner.json \
  --physics newton_mjwarp --num-envs 2048 --hours 3 \
  --output runs/my_task_01
```

任务还需提供想象轨迹所需的可微状态、观测、奖励和终止函数；已有 PPO 的观测
张量不能直接替代这些接口。接入步骤见[中文指南](docs/INTEGRATION.zh-CN.md)，完整接口见[自定义任务指南](docs/CUSTOM_TASKS.md)。

## 智元远征 A3 应用测试

[A3 示例文档](examples/a3/README.zh-CN.md)包含训练、评估、可视化调用和
2048 环境配置。参考动作格式、来源与可用性见[数据说明](examples/a3/DATA.md)，
评估判据与指标见[评估说明](examples/a3/EVALUATION.md)。

当前 A3 默认使用官方原版，提供固定版本下载和 USD 转换脚本，以及示例电机参数。
动作 CSV 仍需另行准备，不附带预训练权重。模型、PD 和训练过程见
[A3 训练说明](examples/a3/TRAINING.zh-CN.md)。本次发布代码、配置和使用文档，
训练记录、模型权重和演示视频暂不包含在内。

## 文档与贡献

- [架构与目录](docs/ARCHITECTURE.md)
- [安装说明](docs/SETUP.md)
- [实现范围与差异](docs/IMPLEMENTATION.md)
- [贡献指南](CONTRIBUTING.md)
- [来源说明](PROVENANCE.md)

代码使用 [MIT 许可](LICENSE)，外部依赖和数据保留各自许可。使用该方法时请引用
SuperMimic 论文，引用信息见 [CITATION.cff](CITATION.cff)。

# 智元远征 A3 参考动作示例

[English](README.md) · [项目入口](../../README.zh-CN.md) ·
[数据说明](DATA.md) · [评估说明](EVALUATION.md)

这个例子用通用 SuperMimic Lab 接口训练智元远征 A3，让机器人跟踪准备好的参考动作。
算法包和 A3 例子放在同一个仓库，方便直接跑通后再换成自己的机器人。
模型来源、PD 参数和训练过程见[训练说明](TRAINING.zh-CN.md)。

可选包 `supermimic-lab-a3` 负责 A3 配置和任务注册；算法、Runner、跟踪环境和
评估器由 `supermimic-lab` 提供。其他项目可以只使用通用包和自己的任务。

## 准备

需要已有的兼容 Isaac Lab 环境。当前默认用智元远征 A3 的官方原版模型，提供固定版本
下载与转换脚本；电机参数随示例提供，动作 CSV 仍需另外准备，详见[数据说明](DATA.md)。
单张 RTX 3090 的示例配置使用 2048 个环境，可以按自己的显存调整。

```bash
python scripts/fetch_a3_assets.py --output /data/a3_official
python scripts/convert_a3_assets.py --asset-root /data/a3_official --visualizer none
```

下载的 URDF 和 meshes 保持原样，转换结果单独保存。两个命令都不会安装依赖。
使用已有 Docker 时，在容器内用它已有的 Python 执行即可。

在仓库根目录，使用 Isaac Lab 环境的 Python 安装：

```bash
python -m pip install --no-deps --no-build-isolation . ./examples/a3
supermimic-train --task supermimic_a3:SuperMimic-Lab-Tracking-A3-v0 --help
```

下方源码 demo 也支持不安装包直接运行。Isaac Sim 容器使用 `/isaac-sim/python.sh`
代替 `python`，数据路径应为容器内可见路径。版本要求见[安装说明](../../docs/SETUP.md)。

## 训练

```bash
python scripts/a3_demo.py train \
  --asset-root /data/a3_official \
  --motion-dir /data/lafan1_a3 \
  --physics newton_mjwarp --num-envs 2048 --hours 3 \
  --stop-on-plateau --output runs/a3_demo_01 --dry-run
```

去掉 `--dry-run` 后开始训练。[配置](../../configs/a3_rtx3090.json)使用完整网络宽度、
模型/策略批次各 512、8 步世界模型窗口、32 步想象轨迹、seed 0 和名义物理参数。
未提供 `--checkpoint` 时，网络与优化器从零初始化。

训练冻结源码和配置，并记录版本、摘要、指标和检查点；初始及每 1000 次更新评估
全部动作。训练预算为预热和初始评估之后的 3 小时，监督进程另设 3.5 小时总超时及
90 秒退出宽限。Ctrl+C 请求在更新边界保存，并取消最终自动评估；输出目录须为新目录。

`--stop-on-plateau` 可在至少 1 小时后、连续三次完整评估满足稳定条件时停止，
平台期不代表收敛。PhysX 改用 `--physics isaacsim_physx`、对应 USD 和新输出目录。
两种后端的有限接口检查入口见[验证说明](../../docs/VALIDATION.md)。

## 评估与可视化

```bash
python scripts/a3_demo.py evaluate \
  --asset-root /data/a3_official \
  --motion-dir /data/lafan1_a3 \
  --physics newton_mjwarp --num-envs 40 \
  --checkpoint runs/a3_demo_01/last.pt \
  --output runs/a3_demo_01/evaluation.json
```

检查点必须已经生成，并与机器人、动作库和后端匹配。评估从各片段首帧开始，
关闭策略噪声和域随机化。失败判据和指标定义见[评估说明](EVALUATION.md)。

可视化将 `evaluate` 换为 `play`，加 `--num-envs 1 --clips 0`；Newton 使用
`--visualizer newton`，PhysX 使用 `--visualizer kit`。`--max-steps` 可限制预览长度，
但截断不计为完成。交互渲染和视频录制尚未验证。

## 使用边界

本次发布代码、示例配置和使用文档，训练数据、模型权重和演示视频后续单独发布。
更换模型时应使用独立运行目录并重新评估。完成片段只表示没有触发失败判据，
不能单独证明跟踪精度、收敛或泛化。评估器默认使用传入的动作库，不会自动建立
未见动作划分。

A3 本体、参数、动作重定向及名义物理设置与论文基准不同。完整数据准备流程不在
当前发布范围，多接口联合训练也尚未实现，详见[实现说明](../../docs/IMPLEMENTATION.md)。

## 训练数据和演示视频

运行结束后，可用 `scripts/export_a3_results.py` 导出配置、耗时、更新次数和完整动作
评估数据。训练预算只计学习时间，不含周期评估；进程总耗时还包含周期评估、初始化和
最终评估，结束后才可确定。
命令见[英文示例](README.md#export-metrics-and-record-a-video)。

`scripts/record_a3_video.py` 使用已有 Isaac Sim 的 MP4 编码扩展与 Newton 渲染器，
录制预先指定的步行、舞蹈和跳跃片段，每段最多30秒。失败会停止该片段并注明结果，
不把自动 reset 之后的画面拼成连续成功。视频元数据保留检查点摘要、实际时长与结果。
当前只完成命令入口检查，实际渲染和编码待训练结束后验证；视频核对后再发布。
MP4 作为 Release 附件或外部媒体保存，报告中提供链接，不放进源码历史。

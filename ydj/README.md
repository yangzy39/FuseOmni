# GPU 永动机 (Perpetual Motion Machine)

在 SCO ACP 集群上保持 GPU 资源占用，防止因空闲被系统回收，同时支持远程提交和管理训练任务。

## 核心特性

- **动态 GPU 保活**：根据显存大小自动计算矩阵尺寸，保持 GPU 利用率 >70%
- **多实例隔离**：每个 JOB_ID 拥有独立的任务队列，互不干扰
- **远程任务管理**：无需登录节点即可提交、查看、停止任务
- **24小时自动关闭**：从启动开始计时，24小时后自动退出，防止资源浪费
- **进程清理**：gpu_stop 时自动杀掉除永动机外的所有 Python 进程

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    本地/任意机器                              │
│  export JOB_ID=train-exp1                                   │
│  source client.sh                                           │
│  gpu_submit train.sh   gpu_status   gpu_stop   gpu_logs     │
└─────────────────────────┬───────────────────────────────────┘
                          │ (通过共享存储 /mnt/afs)
┌─────────────────────────▼───────────────────────────────────┐
│              /mnt/afs/.../gpu_queue/                        │
│  ├── train-exp1/          ← JOB_ID=train-exp1 的队列        │
│  │   ├── pending/                                           │
│  │   ├── running/                                           │
│  │   ├── done/                                              │
│  │   ├── failed/                                            │
│  │   ├── logs/                                              │
│  │   └── control.json                                       │
│  ├── train-exp2/          ← JOB_ID=train-exp2 的队列        │
│  └── ...                                                    │
└─────────────────────────▲───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│                 SCO ACP 计算节点                              │
│  perpetual_motion.py --job-id train-exp1                    │
│  ├── 动态矩阵大小: 根据 GPU 显存自动计算                      │
│  ├── GPU Keep-Alive: 保持利用率 70-85%                       │
│  ├── Queue Monitor: 监控 pending/ 执行新任务                 │
│  └── Task Executor: 执行脚本并输出日志                       │
└─────────────────────────────────────────────────────────────┘
```

## 文件说明

| 文件 | 说明 |
|------|------|
| `perpetual_motion.py` | 永动机主程序，在计算节点上运行 |
| `client.sh` | 客户端工具，在任意机器上使用 |
| `submit.sh` | 提交永动机作业到 SCO ACP |

## 快速开始

### 1. 配置路径

编辑 `submit.sh`，修改以下路径：

```bash
CODE_DIR="/mnt/afs/00036/yzy/FuseOmni/ydj"   # 本项目路径
QUEUE_ROOT="/mnt/afs/00036/yzy/gpu_queue"     # 队列根目录
CONDA_PATH="/mnt/afs/00036/software/conda/bin/activate"
CONDA_ENV="swift"
```

### 2. 提交永动机

```bash
# 语法: ./submit.sh <JOB_ID> [GPU数量] [节点数量]

# 示例
./submit.sh train-exp1           # 2 GPU, 1 节点, 队列: gpu_queue/train-exp1/
./submit.sh train-exp2 4         # 4 GPU, 1 节点
./submit.sh train-exp3 8 2       # 8 GPU, 2 节点
```

### 3. 使用客户端

在任意可以访问共享存储的机器上：

```bash
# 指定要操作的永动机实例
export JOB_ID=train-exp1

# 加载客户端工具
source /mnt/afs/.../FuseOmni/ydj/client.sh

# 查看帮助
gpu_help
```

## 客户端命令

### 实例管理

```bash
# 查看当前实例
gpu_use

# 切换到其他实例
gpu_use train-exp2

# 列出所有实例
gpu_list
```

### 提交任务

```bash
# 提交脚本文件
gpu_submit /path/to/train.sh

# 提交内联命令
gpu_run 'python train.py --epochs 10'

# 指定优先级（数字越小越优先）
gpu_submit train.sh 001
```

### 查看状态

```bash
# 查看队列状态
gpu_status

# 查看日志列表
gpu_logs

# 查看特定任务日志
gpu_logs train

# 实时跟踪日志
gpu_tail
gpu_tail train
```

### 控制任务

```bash
# 停止特定任务
gpu_stop train.sh

# 清空所有待执行任务、停止当前任务、并杀掉所有其他 Python 进程
gpu_stop all

# 关闭永动机
gpu_shutdown
```

> **注意**: `gpu_stop` 会自动杀掉节点上除永动机以外的所有 Python 进程，确保资源完全释放。

## 动态矩阵大小

永动机会根据 GPU 显存自动计算最优矩阵大小：

```python
# 目标: 使用 60% 显存用于矩阵
# 矩阵乘法 C = A @ B 需要 3 个 N×N 矩阵
# 内存 = 3 * N² * 2 bytes (float16)
# N = sqrt(显存 * 0.6 / 6)

# 示例:
# 24GB GPU → 矩阵约 14336×14336
# 48GB GPU → 矩阵约 20480×20480
# 80GB GPU → 矩阵约 26624×26624
```

利用率控制：
- `< 50%`: 加速矩阵运算，提升利用率
- `50-70%`: 正常运算
- `70-85%`: 目标范围
- `> 85%`: 减慢运算，为真实任务让出资源

## 多实例使用场景

```bash
# 实验 1: 小规模测试
./submit.sh test-small 2

# 实验 2: 大规模训练  
./submit.sh train-large 8

# 实验 3: 推理服务
./submit.sh inference 4

# 客户端切换
export JOB_ID=test-small
source client.sh
gpu_submit test_script.sh

gpu_use train-large  # 切换到另一个实例
gpu_submit train_script.sh
```

## 任务脚本示例

### 训练脚本 `train.sh`

```bash
#!/bin/bash
set -e

# 激活环境
source /mnt/afs/00036/software/conda/bin/activate swift

# 切换到代码目录
cd /mnt/afs/00036/yzy/FuseOmni/ms-swift

# 运行训练
CUDA_VISIBLE_DEVICES=0,1 swift sft \
    --model /mnt/afs/00036/yzy/models/Qwen2.5-7B-Instruct \
    --dataset AI-ModelScope/alpaca-gpt4-data-zh \
    --output_dir /mnt/afs/00036/yzy/checkpoints/test
```

## SCO ACP 管理命令

```bash
# 查看作业列表
sco acp jobs list --workspace-name=share-space

# 实时监控（过滤用户名）
watch -n 0.5 "sco acp jobs list --workspace-name=share-space --page-size 500 | awk 'NR<=3 || /yzy/'"

# 删除作业 (作业名为 ydj-<JOB_ID>)
sco acp jobs delete --workspace-name=share-space ydj-train-exp1

# 登录节点（调试用）
Jobid=pt-xxxxxxxx
sco acp jobs exec --workspace-name=share-space \
  --worker-name=$(sco acp jobs get-workers --workspace-name=share-space $Jobid | grep "worker-0" | awk -F'|' '{print $2}' | xargs) \
  $Jobid
```

## 常见问题

### Q: 如何修改 GPU 保活的利用率阈值？

编辑 `perpetual_motion.py`：

```python
GPU_UTIL_THRESHOLD = 50  # 低于此值开始保活
GPU_UTIL_TARGET = 70     # 目标利用率
GPU_UTIL_HIGH = 85       # 高于此值减速
MEMORY_USAGE_TARGET = 0.6  # 显存使用比例
AUTO_SHUTDOWN_HOURS = 24   # 自动关闭时间（小时）
```

### Q: 如何修改自动关闭时间？

编辑 `perpetual_motion.py` 中的 `AUTO_SHUTDOWN_HOURS`：

```python
AUTO_SHUTDOWN_HOURS = 24  # 默认24小时，改为 48 则48小时后关闭
```

永动机启动时会打印自动关闭时间：
```
[INFO] Started at: 2025-01-21 16:00:00
[INFO] Auto-shutdown at: 2025-01-22 16:00:00 (24h)
```

### Q: gpu_stop 会杀掉哪些进程？

`gpu_stop all` 或 `gpu_stop <task>` 会：
1. 停止当前正在运行的任务
2. 杀掉节点上所有 Python 进程（**除永动机自身及其子进程外**）

这确保了训练进程完全退出，释放 GPU 显存。

### Q: 任务执行失败怎么查看原因？

```bash
# 查看失败任务
ls /mnt/afs/.../gpu_queue/<JOB_ID>/failed/

# 查看对应日志
gpu_logs <task_name>
```

### Q: 如何确保任务使用正确的 Conda 环境？

在任务脚本开头激活环境：

```bash
#!/bin/bash
source /mnt/afs/00036/software/conda/bin/activate your_env
# 后续命令...
```

### Q: 永动机意外退出怎么办？

重新提交即可，队列目录中的任务会保留：

```bash
./submit.sh <JOB_ID>  # 使用相同的 JOB_ID
```

### Q: 如何同时运行多个永动机？

每个永动机使用不同的 JOB_ID，它们的队列相互独立：

```bash
./submit.sh exp1 4   # 第一个实例
./submit.sh exp2 4   # 第二个实例
./submit.sh exp3 8   # 第三个实例
```

## License

MIT

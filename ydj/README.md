# GPU 永动机 (Perpetual Motion Machine)

在 SCO ACP 集群上保持 GPU 资源占用，防止因空闲被系统回收，同时支持远程提交和管理训练任务。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    本地/任意机器                              │
│  source client.sh                                           │
│  gpu_submit train.sh   gpu_status   gpu_stop   gpu_logs     │
└─────────────────────────┬───────────────────────────────────┘
                          │ (通过共享存储 /mnt/afs)
┌─────────────────────────▼───────────────────────────────────┐
│              /mnt/afs/00036/yzy/gpu_queue/                  │
│  ├── pending/    ← 待执行脚本                                │
│  ├── running/    ← 正在执行                                  │
│  ├── done/       ← 已完成                                    │
│  ├── failed/     ← 失败                                      │
│  ├── logs/       ← 日志                                      │
│  └── control.json ← 控制信号 (stop/shutdown)                 │
└─────────────────────────▲───────────────────────────────────┘
                          │
┌─────────────────────────┴───────────────────────────────────┐
│                 SCO ACP 计算节点                              │
│  perpetual_motion.py                                        │
│  ├── GPU Keep-Alive: 矩阵运算保持 GPU 利用率 > 30%           │
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

编辑 `submit.sh` 和 `client.sh`，修改以下路径：

```bash
# submit.sh
CODE_DIR="/mnt/afs/00036/yzy/FuseOmni/ydj"  # 本项目在共享存储的路径
QUEUE_BASE="/mnt/afs/00036/yzy/gpu_queue"    # 任务队列目录
CONDA_PATH="/mnt/afs/00036/software/conda/bin/activate"
CONDA_ENV="swift"

# client.sh
export QUEUE_BASE="/mnt/afs/00036/yzy/gpu_queue"
```

### 2. 提交永动机

```bash
# 默认 2 GPU
bash submit.sh

# 或指定 GPU 数量
bash submit.sh 4
```

### 3. 使用客户端

在任意可以访问共享存储的机器上：

```bash
# 加载客户端工具
source /mnt/afs/00036/yzy/FuseOmni/ydj/client.sh

# 查看帮助
gpu_help
```

## 客户端命令

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
gpu_stop 20250121_train.sh

# 清空所有待执行任务并停止当前任务
gpu_stop all

# 关闭永动机
gpu_shutdown
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

### Python 脚本 `train.py`

```python
#!/usr/bin/env python3
import torch
# 你的训练代码...
```

## 工作原理

### GPU 保活

永动机持续运行矩阵乘法保持 GPU 利用率在 30-50%：

```python
# 当 GPU 利用率 < 30% 时
c = torch.mm(a, b)  # 执行矩阵乘法提升利用率

# 当 GPU 利用率 > 50% 时（有真实任务）
time.sleep(interval)  # 降低频率让出资源
```

### 任务队列

1. 用户通过 `gpu_submit` 将脚本复制到 `pending/` 目录
2. 永动机监控 `pending/` 目录，按时间顺序执行任务
3. 执行时脚本移动到 `running/`，日志写入 `logs/`
4. 完成后移动到 `done/` 或 `failed/`

### 远程控制

通过共享存储的 `control.json` 文件传递控制信号：

```json
{"status": "running", "current_task": "train.sh", "stop_task": "train.sh"}
```

## SCO ACP 管理命令

```bash
# 查看作业列表
sco acp jobs list --workspace-name=share-space

# 实时监控（过滤用户名）
watch -n 0.5 "sco acp jobs list --workspace-name=share-space --page-size 500 | awk 'NR<=3 || /yzy/'"

# 删除作业
sco acp jobs delete --workspace-name=share-space pt-xxxxxxxx

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
GPU_UTIL_THRESHOLD = 30  # 低于此值开始保活
GPU_UTIL_TARGET = 50     # 目标利用率
```

### Q: 任务执行失败怎么查看原因？

```bash
# 查看失败任务
ls /mnt/afs/00036/yzy/gpu_queue/failed/

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
bash submit.sh
```

## License

MIT

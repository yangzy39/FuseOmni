#!/usr/bin/env python3
"""
GPU Perpetual Motion Machine (永动机)
=====================================
Keeps GPU busy to prevent idle-kill while monitoring a task queue.

Features:
1. Dynamic GPU utilization control - adjusts workload based on real GPU usage
2. Auto-sized matrix based on GPU memory - ensures >70% utilization when idle
3. Task queue monitoring - executes user-submitted scripts from shared storage
4. Graceful task management - can submit/stop tasks without logging into node
5. Per-instance queue isolation - each JOB_ID has its own queue directory

Architecture:
- Main loop monitors task queue directory
- Background threads keep GPUs warm when idle
- User submits tasks by copying scripts to queue directory
"""

import os
import sys
import time
import signal
import subprocess
import threading
import multiprocessing
from pathlib import Path
from datetime import datetime
import json
import argparse

# ==================== Configuration ====================
# JOB_ID for queue isolation (each perpetual motion instance has its own queue)
JOB_ID = os.environ.get("JOB_ID", "default")

# Queue directories (on shared storage) - isolated by JOB_ID
QUEUE_ROOT = os.environ.get("QUEUE_ROOT", "/mnt/afs/00036/yzy/gpu_queue")
QUEUE_BASE = os.path.join(QUEUE_ROOT, JOB_ID)
PENDING_DIR = os.path.join(QUEUE_BASE, "pending")
RUNNING_DIR = os.path.join(QUEUE_BASE, "running")
DONE_DIR = os.path.join(QUEUE_BASE, "done")
FAILED_DIR = os.path.join(QUEUE_BASE, "failed")
LOG_DIR = os.path.join(QUEUE_BASE, "logs")
CONTROL_FILE = os.path.join(QUEUE_BASE, "control.json")

# GPU settings
GPU_UTIL_THRESHOLD = 50  # Start dummy workload if GPU util < this
GPU_UTIL_TARGET = 70     # Target GPU utilization when running dummy workload
GPU_UTIL_HIGH = 85       # Slow down if GPU util > this (real work running)
CHECK_INTERVAL = 5       # Seconds between queue checks
GPU_CHECK_INTERVAL = 2   # Seconds between GPU util checks

# Memory usage target for matrix allocation (fraction of total GPU memory)
MEMORY_USAGE_TARGET = 0.6  # Use 60% of GPU memory for matrices

# ==================== GPU Keep-Alive ====================
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not available, GPU keep-alive disabled")

gpu_busy_flags = {}  # {gpu_id: threading.Event}
gpu_processes = {}   # {gpu_id: Process}


def get_gpu_info(gpu_id):
    """Get GPU utilization and memory info using nvidia-smi"""
    try:
        cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.total,memory.used,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split()).decode('ascii').strip()
        lines = output.split('\n')
        if gpu_id < len(lines):
            parts = lines[gpu_id].split(',')
            return {
                'utilization': int(parts[0].strip()),
                'memory_total': int(parts[1].strip()),  # MiB
                'memory_used': int(parts[2].strip()),   # MiB
                'memory_free': int(parts[3].strip()),   # MiB
            }
    except Exception as e:
        print(f"[WARN] Failed to get GPU info: {e}")
    return {'utilization': 100, 'memory_total': 0, 'memory_used': 0, 'memory_free': 0}


def get_gpu_utilization(gpu_id):
    """Get GPU utilization percentage using nvidia-smi"""
    return get_gpu_info(gpu_id)['utilization']


def calculate_matrix_size(gpu_id, dtype=torch.float16):
    """
    Calculate optimal matrix size based on GPU memory to achieve target utilization.
    
    For matrix multiplication C = A @ B where A, B, C are all (N, N):
    - Memory needed: 3 * N^2 * bytes_per_element
    - For float16: 3 * N^2 * 2 bytes = 6 * N^2 bytes
    
    We target using MEMORY_USAGE_TARGET of available GPU memory.
    """
    info = get_gpu_info(gpu_id)
    memory_total_mb = info['memory_total']
    
    if memory_total_mb <= 0:
        # Fallback to a reasonable default
        print(f"[GPU-{gpu_id}] Could not detect memory, using default size")
        return 8192
    
    # Calculate available memory in bytes
    # Use total memory * target fraction
    target_memory_bytes = memory_total_mb * 1024 * 1024 * MEMORY_USAGE_TARGET
    
    # bytes per element
    if dtype == torch.float16:
        bytes_per_elem = 2
    elif dtype == torch.float32:
        bytes_per_elem = 4
    else:
        bytes_per_elem = 2
    
    # We need 3 matrices (A, B, C) of size N x N
    # Total memory = 3 * N^2 * bytes_per_elem
    # N^2 = target_memory / (3 * bytes_per_elem)
    # N = sqrt(target_memory / (3 * bytes_per_elem))
    import math
    n_squared = target_memory_bytes / (3 * bytes_per_elem)
    n = int(math.sqrt(n_squared))
    
    # Round down to nearest 1024 for better GPU efficiency
    n = (n // 1024) * 1024
    n = max(1024, n)  # Minimum size
    
    print(f"[GPU-{gpu_id}] Memory: {memory_total_mb} MiB, Matrix size: {n}x{n}, "
          f"Estimated memory usage: {3 * n * n * bytes_per_elem / 1024 / 1024:.0f} MiB")
    
    return n


def gpu_keep_alive_worker(gpu_id, stop_event):
    """Worker process that keeps a single GPU busy with dynamic matrix sizing"""
    if not TORCH_AVAILABLE:
        return
    
    try:
        device = torch.device(f'cuda:{gpu_id}')
        
        # Calculate optimal matrix size based on GPU memory
        size = calculate_matrix_size(gpu_id, dtype=torch.float16)
        
        # Allocate matrices
        a = torch.rand(size, size, device=device, dtype=torch.float16)
        b = torch.rand(size, size, device=device, dtype=torch.float16)
        
        interval = 0.01  # Start with 10ms sleep
        warmup_done = False
        
        print(f"[GPU-{gpu_id}] Keep-alive worker started (matrix: {size}x{size})")
        
        while not stop_event.is_set():
            gpu_util = get_gpu_utilization(gpu_id)
            
            if gpu_util < GPU_UTIL_THRESHOLD:
                # GPU idle, do more work to reach target
                c = torch.mm(a, b)
                _ = c.sum()
                # Speed up to increase utilization
                interval = max(0.001, interval * 0.8)
                
            elif gpu_util < GPU_UTIL_TARGET:
                # Below target but not idle, do some work
                c = torch.mm(a, b)
                _ = c.sum()
                # Maintain current pace
                
            elif gpu_util > GPU_UTIL_HIGH:
                # GPU busy (real work running), slow down significantly
                interval = min(2.0, interval * 1.5)
                
            else:
                # In target range (TARGET to HIGH), minor adjustments
                c = torch.mm(a, b)
                _ = c.sum()
                interval = min(0.1, interval * 1.1)
            
            time.sleep(interval)
            
            # Log status periodically
            if not warmup_done:
                warmup_done = True
                print(f"[GPU-{gpu_id}] Warmup complete, current util: {gpu_util}%")
            
    except Exception as e:
        print(f"[GPU-{gpu_id}] Keep-alive error: {e}")
        import traceback
        traceback.print_exc()


def start_gpu_keepers():
    """Start keep-alive workers for all GPUs"""
    if not TORCH_AVAILABLE:
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] Starting keep-alive for {num_gpus} GPUs")
    print(f"[INFO] Target utilization: {GPU_UTIL_TARGET}%-{GPU_UTIL_HIGH}%")
    
    for gpu_id in range(num_gpus):
        stop_event = multiprocessing.Event()
        p = multiprocessing.Process(
            target=gpu_keep_alive_worker,
            args=(gpu_id, stop_event),
            daemon=True
        )
        p.start()
        gpu_busy_flags[gpu_id] = stop_event
        gpu_processes[gpu_id] = p


def stop_gpu_keepers():
    """Stop all keep-alive workers"""
    for gpu_id, event in gpu_busy_flags.items():
        event.set()
    for gpu_id, p in gpu_processes.items():
        p.terminate()
        p.join(timeout=5)


# ==================== Task Queue ====================
def init_queue():
    """Initialize queue directories"""
    for d in [PENDING_DIR, RUNNING_DIR, DONE_DIR, FAILED_DIR, LOG_DIR]:
        os.makedirs(d, exist_ok=True)
    
    # Initialize control file
    if not os.path.exists(CONTROL_FILE):
        write_control({"status": "running", "current_task": None, "job_id": JOB_ID})
    
    print(f"[INFO] Queue initialized at {QUEUE_BASE}")
    print(f"[INFO] JOB_ID: {JOB_ID}")


def write_control(data):
    """Write control file atomically"""
    data["job_id"] = JOB_ID  # Always include JOB_ID
    tmp_file = CONTROL_FILE + ".tmp"
    with open(tmp_file, 'w') as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_file, CONTROL_FILE)


def read_control():
    """Read control file"""
    try:
        with open(CONTROL_FILE, 'r') as f:
            return json.load(f)
    except:
        return {"status": "running", "current_task": None, "job_id": JOB_ID}


def get_pending_tasks():
    """Get list of pending tasks sorted by submission time"""
    tasks = []
    if not os.path.exists(PENDING_DIR):
        return tasks
    
    for f in os.listdir(PENDING_DIR):
        if f.endswith('.sh') or f.endswith('.py'):
            path = os.path.join(PENDING_DIR, f)
            mtime = os.path.getmtime(path)
            tasks.append((mtime, f, path))
    
    tasks.sort()  # Sort by modification time (FIFO)
    return [(name, path) for _, name, path in tasks]


def execute_task(task_name, task_path):
    """Execute a task script"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(LOG_DIR, f"{task_name}_{timestamp}.log")
    running_path = os.path.join(RUNNING_DIR, task_name)
    
    # Move to running
    os.rename(task_path, running_path)
    
    # Update control
    write_control({"status": "running", "current_task": task_name, "started": timestamp})
    
    print(f"[TASK] Starting: {task_name}")
    print(f"[TASK] Log: {log_file}")
    
    try:
        # Determine executor
        if task_name.endswith('.py'):
            cmd = ['python', running_path]
        else:
            cmd = ['bash', running_path]
        
        # Execute with logging
        with open(log_file, 'w') as log:
            log.write(f"=== Task: {task_name} ===\n")
            log.write(f"=== JOB_ID: {JOB_ID} ===\n")
            log.write(f"=== Started: {timestamp} ===\n\n")
            log.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(running_path) or '.',
                env={**os.environ, 'TASK_NAME': task_name, 'JOB_ID': JOB_ID}
            )
            
            # Monitor for stop signal
            while process.poll() is None:
                control = read_control()
                if control.get("stop_task") == task_name:
                    print(f"[TASK] Stop requested for: {task_name}")
                    process.terminate()
                    time.sleep(2)
                    if process.poll() is None:
                        process.kill()
                    log.write(f"\n=== Task stopped by user ===\n")
                    break
                time.sleep(1)
            
            return_code = process.returncode
            log.write(f"\n=== Finished with code: {return_code} ===\n")
        
        # Move to done/failed
        if return_code == 0:
            dest = os.path.join(DONE_DIR, f"{task_name}_{timestamp}")
            print(f"[TASK] Completed: {task_name}")
        else:
            dest = os.path.join(FAILED_DIR, f"{task_name}_{timestamp}")
            print(f"[TASK] Failed: {task_name} (code {return_code})")
        
        os.rename(running_path, dest)
        
    except Exception as e:
        print(f"[ERROR] Task execution failed: {e}")
        # Move to failed
        dest = os.path.join(FAILED_DIR, f"{task_name}_{timestamp}_error")
        if os.path.exists(running_path):
            os.rename(running_path, dest)
    
    # Clear current task
    write_control({"status": "running", "current_task": None})


def main_loop():
    """Main queue monitoring loop"""
    print("=" * 60)
    print("GPU Perpetual Motion Machine (永动机)")
    print("=" * 60)
    print(f"[INFO] JOB_ID: {JOB_ID}")
    print(f"[INFO] Queue directory: {QUEUE_BASE}")
    print(f"[INFO] Submit tasks to: {PENDING_DIR}")
    print(f"[INFO] Target GPU utilization: {GPU_UTIL_TARGET}%-{GPU_UTIL_HIGH}%")
    print("=" * 60)
    
    init_queue()
    start_gpu_keepers()
    
    try:
        while True:
            # Check control file for shutdown
            control = read_control()
            if control.get("status") == "shutdown":
                print("[INFO] Shutdown requested")
                break
            
            # Check for pending tasks
            tasks = get_pending_tasks()
            if tasks:
                task_name, task_path = tasks[0]
                execute_task(task_name, task_path)
            else:
                # No tasks, just keep GPUs warm
                time.sleep(CHECK_INTERVAL)
                
    except KeyboardInterrupt:
        print("\n[INFO] Shutting down...")
    finally:
        stop_gpu_keepers()
        write_control({"status": "stopped", "current_task": None})


# ==================== Entry Point ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Perpetual Motion Machine")
    parser.add_argument("--job-id", type=str, default=None,
                        help="Job ID for queue isolation (default: from JOB_ID env or 'default')")
    parser.add_argument("--queue-root", type=str, default=None,
                        help="Root directory for queues (default: from QUEUE_ROOT env)")
    args = parser.parse_args()
    
    # Override from command line if provided
    if args.job_id:
        JOB_ID = args.job_id
        os.environ["JOB_ID"] = JOB_ID
    if args.queue_root:
        QUEUE_ROOT = args.queue_root
        os.environ["QUEUE_ROOT"] = QUEUE_ROOT
    
    # Recalculate paths with new JOB_ID
    QUEUE_BASE = os.path.join(QUEUE_ROOT, JOB_ID)
    PENDING_DIR = os.path.join(QUEUE_BASE, "pending")
    RUNNING_DIR = os.path.join(QUEUE_BASE, "running")
    DONE_DIR = os.path.join(QUEUE_BASE, "done")
    FAILED_DIR = os.path.join(QUEUE_BASE, "failed")
    LOG_DIR = os.path.join(QUEUE_BASE, "logs")
    CONTROL_FILE = os.path.join(QUEUE_BASE, "control.json")
    
    main_loop()

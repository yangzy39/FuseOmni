#!/usr/bin/env python3
"""
GPU Perpetual Motion Machine (永动机)
=====================================
Keeps GPU busy to prevent idle-kill while monitoring a task queue.

Features:
1. Dynamic GPU utilization control - adjusts workload based on real GPU usage
2. Task queue monitoring - executes user-submitted scripts from shared storage
3. Graceful task management - can submit/stop tasks without logging into node

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

# ==================== Configuration ====================
# Queue directories (on shared storage)
QUEUE_BASE = os.environ.get("QUEUE_BASE", "/mnt/afs/00036/yzy/gpu_queue")
PENDING_DIR = os.path.join(QUEUE_BASE, "pending")
RUNNING_DIR = os.path.join(QUEUE_BASE, "running")
DONE_DIR = os.path.join(QUEUE_BASE, "done")
FAILED_DIR = os.path.join(QUEUE_BASE, "failed")
LOG_DIR = os.path.join(QUEUE_BASE, "logs")
CONTROL_FILE = os.path.join(QUEUE_BASE, "control.json")

# GPU settings
GPU_UTIL_THRESHOLD = 30  # Start dummy workload if GPU util < this
GPU_UTIL_TARGET = 50     # Target GPU utilization when running dummy workload
CHECK_INTERVAL = 5       # Seconds between queue checks
GPU_CHECK_INTERVAL = 2   # Seconds between GPU util checks

# ==================== GPU Keep-Alive ====================
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not available, GPU keep-alive disabled")

gpu_busy_flags = {}  # {gpu_id: threading.Event}
gpu_processes = {}   # {gpu_id: Process}


def get_gpu_utilization(gpu_id):
    """Get GPU utilization percentage using nvidia-smi"""
    try:
        cmd = "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd.split()).decode('ascii').strip()
        utils = output.split('\n')
        return int(utils[gpu_id]) if gpu_id < len(utils) else 0
    except Exception as e:
        print(f"[WARN] Failed to get GPU util: {e}")
        return 100  # Assume busy on error


def gpu_keep_alive_worker(gpu_id, stop_event):
    """Worker process that keeps a single GPU busy"""
    if not TORCH_AVAILABLE:
        return
    
    try:
        device = torch.device(f'cuda:{gpu_id}')
        size = 1024 * 8  # Matrix size
        interval = 0.01
        
        a = torch.rand(size, size, device=device, dtype=torch.float16)
        b = torch.rand(size, size, device=device, dtype=torch.float16)
        
        print(f"[GPU-{gpu_id}] Keep-alive worker started")
        
        while not stop_event.is_set():
            # Check if we should be active
            gpu_util = get_gpu_utilization(gpu_id)
            
            if gpu_util < GPU_UTIL_THRESHOLD:
                # GPU idle, do some work
                c = torch.mm(a, b)
                _ = c.sum()
                interval = max(0.001, interval * 0.9)
            elif gpu_util > GPU_UTIL_TARGET:
                # GPU busy (real work running), slow down
                interval = min(1.0, interval * 1.5)
            
            time.sleep(interval)
            
    except Exception as e:
        print(f"[GPU-{gpu_id}] Keep-alive error: {e}")


def start_gpu_keepers():
    """Start keep-alive workers for all GPUs"""
    if not TORCH_AVAILABLE:
        return
    
    num_gpus = torch.cuda.device_count()
    print(f"[INFO] Starting keep-alive for {num_gpus} GPUs")
    
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
        write_control({"status": "running", "current_task": None})
    
    print(f"[INFO] Queue initialized at {QUEUE_BASE}")


def write_control(data):
    """Write control file atomically"""
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
        return {"status": "running", "current_task": None}


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
            log.write(f"=== Started: {timestamp} ===\n\n")
            log.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=log,
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(running_path) or '.',
                env={**os.environ, 'TASK_NAME': task_name}
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
    print("[INFO] Perpetual motion machine started")
    print(f"[INFO] Queue directory: {QUEUE_BASE}")
    print(f"[INFO] Submit tasks by copying scripts to: {PENDING_DIR}")
    print("[INFO] Press Ctrl+C to stop")
    
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
    main_loop()

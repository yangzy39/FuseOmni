#!/bin/bash
# ========== GPU Perpetual Motion Machine - Client Tools ==========
# Usage: source client.sh
#        gpu_submit my_script.sh
#        gpu_status
#        gpu_stop my_script.sh
# ==================================================================

# Configuration - modify as needed
export QUEUE_BASE="${QUEUE_BASE:-/mnt/afs/00036/yzy/gpu_queue}"
export PENDING_DIR="${QUEUE_BASE}/pending"
export RUNNING_DIR="${QUEUE_BASE}/running"
export DONE_DIR="${QUEUE_BASE}/done"
export FAILED_DIR="${QUEUE_BASE}/failed"
export LOG_DIR="${QUEUE_BASE}/logs"
export CONTROL_FILE="${QUEUE_BASE}/control.json"

# ==================== Submit Task ====================
gpu_submit() {
    if [ -z "$1" ]; then
        echo "Usage: gpu_submit <script.sh|script.py> [priority]"
        echo "  priority: Optional timestamp prefix for ordering (default: current time)"
        return 1
    fi
    
    local script="$1"
    local priority="${2:-$(date +%Y%m%d%H%M%S)}"
    
    if [ ! -f "$script" ]; then
        echo "Error: Script not found: $script"
        return 1
    fi
    
    # Create queue directories if needed
    mkdir -p "$PENDING_DIR"
    
    # Copy script to pending queue with priority prefix
    local basename=$(basename "$script")
    local dest="${PENDING_DIR}/${priority}_${basename}"
    
    cp "$script" "$dest"
    chmod +x "$dest"
    
    echo "Task submitted: ${priority}_${basename}"
    echo "Queue position: $(ls -1 "$PENDING_DIR" 2>/dev/null | wc -l)"
    echo ""
    echo "Monitor with: gpu_status"
    echo "View logs at: ${LOG_DIR}/"
}

# ==================== Submit Inline Command ====================
gpu_run() {
    if [ -z "$1" ]; then
        echo "Usage: gpu_run '<command>'"
        echo "  Example: gpu_run 'python train.py --epochs 10'"
        return 1
    fi
    
    local cmd="$1"
    local timestamp=$(date +%Y%m%d%H%M%S)
    local script_name="inline_${timestamp}.sh"
    local script_path="${PENDING_DIR}/${script_name}"
    
    mkdir -p "$PENDING_DIR"
    
    cat > "$script_path" << EOF
#!/bin/bash
# Auto-generated inline task
# Submitted: $(date)
# Command: $cmd

set -e
cd "${PWD}"
$cmd
EOF
    
    chmod +x "$script_path"
    
    echo "Inline task submitted: $script_name"
    echo "Command: $cmd"
}

# ==================== Queue Status ====================
gpu_status() {
    echo "========== GPU Queue Status =========="
    echo ""
    
    # Current task
    if [ -f "$CONTROL_FILE" ]; then
        echo "Control: $(cat "$CONTROL_FILE")"
    fi
    echo ""
    
    # Pending
    echo "--- Pending Tasks ---"
    if [ -d "$PENDING_DIR" ] && [ "$(ls -A "$PENDING_DIR" 2>/dev/null)" ]; then
        ls -lt "$PENDING_DIR" | head -20
    else
        echo "(empty)"
    fi
    echo ""
    
    # Running
    echo "--- Running Tasks ---"
    if [ -d "$RUNNING_DIR" ] && [ "$(ls -A "$RUNNING_DIR" 2>/dev/null)" ]; then
        ls -lt "$RUNNING_DIR"
    else
        echo "(none)"
    fi
    echo ""
    
    # Recent completed
    echo "--- Recent Completed (last 5) ---"
    if [ -d "$DONE_DIR" ] && [ "$(ls -A "$DONE_DIR" 2>/dev/null)" ]; then
        ls -lt "$DONE_DIR" | head -6
    else
        echo "(none)"
    fi
    echo ""
    
    # Recent failed
    echo "--- Recent Failed (last 5) ---"
    if [ -d "$FAILED_DIR" ] && [ "$(ls -A "$FAILED_DIR" 2>/dev/null)" ]; then
        ls -lt "$FAILED_DIR" | head -6
    else
        echo "(none)"
    fi
    
    echo "======================================"
}

# ==================== Stop Task ====================
gpu_stop() {
    if [ -z "$1" ]; then
        echo "Usage: gpu_stop <task_name|all>"
        echo "  Stop a running task or 'all' to clear pending queue"
        return 1
    fi
    
    local target="$1"
    
    if [ "$target" = "all" ]; then
        # Clear pending queue
        rm -f "${PENDING_DIR}"/*
        echo "Cleared all pending tasks"
        
        # Signal stop to running task
        if [ -f "$CONTROL_FILE" ]; then
            local current=$(python3 -c "import json; print(json.load(open('$CONTROL_FILE')).get('current_task', ''))" 2>/dev/null)
            if [ -n "$current" ]; then
                echo '{"status": "running", "current_task": "'$current'", "stop_task": "'$current'"}' > "$CONTROL_FILE"
                echo "Stop signal sent to: $current"
            fi
        fi
    else
        # Stop specific task
        # Check if in pending (just remove it)
        if [ -f "${PENDING_DIR}/${target}" ]; then
            rm -f "${PENDING_DIR}/${target}"
            echo "Removed pending task: $target"
            return 0
        fi
        
        # Check if running (signal stop)
        if [ -f "${RUNNING_DIR}/${target}" ]; then
            echo '{"status": "running", "current_task": "'$target'", "stop_task": "'$target'"}' > "$CONTROL_FILE"
            echo "Stop signal sent to running task: $target"
            return 0
        fi
        
        echo "Task not found: $target"
        return 1
    fi
}

# ==================== View Logs ====================
gpu_logs() {
    local task="${1:-}"
    
    if [ -z "$task" ]; then
        echo "Recent logs:"
        ls -lt "$LOG_DIR" 2>/dev/null | head -10
        echo ""
        echo "Usage: gpu_logs <task_name_pattern>"
        return 0
    fi
    
    # Find matching log
    local log_file=$(ls -t "${LOG_DIR}"/*"${task}"* 2>/dev/null | head -1)
    
    if [ -n "$log_file" ]; then
        echo "=== Log: $log_file ==="
        tail -100 "$log_file"
    else
        echo "No log found matching: $task"
    fi
}

# ==================== Tail Log (Follow) ====================
gpu_tail() {
    local task="${1:-}"
    
    if [ -z "$task" ]; then
        # Follow latest log
        local log_file=$(ls -t "${LOG_DIR}"/*.log 2>/dev/null | head -1)
    else
        local log_file=$(ls -t "${LOG_DIR}"/*"${task}"* 2>/dev/null | head -1)
    fi
    
    if [ -n "$log_file" ]; then
        echo "=== Following: $log_file ==="
        tail -f "$log_file"
    else
        echo "No log found"
    fi
}

# ==================== Shutdown Perpetual Motion ====================
gpu_shutdown() {
    echo '{"status": "shutdown", "current_task": null}' > "$CONTROL_FILE"
    echo "Shutdown signal sent to perpetual motion machine"
}

# ==================== Help ====================
gpu_help() {
    echo "GPU Queue Client Commands:"
    echo ""
    echo "  gpu_submit <script.sh>  - Submit a script to the queue"
    echo "  gpu_run '<command>'     - Submit an inline command"
    echo "  gpu_status              - Show queue status"
    echo "  gpu_stop <task|all>     - Stop a task or clear queue"
    echo "  gpu_logs [pattern]      - View task logs"
    echo "  gpu_tail [pattern]      - Follow task log in real-time"
    echo "  gpu_shutdown            - Shutdown the perpetual motion machine"
    echo ""
    echo "Queue directory: $QUEUE_BASE"
}

echo "GPU Queue Client loaded. Type 'gpu_help' for commands."

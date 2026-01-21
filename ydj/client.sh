#!/bin/bash
# ========== GPU Perpetual Motion Machine - Client Tools ==========
# Usage: 
#   export JOB_ID=my-job-name  # Required: specify which instance to use
#   source client.sh
#   gpu_submit my_script.sh
#   gpu_status
# ==================================================================

# ========== Configuration ==========
# JOB_ID must be set before sourcing this script
if [ -z "$JOB_ID" ]; then
    echo "WARNING: JOB_ID not set. Using 'default'."
    echo "Set JOB_ID before sourcing: export JOB_ID=your-job-name"
    export JOB_ID="default"
fi

export QUEUE_ROOT="${QUEUE_ROOT:-/mnt/afs/00036/yzy/gpu_queue}"
export QUEUE_BASE="${QUEUE_ROOT}/${JOB_ID}"
export PENDING_DIR="${QUEUE_BASE}/pending"
export RUNNING_DIR="${QUEUE_BASE}/running"
export DONE_DIR="${QUEUE_BASE}/done"
export FAILED_DIR="${QUEUE_BASE}/failed"
export LOG_DIR="${QUEUE_BASE}/logs"
export CONTROL_FILE="${QUEUE_BASE}/control.json"

# ==================== Switch Job ====================
gpu_use() {
    if [ -z "$1" ]; then
        echo "Current JOB_ID: $JOB_ID"
        echo "Queue path: $QUEUE_BASE"
        echo ""
        echo "Usage: gpu_use <job_id>"
        echo "  Switch to a different perpetual motion instance"
        echo ""
        echo "Available instances:"
        ls -1 "$QUEUE_ROOT" 2>/dev/null || echo "(none)"
        return 0
    fi
    
    export JOB_ID="$1"
    export QUEUE_BASE="${QUEUE_ROOT}/${JOB_ID}"
    export PENDING_DIR="${QUEUE_BASE}/pending"
    export RUNNING_DIR="${QUEUE_BASE}/running"
    export DONE_DIR="${QUEUE_BASE}/done"
    export FAILED_DIR="${QUEUE_BASE}/failed"
    export LOG_DIR="${QUEUE_BASE}/logs"
    export CONTROL_FILE="${QUEUE_BASE}/control.json"
    
    echo "Switched to JOB_ID: $JOB_ID"
    echo "Queue path: $QUEUE_BASE"
}

# ==================== Submit Task ====================
gpu_submit() {
    if [ -z "$1" ]; then
        echo "Usage: gpu_submit <script.sh|script.py> [priority]"
        echo "  priority: Optional timestamp prefix for ordering (default: current time)"
        echo ""
        echo "Current JOB_ID: $JOB_ID"
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
    
    echo "Task submitted to: $JOB_ID"
    echo "  File: ${priority}_${basename}"
    echo "  Queue position: $(ls -1 "$PENDING_DIR" 2>/dev/null | wc -l)"
    echo ""
    echo "Monitor with: gpu_status"
}

# ==================== Submit Inline Command ====================
gpu_run() {
    if [ -z "$1" ]; then
        echo "Usage: gpu_run '<command>'"
        echo "  Example: gpu_run 'python train.py --epochs 10'"
        echo ""
        echo "Current JOB_ID: $JOB_ID"
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
# JOB_ID: $JOB_ID
# Submitted: $(date)
# Command: $cmd

set -e
cd "${PWD}"
$cmd
EOF
    
    chmod +x "$script_path"
    
    echo "Inline task submitted to: $JOB_ID"
    echo "  File: $script_name"
    echo "  Command: $cmd"
}

# ==================== Queue Status ====================
gpu_status() {
    echo "========== GPU Queue Status =========="
    echo "JOB_ID: $JOB_ID"
    echo "Queue:  $QUEUE_BASE"
    echo ""
    
    # Current task
    if [ -f "$CONTROL_FILE" ]; then
        echo "Control: $(cat "$CONTROL_FILE")"
    else
        echo "Control: (not initialized - perpetual motion not running?)"
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

# ==================== List All Instances ====================
gpu_list() {
    echo "========== All Perpetual Motion Instances =========="
    echo "Queue root: $QUEUE_ROOT"
    echo ""
    
    if [ ! -d "$QUEUE_ROOT" ]; then
        echo "(no instances found)"
        return 0
    fi
    
    for dir in "$QUEUE_ROOT"/*/; do
        if [ -d "$dir" ]; then
            local name=$(basename "$dir")
            local control="${dir}control.json"
            local pending_count=$(ls -1 "${dir}pending" 2>/dev/null | wc -l)
            local status="unknown"
            
            if [ -f "$control" ]; then
                status=$(python3 -c "import json; print(json.load(open('$control')).get('status', 'unknown'))" 2>/dev/null || echo "unknown")
            fi
            
            printf "  %-20s  status: %-10s  pending: %d\n" "$name" "$status" "$pending_count"
        fi
    done
    
    echo ""
    echo "Switch instance: gpu_use <job_id>"
    echo "======================================================"
}

# ==================== Stop Task ====================
gpu_stop() {
    if [ -z "$1" ]; then
        echo "Usage: gpu_stop <task_name|all>"
        echo "  Stop a running task or 'all' to clear pending queue and kill Python processes"
        echo ""
        echo "Current JOB_ID: $JOB_ID"
        return 1
    fi
    
    local target="$1"
    
    if [ "$target" = "all" ]; then
        # Clear pending queue
        rm -f "${PENDING_DIR}"/*
        echo "Cleared all pending tasks for: $JOB_ID"
        
        # Signal stop to running task and kill all Python processes
        if [ -f "$CONTROL_FILE" ]; then
            local current=$(python3 -c "import json; print(json.load(open('$CONTROL_FILE')).get('current_task', ''))" 2>/dev/null)
            if [ -n "$current" ] && [ "$current" != "None" ]; then
                echo "{\"status\": \"running\", \"current_task\": \"$current\", \"stop_task\": \"$current\", \"kill_python\": true, \"job_id\": \"$JOB_ID\"}" > "$CONTROL_FILE"
                echo "Stop signal sent to: $current"
                echo "Kill Python processes signal sent"
            else
                # No current task, just send kill_python signal
                echo "{\"status\": \"running\", \"current_task\": null, \"kill_python\": true, \"job_id\": \"$JOB_ID\"}" > "$CONTROL_FILE"
                echo "Kill Python processes signal sent"
            fi
        fi
    else
        # Stop specific task
        # Check if in pending (just remove it)
        local found=0
        for f in "${PENDING_DIR}"/*"${target}"*; do
            if [ -f "$f" ]; then
                rm -f "$f"
                echo "Removed pending task: $(basename "$f")"
                found=1
            fi
        done
        
        if [ $found -eq 0 ]; then
            # Check if running (signal stop)
            if [ -f "$CONTROL_FILE" ]; then
                local current=$(python3 -c "import json; print(json.load(open('$CONTROL_FILE')).get('current_task', ''))" 2>/dev/null)
                if [[ "$current" == *"$target"* ]]; then
                    echo "{\"status\": \"running\", \"current_task\": \"$current\", \"stop_task\": \"$current\", \"kill_python\": true, \"job_id\": \"$JOB_ID\"}" > "$CONTROL_FILE"
                    echo "Stop signal sent to running task: $current"
                    echo "Kill Python processes signal sent"
                    return 0
                fi
            fi
            echo "Task not found: $target"
            return 1
        fi
    fi
}

# ==================== View Logs ====================
gpu_logs() {
    local task="${1:-}"
    
    if [ -z "$task" ]; then
        echo "Recent logs for: $JOB_ID"
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
        echo "No log found for: $JOB_ID"
    fi
}

# ==================== Shutdown Perpetual Motion ====================
gpu_shutdown() {
    echo "{\"status\": \"shutdown\", \"current_task\": null, \"job_id\": \"$JOB_ID\"}" > "$CONTROL_FILE"
    echo "Shutdown signal sent to: $JOB_ID"
    echo "Remember to delete the SCO ACP job: sco acp jobs delete --workspace-name=share-space ydj-${JOB_ID}"
}

# ==================== Help ====================
gpu_help() {
    echo "GPU Queue Client Commands"
    echo "========================="
    echo ""
    echo "Current JOB_ID: $JOB_ID"
    echo "Queue path:     $QUEUE_BASE"
    echo ""
    echo "Instance Management:"
    echo "  gpu_use <job_id>      - Switch to a different instance"
    echo "  gpu_list              - List all instances"
    echo ""
    echo "Task Management:"
    echo "  gpu_submit <script>   - Submit a script to the queue"
    echo "  gpu_run '<command>'   - Submit an inline command"
    echo "  gpu_status            - Show queue status"
    echo "  gpu_stop <task|all>   - Stop a task or clear queue"
    echo ""
    echo "Logs:"
    echo "  gpu_logs [pattern]    - View task logs"
    echo "  gpu_tail [pattern]    - Follow task log in real-time"
    echo ""
    echo "Control:"
    echo "  gpu_shutdown          - Shutdown the perpetual motion machine"
    echo ""
}

echo "GPU Queue Client loaded for JOB_ID: $JOB_ID"
echo "Type 'gpu_help' for commands, 'gpu_use <id>' to switch instance."

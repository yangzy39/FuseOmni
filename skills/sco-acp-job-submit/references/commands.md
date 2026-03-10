# SCO ACP Command Reference

## Job Lifecycle Commands

### Create Job
```bash
sco acp jobs create \
  --workspace-name=share-space \
  --aec2-name=share-cluster \
  --job-name=<name> \
  --container-image-url=<image> \
  --training-framework=pytorch \
  --worker-nodes=1 \
  --worker-spec=n6ls.iu.i40.<gpu_count> \
  --priority=normal \
  --storage-mount=<storage_id>:/mnt/afs \
  --command="<shell_command>"
```

### Delete Job
```bash
sco acp jobs delete --workspace-name=share-space <job_id>
```

**Warning**: Immediately terminates training and releases GPU. Unsaved work is lost.

### List Jobs
```bash
# All jobs
sco acp jobs list --workspace-name=share-space

# Real-time monitoring (filter by username)
watch -n 0.5 "sco acp jobs list --workspace-name=share-space --page-size 500 | awk 'NR<=3 || /<username>/'"
```

### Login to Running Node
```bash
Jobid=<job_id>
sco acp jobs exec \
  --workspace-name=share-space \
  --worker-name=$(sco acp jobs get-workers --workspace-name=share-space $Jobid | grep "worker-0" | awk -F'|' '{print $2}' | xargs) \
  $Jobid
```

Once logged in:
- `nvidia-smi` - Check GPU usage
- `tail -f /path/to/log.txt` - Monitor training logs
- `htop` - Check CPU/memory

## Fixed Parameters

### Workspace & Cluster
```bash
--workspace-name=share-space
--aec2-name=share-cluster
```

### Container Image (CUDA 12.4 + PyTorch 2.3)
```bash
--container-image-url="registry.cn-sh-01.sensecore.cn/lepton-trainingjob/nvidia24.04-ubuntu22.04-py3.10-cuda12.4-cudnn9.1-torch2.3.0-transformerengine1.5:v1.0.0-20241130-nvdia-base-image"
```

### Storage Mount
```bash
--storage-mount=01995892-d478-76d8-aec7-13fd8284477e:/mnt/afs
```

## Worker Spec Format

Pattern: `n6ls.iu.i40.<gpu_count>`

| GPUs | Spec |
|------|------|
| 1 | `n6ls.iu.i40.1` |
| 2 | `n6ls.iu.i40.2` |
| 4 | `n6ls.iu.i40.4` |
| 8 | `n6ls.iu.i40.8` |

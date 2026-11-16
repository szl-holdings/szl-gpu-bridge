# Training alongside a serving GPU

`shared_gpu_supervisor.py` supervises one training process group on Linux. It
refuses admission unless the selected GPU has at least 32 GiB free and the local
serving health route returns HTTP 200. A stable per-GPU `flock` prevents duplicate
launches by cooperating supervisors. Other launchers must use the same lock.

Use a dedicated environment, an immutable worker revision, a unique output and
state directory, and the training worker's own allocation cap. Provision credentials
through the existing runtime environment; the supervisor inherits them but does
not record environment values or command arguments in its state.
Set `CUDA_VISIBLE_DEVICES=0` on this single-GPU host and
`SZL_DISABLE_RUNTIME_INSTALL=1` after provisioning the environment. Missing optional
energy libraries then remain unavailable without changing installed dependencies.
Pass immutable `--base-revision` and `--dataset-revision` commit SHAs to the worker.

```sh
python3 tools/a6000-migration/shared_gpu_supervisor.py \
  --lock-file /home/Ubuntu/.cache/szl/gpu-0-training.lock \
  --state-dir /home/Ubuntu/lean-v2-run/supervisor \
  --health-url http://127.0.0.1:8081/health \
  --gpu-index 0 --min-free-mib 32768 --abort-free-mib 4096 \
  --poll-seconds 15 --health-failures 3 --grace-seconds 90 \
  --max-seconds 21600 -- \
  /home/Ubuntu/lean-v2-run/venv/bin/python tools/salad/salad_worker.py \
  --base Qwen/Qwen2.5-Coder-3B-Instruct \
  --dataset SZLHOLDINGS/lean-math-sft-v2 --dataset-train-file train.jsonl \
  --adapter-out SZLHOLDINGS/salad-lean-math-v2-r1 \
  --max-steps 200 --save-steps 25 \
  --output-dir /home/Ubuntu/lean-v2-run/checkpoints \
  --gradient-checkpointing --gpu-memory-budget-mib 24576 --gpu-reserve-mib 8192
```

The GPU index must refer to the same physical GPU as the worker's CUDA device.
Coordinate or retire an older waiter for the same adapter before launching; this
lock cannot prevent a legacy launcher that does not participate in the lock.
Read back the private target repository before launching so a completed run is not
accidentally repeated. Pin dataset and base revisions in the launch record.

The supervisor samples every 15 seconds. Three consecutive failed health checks,
two consecutive samples below 4 GiB free, unavailable GPU telemetry, an external
termination request, or the six-hour deadline cause SIGTERM to its own child
process group. If the child has not exited after 90 seconds, it sends SIGKILL to
that same group. It never searches for or signals serving processes. Checkpoints
must already be saved periodically: SIGTERM does not guarantee a final save.

`state.json` is atomically replaced with mode 0600. It records observed memory,
health, child PID, stop reason and exit code. A lock conflict writes a separate
`refused-<pid>.json` without overwriting an active run's state. Exit status 75 is a
refusal or protective stop; 124 is a deadline; 70 is a supervisor error. Otherwise
the child's exit status is returned. Child success is not proof of completed
training: verify newly trained steps, checkpoint contents, final publication,
dataset digest and the completion receipt independently.

This protects memory headroom and detected service availability. It does not
partition GPU compute or guarantee serving latency. Record a short generation
baseline, check generation during the first training steps, and stop the training
supervisor if serving latency is unacceptable. Do not change serving processes to
make a training job fit.

Shared-mode energy receipts explicitly describe whole-device consumption,
including the serving workload. They do not attribute joules to training alone.

Run the GPU-free supervision tests with:

```sh
python -m unittest discover -s tests -p 'test_shared_gpu_supervisor.py' -v
```

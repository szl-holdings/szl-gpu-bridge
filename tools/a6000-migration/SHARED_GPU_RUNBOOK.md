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

## Data format and quality admission

For string `prompt`/`completion` instruction datasets, explicitly select
`--prompt-completion-format chat`. The pinned tokenizer serializes and tokenizes
user and assistant turns with no extra generation prompt. The default `verbatim` mode
preserves both fields exactly, but now refuses a boundary with no whitespace on
either side. This intentionally requires review of previously accepted ambiguous
joins; it does not infer that every whitespace-free continuation is invalid.
The worker never inserts a newline or strips the source fields automatically.
Preformatted `text` retains its serializer. Both chat pairs and `messages` now
produce token IDs directly; TRL skips its plain-text EOS/tokenization path. This
avoids an extra EOS after templates ending with EOS plus a newline and avoids
adding a second BOS. Full-sequence training loss remains unchanged.

Validation runs before model weights are loaded. The dataset/completion receipts
and each checkpoint/final directory carry a `training_format` contract with the
mode, formatter version, resolved chat-template SHA-256, tokenizer base and
requested immutable revision. A changed contract cannot resume a checkpoint.
Chat modes require a matching checkpoint contract, including for `messages`
datasets. A legacy checkpoint without that metadata requires a new adapter
target and output directory; its missing contract cannot qualify a chat-format
change. Legacy text/verbatim serializers may resume after validation. Completed
targets still exit without new training.

The November 16 Lean v2 run completed 200 steps, but a bounded five-prompt paired
diagnostic found incomplete adapter responses in all five cases. Its first 2,000
rows also exposed missing prompt/completion boundary whitespace and suspicious
trailing attributes. **Do not repeat or promote that run from this example.**
Review the completion extraction, use a separately pinned corrected dataset and
new private target, then require held-out proof verification and comparison with
the base model before a quality claim or promotion. Chat serialization alone
does not repair an incorrectly extracted completion or establish improvement.

## Guarded launch template

Replace the reviewed revision, dataset, target and directory values below before
launching. They are intentionally not the completed Lean v2 lane's identifiers.

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
  --base-revision REVIEWED_BASE_COMMIT \
  --dataset REVIEWED_CORRECTED_DATASET --dataset-train-file train.jsonl \
  --dataset-revision REVIEWED_DATASET_COMMIT \
  --prompt-completion-format chat \
  --adapter-out NEW_PRIVATE_ADAPTER_TARGET \
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

The existing `complete_record.py` is scoped to the original Lean v2 lane. Do not
reuse it for a corrected replacement target until its destination contract has
been updated and reviewed; preserve the completed lane's receipt. For its
original scope, invoke it only after the supervisor exits.
Supply `--launch-spec`, `--worker-log`, `--supervisor-state`, and `--output-dir`.
The launch specification contains `run_id`, `host`, `source_commit`, a
`source_files` mapping of file names to SHA-256 hashes, `base` and `dataset`
objects containing repository and immutable revision, `adapter_repo`, and
`max_steps` (200). The publisher binds the fresh log and terminal state, checks
the private adapter's pinned checkpoint metadata, and verifies exact-commit
publication to the existing Lean lane receipt path. Model quality, execution
provenance, tensor verification, and promotion remain separate qualifications.
It retains local evidence and returns failure if publication cannot be verified.

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

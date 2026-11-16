#!/usr/bin/env python3
"""salad_worker.py — the resumable burst-training worker (v3.7, 2026-11-16).

How NVIDIA et al. run training on spot-style fleets (Salad, Lambda on-demand,
k8s spot pools):
  1. Pre-built image — every dependency baked in; no pip at runtime.
  2. Durable checkpoints — every save_steps, weights + optimizer state go to
     object storage (here: the HF Hub), so a preempted node loses nothing.
  3. Resume-on-start — the worker checks for the latest checkpoint and
     continues from it; jobs are idempotent.
  4. Graceful preemption — SIGTERM handler checkpoints before exit.
  5. Receipts — boot/step/complete receipts to the chain; silence is an alarm.

v2 fixed the 2026-11-14 lane failures (F1 LazyRow serialization, F2
multi-config datasets, F3 split-schema mismatch, F4 no-kernel-image GPUs ->
fail-closed arch precheck, F5 stub datasets -> --min-examples guard).

v3 (this) adds MEASURED energy receipts — the leader method (CodeCarbon):
read the cumulative NVML energy counter (nvmlDeviceGetTotalEnergyConsumption,
mJ) at boot and at completion; report the device-level delta in joules.
Power-only samples are NEVER labeled measured energy. If pynvml/NVML or the
energy counter is unavailable, the receipt says UNAVAILABLE — never a
fabricated joule. Counter resets (driver reload) are detected: a negative
delta is reported UNAVAILABLE with a note, not a wrapped-around number.

v3.1 adds F6 - the resume-past-complete honesty guard. A Salad respawn of a
finished lane resumes at global_step >= max_steps; HF Trainer then runs stray
epoch-boundary steps and saves a junk checkpoint every step, and the worker
printed a fake training-complete receipt for a no-op boot (observed
2026-11-15: resumed checkpoint-200 -> 18 stray steps -> checkpoint-218, and a
"complete in 12.2s, train_loss 0" receipt). Now the worker reads the resumed
checkpoint trainer_state.json BEFORE loading the model: if
global_step >= max_steps it emits an already-complete receipt (with the
final adapter verified present), exits 0 without training, and never
fabricates completion metrics.

v3.2 adds a zeus-ml energy window - the ML.ENERGY-class measurement that
sits alongside the v3 NVML cumulative counter. A ZeusMonitor window opens
at boot and closes in the complete / already-complete receipts as its own
receipt field (zeus_window). Same doctrine: if zeus-ml cannot import or the
window cannot open/close, the field says UNAVAILABLE - joules are never
fabricated, and the v3 NVML fields are left untouched.

v3.3 adds F7 - object-JSON dataset support in the F3 direct-download fallback.
Some estate datasets publish a top-level JSON object (e.g. a proving-ground
atlas) instead of JSONL; the old fallback crashed on the first pretty-printed
line, turning the lane into a crash-restart loop with zero durable output
(observed 2026-11-15 on szl-train-formula-proving-ground-r1). The fallback
now accepts JSONL, a whole-file JSON list, or an object whose known keys
(atlas/rows/data/examples/items/train) hold the row list, and fails closed
with a receipt if none is usable.

v3.4 adds F8 - the dataset-cap honesty flag. The worker silently capped every
lane at 2000 examples (a memory guard from the burst lanes); a flagship mix of
10270 rows would have trained on a fifth of the corpus with no receipted trace.
The cap is now --max-examples (default 2000 = legacy behavior for existing
lanes), it is receipted in a dataset receipt BEFORE training starts, and
receipt uploads retry (3 attempts, 60s apart) so a transient Hub 429 cannot
silently drop a proof receipt.

v3.5 adds F9 - verified, honest resume-on-start. Resume existed since v1, but
every failure mode degraded SILENTLY to GENESIS: a repo-listing error returned
None, a partial local checkpoint dir (worker killed mid-save) shadowed the
durable Hub copy, and a failed checkpoint download crashed or restarted from
step 0. On a spot fleet that is an unbounded from-scratch respawn loop with
zero durable progress and no receipted trace (observed 2026-11-15 on the v4
lane: a respawn trained step 1 at full compute despite checkpoint-50 on the
Hub). Now: listing failures raise, partial local dirs are deleted and
re-downloaded from the Hub, downloads retry (3 attempts, 60s apart) then
fail closed with a receipt, and the boot receipt says resume_from
"checkpoint-N (verified)" vs "GENESIS (no checkpoints on hub)".

v3.6 adds exact terminal-step evidence. A preempted or early-stopped Trainer
must not publish a final adapter or report the requested budget as completed
work. Only an integer TrainerState.global_step equal to max_steps qualifies
this worker's completion branch. Existing checkpoint uploads remain available
for resumption; their durability is not asserted by this terminal guard.
Receipts retain the v1 kind and add a distinct event field. A resumed complete
checkpoint with a missing/unobservable final adapter returns failure, not success.

Fail-closed posture unchanged: private-repo verification before every upload,
receipts to the chain, SIGTERM checkpointing, pickle patterns excluded from
published adapters. Visibility flips remain human decisions.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, re, signal, sys, time

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/hf-cache")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from huggingface_hub import HfApi

DS_RECEIPTS = "SZLHOLDINGS/frontier-54-runs"
STOP = {"flag": False}
# Trainer resume state is pickle-based (torch.save / pickle). It may live in the PRIVATE
# working repo for resume, but must never be part of a published adapter.
PICKLE_PATTERNS = ["*.pt", "*.pth", "*.bin", "*.pkl", "*.pickle", "*.ckpt",
                   "optimizer.*", "scheduler.*", "rng_state*", "training_args.bin"]


def receipt(kind: str, detail: str, extra: dict | None = None):
    # Extra measurements cannot replace the receipt envelope or inject an old hash.
    r = {**(extra or {}), "kind": "szl.salad-worker/v1", "event": kind,
         "detail": detail, "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    r.pop("receipt_sha256", None)
    blob = json.dumps(r, sort_keys=True)
    r["receipt_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    print(json.dumps(r), flush=True)
    tok = os.environ.get("HF_TOKEN", "")
    if tok:
        for attempt in range(3):  # v3.4: a transient Hub 429 must not drop a proof receipt
            try:
                HfApi(token=tok).upload_file(
                    path_or_fileobj=json.dumps(r, indent=2).encode(),
                    path_in_repo=f"mesh/salad-{r['receipt_sha256'][:12]}.json",
                    repo_id=DS_RECEIPTS, repo_type="dataset")
                break
            except Exception as e:
                if attempt == 2:
                    print(f"receipt upload failed after retries (training continues): {str(e)[:80]}", flush=True)
                else:
                    time.sleep(60)
    return r


def on_term(signum, frame):
    if not STOP["flag"]:
        STOP["flag"] = True
        # Signal handlers cannot spend the shutdown grace on network retries.
        # The supervisor and normal terminal flow record the interruption.
        STOP["signal"] = signum
signal.signal(signal.SIGTERM, on_term)
signal.signal(signal.SIGINT, on_term)


class VisibilityError(RuntimeError):
    """Raised when a repo that must hold pickle-bearing checkpoints is not verifiably private."""


class ResumeError(RuntimeError):
    """F9: raised when resume-on-start cannot be verified — never degrade silently to GENESIS."""


def verify_private(api, repo_id: str, repo_type: str = "model") -> None:
    """Fail closed unless the Hub reports this repo as private RIGHT NOW."""
    try:
        info = api.repo_info(repo_id=repo_id, repo_type=repo_type)
    except Exception as e:  # inaccessible, deleted, transient error -> unknown -> refuse
        raise VisibilityError(f"cannot verify visibility of {repo_id}: {type(e).__name__}") from None
    if getattr(info, "private", None) is not True:
        raise VisibilityError(
            f"{repo_id} is not private (private={getattr(info, 'private', None)!r}); refusing upload. "
            "Changing visibility is a human decision.")


def ensure_private_repo(api, repo_id: str, token: str, repo_type: str = "model") -> None:
    api.create_repo(repo_id, exist_ok=True, private=True, token=token, repo_type=repo_type)
    verify_private(api, repo_id, repo_type)


def latest_checkpoint(api: HfApi, repo: str) -> str | None:
    """The resume-on-start contract: find the newest checkpoint-* revision.

    F9: a listing failure is never silently folded into GENESIS — it raises so
    the boot receipt can say 'could not check' instead of pretending the lane
    has no checkpoints. On a spot fleet a silent GENESIS is an unbounded
    from-scratch respawn loop with zero durable progress.
    """
    try:
        files = api.list_repo_files(repo_id=repo, repo_type="model")
    except Exception as e:
        raise ResumeError(f"cannot list checkpoints of {repo}: {type(e).__name__}: {str(e)[:120]}") from None
    stops = sorted(int(f.split("/")[0].split("-")[-1]) for f in files
                   if f.startswith("checkpoint-") and "/" in f
                   and f.split("/")[0].split("-")[-1].isdigit())
    return f"checkpoint-{stops[-1]}" if stops else None


def fetch_resume_checkpoint(api: HfApi, repo: str, output_dir: str) -> str | None:
    """F9: verified, honest resume-on-start.

    The Hub listing is authoritative. A LOCAL checkpoint dir is trusted only
    if its trainer_state.json is readable — a worker killed mid-save leaves a
    partial dir that would otherwise silently shadow the durable Hub copy and
    force a from-scratch restart (observed 2026-11-15 on the v4 lane: a
    respawn trained step 1 at full compute despite checkpoint-50 on the Hub).
    Partial local dirs are deleted and re-downloaded; a download that still
    fails after retries raises instead of degrading into silent GENESIS.
    """
    ck = latest_checkpoint(api, repo)  # raises ResumeError on listing failure
    if not ck:
        return None
    local = os.path.join(output_dir, ck)
    if os.path.isdir(local) and resume_global_step(local) is None:
        import shutil
        shutil.rmtree(local, ignore_errors=True)
        receipt("warn", f"local {ck} had no readable trainer_state.json "
                "(killed mid-save?) — deleted; re-downloading the durable Hub copy")
    if not os.path.isdir(local):
        for attempt in range(3):
            try:
                api.snapshot_download(repo_id=repo, repo_type="model",
                                      allow_patterns=[f"{ck}/*"], local_dir=output_dir)
                break
            except Exception as e:
                if attempt == 2:
                    raise ResumeError(f"download of {ck} failed after 3 attempts: {str(e)[:120]}") from None
                time.sleep(60)
    gs = resume_global_step(local)
    if gs is None:
        raise ResumeError(f"{ck} present but trainer_state.json unreadable — refusing silent GENESIS")
    return local


def resume_global_step(resume_dir: str) -> int | None:
    """F6: read the resumed checkpoint's global_step without loading weights."""
    try:
        with open(os.path.join(resume_dir, "trainer_state.json"), encoding="utf-8") as f:
            return int(json.load(f).get("global_step", 0))
    except Exception:
        return None


def _compatible_cuda_cubin(cap: tuple[int, int], arch_list: list[str]) -> str | None:
    """Choose a generic cubin using NVIDIA's same-major, nondecreasing-minor rule.

    In particular, sm_86 is valid on an Ada sm_89 GPU. Do not infer generic
    compatibility from architecture/family-specific a/f suffixes. PTX-only
    builds need separate qualification; a missing match means unqualified,
    not proof that the runtime could never execute that build.
    https://docs.nvidia.com/cuda/ada-compatibility-guide/
    """
    candidates = []
    for arch in arch_list:
        match = re.fullmatch(r"sm_([0-9]+)", arch)
        if match:
            target = int(match.group(1))
            if target // 10 == cap[0] and target % 10 <= cap[1]:
                candidates.append((target, arch))
    return max(candidates)[1] if candidates else None


def _cuda_bf16_smoke(device_index: int) -> dict:
    """Exercise this worker's dtype, native kernels, GEMM, and autograd before load.

    All constants and expected results are exactly representable in BF16.
    This bounded probe is runtime evidence, not whole-model qualification.
    """
    with torch.cuda.device(device_index), torch.enable_grad():
        a = torch.full((32, 32), 0.125, device=f"cuda:{device_index}",
                       dtype=torch.bfloat16, requires_grad=True)
        b = torch.full((32, 32), 0.25, device=f"cuda:{device_index}",
                       dtype=torch.bfloat16, requires_grad=True)
        loss = (a @ b).float().square().mean()
        loss.backward()
        # Launches are asynchronous: a queued kernel is not evidence of success.
        torch.cuda.synchronize(device_index)
        values = ((loss, 1.0), (a.grad, 1.0 / 64), (b.grad, 1.0 / 128))
        for value, expected in values:
            if value is None or not bool(torch.isfinite(value).all().item()):
                raise RuntimeError("BF16 forward/backward produced non-finite or missing values")
            if not bool((value == expected).all().item()):
                raise RuntimeError("BF16 forward/backward failed the known-result check")
        return {"dtype": "bfloat16", "forward_backward": "PASS", "loss": loss.item()}


def cuda_arch_precheck() -> None:
    """F4: qualify generic cubins AND real BF16 execution on every visible GPU."""
    evidence = {"torch_version": torch.__version__,
                "cuda_arch_policy": "generic-cubin-same-major-plus-bf16-smoke",
                "devices": []}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable — refusing to train on CPU")
        arch_list = torch.cuda.get_arch_list()
        evidence["arch_list"] = arch_list
        device_count = torch.cuda.device_count()
        if device_count < 1:
            raise RuntimeError("CUDA reports no visible devices")
        # device_map='auto' can place model layers on any visible GPU.
        for index in range(device_count):
            cap = torch.cuda.get_device_capability(index)
            device = {"index": index, "gpu": torch.cuda.get_device_name(index),
                      "sm": f"sm_{cap[0]}{cap[1]}",
                      "compatible_cubin": _compatible_cuda_cubin(cap, arch_list)}
            evidence["devices"].append(device)
            if device["compatible_cubin"] is None:
                raise RuntimeError(
                    f"GPU {index} ({device['gpu']}, {device['sm']}) has no qualified "
                    "generic cubin target in this torch build; PTX-only and "
                    "architecture-specific builds require separate qualification")
            device["bf16_smoke"] = _cuda_bf16_smoke(index)
    except Exception as error:
        receipt("fail-closed", f"CUDA training precheck failed: {error}", evidence)
        raise SystemExit(4) from error
    receipt("cuda-precheck", "Generic cubins and BF16 forward/backward verified "
            "on all visible GPUs; whole-model qualification still required", evidence)


def configure_shared_gpu_budget(budget_mib: int, reserve_mib: int) -> dict:
    """Admit one visible GPU and cap this process's PyTorch caching allocator.

    This is not a hardware partition or a limit on non-PyTorch allocations.
    The external supervisor must continue checking serving health/headroom.
    """
    if type(budget_mib) is not int or budget_mib <= 0:
        raise ValueError("GPU budget must be a positive integer MiB value")
    if type(reserve_mib) is not int or reserve_mib < 0:
        raise ValueError("GPU reserve must be a nonnegative integer MiB value")
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("shared-GPU admission requires exactly one visible CUDA device")
    free, total = torch.cuda.mem_get_info(0)
    budget, reserve = budget_mib * 1024**2, reserve_mib * 1024**2
    if total <= 0 or budget + reserve > free:
        raise RuntimeError("insufficient free GPU memory for budget plus serving reserve")
    torch.cuda.set_per_process_memory_fraction(budget / total, device=0)
    evidence = {"allocator_budget_mib": budget_mib, "serving_reserve_mib": reserve_mib,
                "free_before_mib": free // 1024**2, "total_mib": total // 1024**2,
                "scope": "PYTORCH_CACHING_ALLOCATOR_NOT_HARDWARE_PARTITION"}
    receipt("gpu-admission", "Shared GPU memory admission passed; serving health "
            "must remain supervised", evidence)
    return evidence


def _ensure_pynvml():
    """Best-effort pynvml (NVML bindings) for MEASURED joules; UNAVAILABLE is honest."""
    try:
        import pynvml  # type: ignore
        return pynvml
    except Exception:
        if os.environ.get("SZL_DISABLE_RUNTIME_INSTALL") == "1":
            return None
        try:
            import subprocess as _sp
            _sp.run([sys.executable, "-m", "pip", "install", "-q", "pynvml"],
                    timeout=120, check=False)
            import pynvml  # type: ignore
            return pynvml
        except Exception:
            return None


_NVML = None
_NVML_TRIED = False


def sample_energy_j() -> dict:
    """v3: leader-method (CodeCarbon) cumulative NVML energy counter, device level.

    Returns {"label": "MEASURED"|"UNAVAILABLE", "energy_j": float|None, "gpu": str|None}.
    Power-only samples are NOT labeled measured energy; a missing counter is UNAVAILABLE.
    """
    global _NVML, _NVML_TRIED
    try:
        if _NVML is None and not _NVML_TRIED:
            _NVML_TRIED = True
            _NVML = _ensure_pynvml()
        if _NVML is None:
            return {"label": "UNAVAILABLE", "energy_j": None, "gpu": None,
                    "note": "pynvml unavailable; joules NOT fabricated"}
        _NVML.nvmlInit()
        h = _NVML.nvmlDeviceGetHandleByIndex(0)
        e = _NVML.nvmlDeviceGetTotalEnergyConsumption(h) / 1000.0  # mJ -> J
        name = _NVML.nvmlDeviceGetName(h)
        name = name.decode() if isinstance(name, bytes) else name
        return {"label": "MEASURED", "energy_j": float(e), "gpu": name}
    except Exception as e:
        return {"label": "UNAVAILABLE", "energy_j": None, "gpu": None,
                "note": f"NVML energy counter unavailable ({type(e).__name__}); joules NOT fabricated"}


def energy_delta(start: dict, end: dict) -> dict:
    """Device-level joules between two MEASURED samples; negative delta = counter reset -> UNAVAILABLE."""
    if start.get("label") == "MEASURED" and end.get("label") == "MEASURED":
        a, b = start.get("energy_j"), end.get("energy_j")
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            if b >= a:
                return {"label": "MEASURED", "joules": round(b - a, 1),
                        "method": "nvmlDeviceGetTotalEnergyConsumption cumulative-counter delta (device level)"}
            return {"label": "UNAVAILABLE", "joules": None,
                    "note": "NVML energy counter reset mid-run (driver reload?); delta NOT fabricated"}
    return {"label": "UNAVAILABLE", "joules": None,
            "note": "energy counters unavailable at start or end; joules NOT fabricated"}


_ZEUS_STATE = {"monitor": None, "window": False, "tried": False}


def _ensure_zeus():
    """Best-effort zeus-ml import (mirrors the pynvml fallback); None is honest."""
    try:
        from zeus.monitor import ZeusMonitor  # type: ignore
        return ZeusMonitor
    except Exception:
        if os.environ.get("SZL_DISABLE_RUNTIME_INSTALL") == "1":
            return None
        if not _ZEUS_STATE["tried"]:
            _ZEUS_STATE["tried"] = True
            try:
                import subprocess as _sp
                _sp.run([sys.executable, "-m", "pip", "install", "-q", "zeus-ml"],
                        timeout=240, check=False)
                from zeus.monitor import ZeusMonitor  # type: ignore
                return ZeusMonitor
            except Exception:
                return None
        return None


def zeus_begin_window() -> dict:
    """v3.2: open a ZeusMonitor window around training; UNAVAILABLE is honest."""
    try:
        ZeusMonitor = _ensure_zeus()
        if ZeusMonitor is None:
            return {"label": "UNAVAILABLE", "joules": None,
                    "note": "zeus-ml unavailable; joules NOT fabricated"}
        mon = ZeusMonitor(gpu_indices=[0])
        mon.begin_window("szl_train")
        _ZEUS_STATE["monitor"] = mon
        _ZEUS_STATE["window"] = True
        return {"label": "MEASURED", "window": "szl_train",
                "method": "zeus-ml ZeusMonitor window (begin at boot)"}
    except Exception as e:
        _ZEUS_STATE["monitor"] = None
        _ZEUS_STATE["window"] = False
        return {"label": "UNAVAILABLE", "joules": None,
                "note": f"zeus window could not open ({type(e).__name__}); joules NOT fabricated"}


def zeus_end_window() -> dict:
    """v3.2: close the window and report window joules; UNAVAILABLE is honest."""
    mon = _ZEUS_STATE.get("monitor")
    if mon is None or not _ZEUS_STATE.get("window"):
        return {"label": "UNAVAILABLE", "joules": None,
                "note": "no zeus window open; joules NOT fabricated"}
    try:
        m = mon.end_window("szl_train")
        _ZEUS_STATE["window"] = False
        j = getattr(m, "total_energy", None)
        if isinstance(j, (int, float)) and j >= 0:
            return {"label": "MEASURED", "joules": round(float(j), 1),
                    "method": "zeus-ml ZeusMonitor window total_energy"}
        return {"label": "UNAVAILABLE", "joules": None,
                "note": "zeus window closed but total_energy unusable; NOT fabricated"}
    except Exception as e:
        _ZEUS_STATE["window"] = False
        return {"label": "UNAVAILABLE", "joules": None,
                "note": f"zeus window close failed ({type(e).__name__}); joules NOT fabricated"}


def load_train_split(a, api: HfApi):
    """F2/F3: config-aware, split-safe train loading with train-only fallback."""
    kwargs = {"split": "train"}
    if getattr(a, "dataset_revision", ""):
        kwargs["revision"] = a.dataset_revision
    if a.dataset_train_file:
        kwargs["data_files"] = {"train": a.dataset_train_file}
    if a.dataset_config:
        kwargs["name"] = a.dataset_config
    try:
        return load_dataset(a.dataset, **kwargs)
    except ValueError as e:
        # F2: multi-config dataset without a config
        msg = str(e)
        if "available configs" in msg and "pick one" in msg:
            import re as _re
            first = msg.split("available configs:")[-1].splitlines()[0]
            cfgs = [c for c in _re.findall(r"'([^']+)'", first) if "/" not in c]
            receipt("fail-closed",
                    f"dataset {a.dataset} needs --dataset-config; available: {cfgs}",
                    {"configs": cfgs})
            raise SystemExit(5)
        raise
    except Exception as e:
        # F3: split-schema mismatch — retry train-only from the Hub's train.jsonl
        receipt("warn", f"dataset generation failed ({str(e)[:100]}); retrying train-only load")
        train_file = a.dataset_train_file or "train.jsonl"
        try:
            # A pinned Hub revision must never fall back to an unrelated local file.
            local_train_file = train_file
            if getattr(a, "dataset_revision", ""):
                from huggingface_hub import hf_hub_download
                local_train_file = hf_hub_download(
                    repo_id=a.dataset, repo_type="dataset", filename=train_file,
                    token=a.hf_token, revision=a.dataset_revision)
            return load_dataset("json", data_files={"train": local_train_file},
                                split="train", token=a.hf_token)
        except Exception:
            from huggingface_hub import hf_hub_download
            from datasets import Dataset as _Dataset
            receipt("warn", f"loading {train_file} directly from the Hub")
            p = hf_hub_download(repo_id=a.dataset, repo_type="dataset",
                                filename=train_file, token=a.hf_token,
                                revision=getattr(a, "dataset_revision", "") or None)
            rows = None
            try:
                rows = [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]
            except json.JSONDecodeError:
                with open(p, encoding="utf-8") as f:
                    obj = json.load(f)
                if isinstance(obj, list):
                    rows = obj
                elif isinstance(obj, dict):
                    for k in ("atlas", "rows", "data", "examples", "items", "train"):
                        if isinstance(obj.get(k), list):
                            rows = obj[k]
                            receipt("warn", "object-JSON dataset; using list under key " + k + " (" + str(len(rows)) + " rows)")
                            break
                    if rows is None:
                        rows = next((v for v in obj.values() if isinstance(v, list)), None)
                        if rows is not None:
                            receipt("warn", "object-JSON dataset; using first list-valued field (" + str(len(rows)) + " rows)")
            if not rows:
                receipt("fail-closed", "no usable rows in " + train_file + " (expected JSONL, JSON list, or keyed list)")
                raise SystemExit(5)
            return _Dataset.from_list(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--base-revision", default="",
                    help="optional immutable base model revision")
    ap.add_argument("--dataset-revision", default="",
                    help="optional immutable dataset revision")
    ap.add_argument("--adapter-out", required=True)
    ap.add_argument("--max-steps", type=int, default=200)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--max-examples", type=int, default=2000,
                    help="F8: cap on examples per lane (default 2000 = legacy cap); "
                         "raise to train on the full mix")
    ap.add_argument("--min-examples", type=int, default=8,
                    help="F5 honesty guard: refuse to train on stub-sized data")
    ap.add_argument("--dataset-config", default="",
                    help="F2: config name for multi-config datasets")
    ap.add_argument("--dataset-train-file", default="",
                    help="F3: explicit train data file for split-schema-broken datasets")
    ap.add_argument("--hf-token", default=os.environ.get("HF_TOKEN", ""))
    ap.add_argument("--publish-final-to", default="",
                    help="optional separate repo for the final adapter (safetensors/config/tokenizer only)")
    ap.add_argument("--output-dir", default="~/ckpt",
                    help="dedicated local checkpoint directory for this lane")
    ap.add_argument("--gradient-checkpointing", action="store_true",
                    help="recompute activations to reduce training memory")
    ap.add_argument("--gpu-memory-budget-mib", type=int, default=0,
                    help="opt-in PyTorch allocator cap; requires one visible GPU")
    ap.add_argument("--gpu-reserve-mib", type=int, default=8192,
                    help="additional free memory required at shared-GPU admission")
    a = ap.parse_args()
    if min(a.max_steps, a.save_steps, a.max_examples, a.min_examples) < 1:
        ap.error("step, save, example and minimum-example bounds must be positive")
    if a.max_examples < a.min_examples:
        ap.error("max-examples cannot be smaller than min-examples")
    if a.gpu_memory_budget_mib < 0 or a.gpu_reserve_mib < 0:
        ap.error("GPU budget and reserve cannot be negative")
    if not a.output_dir.strip():
        ap.error("output-dir cannot be empty")
    for revision in (a.base_revision, a.dataset_revision):
        if revision and not re.fullmatch(r"[0-9a-f]{40}", revision):
            ap.error("explicit revisions must be immutable 40-character commit SHAs")
    a.output_dir = os.path.expanduser(a.output_dir)
    energy_scope = ("DEVICE_TOTAL_INCLUDES_CORESIDENT_WORKLOADS" if a.gpu_memory_budget_mib
                    else "DEVICE_TOTAL_NOT_JOB_ATTRIBUTION")
    if not a.hf_token:
        print("FAIL-CLOSED: HF_TOKEN required (env or --hf-token)", file=sys.stderr)
        return 2

    api = HfApi(token=a.hf_token)
    # The working repo holds resumable (pickle-bearing) checkpoints, so it is created PRIVATE.
    # create_repo(exist_ok=True) leaves an EXISTING public repo public, so verify explicitly.
    try:
        ensure_private_repo(api, a.adapter_out, a.hf_token)
    except VisibilityError as e:
        print(f"FAIL-CLOSED: {e}", file=sys.stderr)
        return 3
    try:
        resume = fetch_resume_checkpoint(api, a.adapter_out, a.output_dir)
    except ResumeError as e:
        # F9 fail-closed: never silently restart from step 0 when resume
        # infrastructure is broken — that is an unbounded spot-GPU burn loop.
        receipt("fail-closed", f"resume infrastructure failed — refusing silent "
                f"from-scratch restart: {e}")
        print(f"FAIL-CLOSED: resume failed: {e}", file=sys.stderr)
        return 7
    energy_start = sample_energy_j()  # v3: cumulative counter at boot
    zeus_start = zeus_begin_window()  # v3.2: ML.ENERGY-class window at boot
    receipt("boot", f"worker started — base={a.base} dataset={a.dataset}"
            + (f" config={a.dataset_config}" if a.dataset_config else ""),
            {"resume_from": (os.path.basename(resume) + " (verified)" if resume
                             else "GENESIS (no checkpoints on hub)"),
             "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
             "node": os.environ.get("HOSTNAME", "unknown"),
             "worker_version": "v3.7-20261116",
             "base_revision": a.base_revision or None,
             "dataset_revision": a.dataset_revision or None,
             "energy_start": energy_start,
             "energy_scope": energy_scope,
             "zeus_window": zeus_start})

    # F6: resume-past-complete honesty guard. Salad respawns a container whose
    # command exited; a finished lane would otherwise resume at global_step >=
    # max_steps, run stray epoch-boundary steps, push a junk checkpoint every
    # step, and print a fake completion receipt. Exit honestly instead.
    if resume:
        rgs = resume_global_step(resume)
        if rgs is not None and rgs >= a.max_steps:
            energy_end = sample_energy_j()
            final_present = False
            try:
                verify_private(api, a.adapter_out)
                final_present = "final/adapter_model.safetensors" in api.list_repo_files(
                    a.adapter_out, repo_type="model")
            except Exception:
                final_present = False
            receipt("already-complete" if final_present else "incomplete",
                    f"resumed checkpoint global_step={rgs} >= max_steps={a.max_steps} — "
                    "training already complete on this lane; no new steps this boot"
                    + ("" if final_present else "; FINAL ADAPTER MISSING — investigate"),
                    {"resume_global_step": rgs, "max_steps": a.max_steps,
                     "final_adapter_present": final_present,
                     "new_steps_this_boot": 0,
                     "completion_qualified": False,
                     "scope": "RESUME_TERMINAL_METADATA_NOT_NEW_TRAINING",
                     "energy_scope": energy_scope,
                     "zeus_window": zeus_end_window(),
                     "energy_start": energy_start, "energy_end": energy_end,
                     "energy_j": energy_delta(energy_start, energy_end)})
            if not final_present:
                print("FAIL-CLOSED: terminal checkpoint has no verified-present final adapter; "
                      "refusing completion or automatic retraining", file=sys.stderr, flush=True)
                return 7
            print(f"ALREADY-COMPLETE: global_step {rgs} >= max_steps {a.max_steps}; "
                  "exiting without training", flush=True)
            return 0

    # A terminal resume does no training; probe CUDA only for an admitted run.
    if a.gpu_memory_budget_mib:
        try:
            configure_shared_gpu_budget(a.gpu_memory_budget_mib, a.gpu_reserve_mib)
        except Exception as error:
            receipt("fail-closed", f"Shared GPU admission failed: {error}")
            return 8
    cuda_arch_precheck()  # F4: must still precede model loading/training.
    base_kwargs = {"revision": a.base_revision} if a.base_revision else {}
    tok = AutoTokenizer.from_pretrained(a.base, **base_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, device_map="auto", **base_kwargs)
    if a.gradient_checkpointing:
        model.config.use_cache = False
    ds = load_train_split(a, api)  # F2/F3

    # F5 honesty guard: stub-sized "datasets" (spec files, READMEs) are not training data.
    if len(ds) < a.min_examples:
        receipt("fail-closed",
                f"dataset {a.dataset} has {len(ds)} examples (< {a.min_examples}) — "
                "refusing to train on stub data; publish a real mix first",
                {"examples": len(ds)})
        print(f"FAIL-CLOSED: {len(ds)} examples is stub-sized; not training", file=sys.stderr)
        return 6

    # F8: the cap is now an explicit, receipted argument (legacy default 2000).
    n_raw = len(ds)
    n_used = min(a.max_examples, n_raw)
    receipt("dataset", f"dataset ready — {n_raw} examples, training on {n_used}",
            {"examples": n_raw, "examples_used": n_used, "max_examples": a.max_examples})
    ds = ds.select(range(n_used))
    def fmt(ex):
        ex = dict(ex)  # F1: LazyRow -> plain dict; JSON-safe in every branch
        if "text" in ex: return {"text": ex["text"]}
        if "messages" in ex:
            return {"text": tok.apply_chat_template(ex["messages"], tokenize=False)}
        if "prompt" in ex and "completion" in ex:
            return {"text": ex["prompt"] + ex["completion"]}
        return {"text": json.dumps(ex)}
    ds = ds.map(fmt, remove_columns=ds.column_names)

    cfg = SFTConfig(
        output_dir=a.output_dir,
        max_length=2048,
        per_device_train_batch_size=1,          # spot-metal safe: fits 24GB cards
        gradient_accumulation_steps=16,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=20,
        max_steps=a.max_steps,
        save_steps=a.save_steps, save_total_limit=2,
        bf16=True, optim="adamw_torch", report_to=[], logging_steps=10,
        gradient_checkpointing=a.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if a.gradient_checkpointing else None,
        packing=False)
    peft = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type="CAUSAL_LM")

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft,
                         processing_class=tok)

    from transformers import TrainerCallback
    class ReceiptCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            if STOP["flag"]:
                # Request a resumable save at the next optimizer step. This is
                # best-effort: the supervisor may exhaust its shutdown grace.
                control.should_save = True
                control.should_training_stop = True
            return control

        def on_save(self, args, state, control, **kw):
            # push the checkpoint to durable storage immediately
            ck = f"checkpoint-{state.global_step}"
            path = os.path.join(args.output_dir, ck)
            if os.path.exists(path):
                try:
                    # re-check every time: visibility can change while training runs
                    verify_private(api, a.adapter_out)
                except VisibilityError as e:
                    receipt("checkpoint-blocked", f"upload refused: {str(e)[:120]}")
                    STOP["flag"] = True
                    STOP["visibility"] = True
                    control.should_training_stop = True
                    return
                try:
                    api.upload_folder(folder_path=path, path_in_repo=ck,
                        repo_id=a.adapter_out, repo_type="model")
                    receipt("checkpoint", f"checkpoint {ck} pushed to durable storage")
                except Exception as e:
                    receipt("checkpoint-fail", f"push failed: {str(e)[:80]}")
            if STOP["flag"]:
                control.should_training_stop = True
    trainer.add_callback(ReceiptCallback())

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume) if resume else trainer.train()
    observed_steps = getattr(getattr(trainer, "state", None), "global_step", None)
    actual_steps = observed_steps if type(observed_steps) is int and observed_steps >= 0 else None
    if actual_steps != a.max_steps:
        # Stop signals, early exits, invalid state and overshoot are not completion.
        # Do not overwrite final/ with a partial adapter. Checkpoint preservation
        # remains the callback's responsibility; this branch cannot prove durability.
        energy_end = sample_energy_j()
        receipt("interrupted" if STOP["flag"] else "incomplete",
                "training did not reach the exact admitted target; final publication withheld",
                {"actual_steps": actual_steps, "target_steps": a.max_steps,
                 "termination_signal": STOP.get("signal"),
                 "completion_qualified": False, "final_publication_attempted": False,
                 "checkpoint_durability": "UNVERIFIED_BY_TERMINAL_GUARD",
                 "energy_scope": energy_scope,
                 "energy_start": energy_start, "energy_end": energy_end,
                 "zeus_window": zeus_end_window(),
                 "energy_j": energy_delta(energy_start, energy_end)})
        print("FAIL-CLOSED: incomplete or unqualified trainer state; final adapter not published",
              file=sys.stderr, flush=True)
        return 7
    final = os.path.join(a.output_dir, "final")
    trainer.save_model(final)
    if STOP.get("visibility"):
        print("FAIL-CLOSED: working repo visibility could not be verified; final adapter kept local only",
              file=sys.stderr)
        return 3
    try:
        verify_private(api, a.adapter_out)
    except VisibilityError as e:
        print(f"FAIL-CLOSED: {e}; final adapter kept local only", file=sys.stderr)
        return 3
    api.upload_folder(folder_path=final, path_in_repo="final",
        repo_id=a.adapter_out, repo_type="model", ignore_patterns=PICKLE_PATTERNS)
    if a.publish_final_to:
        try:
            ensure_private_repo(api, a.publish_final_to, a.hf_token)
        except VisibilityError as e:
            print(f"FAIL-CLOSED: {e}; final adapter not published", file=sys.stderr)
            return 3
        api.upload_folder(folder_path=final, repo_id=a.publish_final_to, repo_type="model",
                          ignore_patterns=PICKLE_PATTERNS)  # visibility change stays a human decision
    energy_end = sample_energy_j()  # v3: cumulative counter at completion
    receipt("complete", f"training complete in {round(time.time()-t0,1)}s",
            {"steps": actual_steps, "actual_steps": actual_steps,
             "target_steps": a.max_steps, "completion_qualified": True,
             "completion_scope": "EXACT_TRAINER_STEP_TARGET_NOT_MODEL_QUALITY",
             "energy_scope": energy_scope,
             "adapter": a.adapter_out,
             "energy_start": energy_start,
             "energy_end": energy_end,
             "zeus_window": zeus_end_window(),
             "energy_j": energy_delta(energy_start, energy_end)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

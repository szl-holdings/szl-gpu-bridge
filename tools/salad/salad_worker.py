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

v3.7 requires receipt publication and byte readback at the returned Hub commit.
The channel is explicit (the existing dataset, or an explicitly selected PRIVATE
model repo); credentials are never widened and there is no destination fallback.
Boot admission precedes model loading, heartbeat loss requests a checkpoint and
stops training, and a missing final receipt returns exit 8 without claiming success.
Local pending bytes are recovery evidence, not proof of publication.
"""
from __future__ import annotations
import argparse, datetime, hashlib, json, os, signal, sys, time
from pathlib import Path
import re
import uuid

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME", os.path.expanduser("~/hf-cache")))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer
from peft import LoraConfig
from datasets import load_dataset
from huggingface_hub import HfApi

DS_RECEIPTS = "SZLHOLDINGS/frontier-54-runs"
STOP = {"flag": False}
RECEIPTS = None
# Trainer resume state is pickle-based (torch.save / pickle). It may live in the PRIVATE
# working repo for resume, but must never be part of a published adapter.
PICKLE_PATTERNS = ["*.pt", "*.pth", "*.bin", "*.pkl", "*.pickle", "*.ckpt",
                   "optimizer.*", "scheduler.*", "rng_state*", "training_args.bin"]


class ReceiptError(RuntimeError):
    """The current receipt cannot be proven published; never implies completion."""


class ArtifactError(RuntimeError):
    """A final adapter upload failed; provider exception bodies stay private."""


def hub_revision(commit) -> str:
    revision = getattr(commit, "oid", None)
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-f]{40}", revision) is None:
        raise ReceiptError("Hub did not return an immutable commit revision")
    return revision


def transport_label(error: Exception) -> str:
    # SDK exception strings can contain request headers or signed URLs.
    status = getattr(getattr(error, "response", None), "status_code", None)
    return f"HTTP {status}" if type(status) is int else type(error).__name__


def receipt_retry(operation, stage):
    for attempt in range(3):
        try:
            return operation()
        except ReceiptError:
            raise
        except Exception as error:
            status = getattr(getattr(error, "response", None), "status_code", None)
            # Permission/not-found/invalid-input failures require reconciliation,
            # not retries with another repository or stronger credentials.
            permanent = type(status) is int and 400 <= status < 500 and status not in (408, 429)
            if permanent or attempt == 2:
                raise ReceiptError(f"{stage} failed ({transport_label(error)})") from None
            time.sleep(5 * (attempt + 1))


class ReceiptChannel:
    """Publish current-run evidence only; no replay or historical receipt invention."""

    def __init__(self, api, token, repo_id, repo_type, outbox, adapter):
        if not token:
            raise ReceiptError("a receipt-channel token is required")
        if not isinstance(repo_id, str) or re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*", repo_id
        ) is None:
            raise ReceiptError("receipt repository must be an explicit owner/name")
        if repo_type not in ("dataset", "model"):
            raise ReceiptError("receipt repository type must be dataset or model")
        self.api, self.token = api, token
        self.repo_id, self.repo_type = repo_id, repo_type
        self.run_id = uuid.uuid4().hex
        self.adapter = adapter
        self.outbox = Path(outbox) / self.run_id
        self.verify_outbox()

    def verify_outbox(self):
        artifact_root = Path(os.path.expanduser("~/ckpt")).resolve()
        if self.outbox.resolve().is_relative_to(artifact_root):
            raise ReceiptError("receipt outbox must be outside the artifact directory tree")

    def verify_channel(self):
        if self.repo_type == "model":
            try:
                verify_private(self.api, self.repo_id, repo_type="model")
            except VisibilityError:
                raise ReceiptError("model receipt channel is not verified private") from None

    def publish(self, value):
        from huggingface_hub import hf_hub_download

        raw = (json.dumps(value, indent=2, allow_nan=False) + "\n").encode()
        digest = value["receipt_sha256"]
        path = f"mesh/salad-{digest}.json"
        self.verify_outbox()
        # Retain the exact attempted bytes, even if the server accepts a write but
        # its response is lost. A pending file is never a durable-publication claim.
        try:
            self.outbox.mkdir(parents=True, exist_ok=True)
            pending = self.outbox / f"{digest}.pending.json"
            with pending.open("xb") as stream:
                stream.write(raw)
        except FileExistsError:
            if pending.read_bytes() != raw:
                raise ReceiptError("local receipt candidate conflicts") from None
        except OSError:
            raise ReceiptError("cannot retain local receipt candidate") from None
        self.verify_channel()
        commit = receipt_retry(lambda: self.api.upload_file(
            path_or_fileobj=raw, path_in_repo=path,
            repo_id=self.repo_id, repo_type=self.repo_type,
            commit_message=f"Record Salad {value['event']} {digest[:12]}",
        ), "receipt upload")
        revision = hub_revision(commit)

        def readback():
            local = hf_hub_download(
                repo_id=self.repo_id, repo_type=self.repo_type, filename=path,
                revision=revision, token=self.token, force_download=True,
            )
            with open(local, "rb") as stream:
                observed = stream.read(len(raw) + 1)
            if observed != raw:
                raise ReceiptError("receipt byte readback does not match the attempted bytes")

        # Read the exact returned commit, never the moving default branch or an
        # unverified cached copy. A failed readback must not re-upload the receipt.
        receipt_retry(readback, "receipt readback")
        self.verify_channel()
        publication = {
            "kind": "szl.salad-receipt-publication/v1",
            "receipt_event": value["event"], "receipt_sha256": digest,
            "run_id": self.run_id, "repo_id": self.repo_id,
            "repo_type": self.repo_type, "path": path, "revision": revision,
            "file_sha256": hashlib.sha256(raw).hexdigest(),
            "publication_verified": True,
            "scope": "RECEIPT_BYTES_AT_COMMIT_NOT_MODEL_QUALITY_OR_FRESHNESS",
        }
        try:
            acknowledgement = self.outbox / f"{digest}.publication.json"
            temporary = acknowledgement.with_suffix(".tmp")
            temporary.write_text(json.dumps(publication, indent=2) + "\n", encoding="utf-8")
            os.replace(temporary, acknowledgement)
        except OSError:
            raise ReceiptError("cannot retain receipt publication acknowledgement") from None
        return publication


def build_receipt(kind: str, detail: str, extra: dict | None = None):
    # Extra measurements cannot replace the receipt envelope or inject an old hash.
    r = {**(extra or {}), "kind": "szl.salad-worker/v1", "event": kind,
         "detail": detail, "ts": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    r.pop("receipt_sha256", None)
    blob = json.dumps(r, sort_keys=True, allow_nan=False)
    r["receipt_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
    return r


def receipt(kind: str, detail: str, extra: dict | None = None):
    if RECEIPTS is None:
        raise ReceiptError("receipt channel has not been configured")
    value = build_receipt(kind, detail, {
        **(extra or {}), "run_id": RECEIPTS.run_id, "adapter": RECEIPTS.adapter,
        "receipt_destination": {"repo_id": RECEIPTS.repo_id, "repo_type": RECEIPTS.repo_type},
    })
    publication = RECEIPTS.publish(value)
    # Successful event output follows verified publication. On failure only the
    # local pending file exists; stdout cannot masquerade as a durable receipt.
    print(json.dumps(publication), flush=True)
    print(json.dumps(value), flush=True)
    return value


def on_term(signum, frame):
    if not STOP["flag"]:
        STOP["flag"] = True
        # Signal handlers cannot safely perform blocking Hub I/O. The Trainer
        # callback requests a save and emits terminal evidence in normal control.
        print("PREEMPTED: checkpoint requested at next Trainer step boundary", flush=True)
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
    try:
        api.create_repo(repo_id, exist_ok=True, private=True, token=token, repo_type=repo_type)
    except Exception as error:
        raise VisibilityError(f"cannot prepare private repository ({transport_label(error)})") from None
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
        raise ResumeError(f"cannot list checkpoints of {repo}: {transport_label(e)}") from None
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
                    raise ResumeError(f"download of {ck} failed after 3 attempts: {transport_label(e)}") from None
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


def cuda_arch_precheck() -> None:
    """F4: fail-closed BEFORE training if this torch build lacks kernels for the GPU."""
    if not torch.cuda.is_available():
        receipt("fail-closed", "CUDA unavailable — refusing to train on CPU")
        raise SystemExit(4)
    cap = torch.cuda.get_device_capability(0)
    sm = f"sm_{cap[0]}{cap[1]}"
    arch_list = torch.cuda.get_arch_list()
    if sm not in arch_list:
        gpu = torch.cuda.get_device_name(0)
        receipt("fail-closed",
                f"torch {torch.__version__} lacks {sm} kernels for {gpu} "
                f"(arch_list={arch_list}) — pin gpu_classes or use a newer CUDA build",
                {"gpu": gpu, "sm": sm})
        raise SystemExit(4)


def _ensure_pynvml():
    """Best-effort pynvml (NVML bindings) for MEASURED joules; UNAVAILABLE is honest."""
    try:
        import pynvml  # type: ignore
        return pynvml
    except Exception:
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
        receipt("warn", f"dataset generation failed ({transport_label(e)}); retrying train-only load")
        train_file = a.dataset_train_file or "train.jsonl"
        try:
            return load_dataset("json", data_files={"train": train_file},
                                split="train", token=a.hf_token)
        except Exception:
            from huggingface_hub import hf_hub_download
            from datasets import Dataset as _Dataset
            receipt("warn", f"loading {train_file} directly from the Hub")
            p = hf_hub_download(repo_id=a.dataset, repo_type="dataset",
                                filename=train_file, token=a.hf_token)
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
    ap.add_argument("--receipts-repo", default=os.environ.get("SZL_RECEIPTS_REPO", DS_RECEIPTS),
                    help="existing authorized receipt repository; no automatic fallback")
    ap.add_argument("--receipts-repo-type", choices=("dataset", "model"),
                    default=os.environ.get("SZL_RECEIPTS_REPO_TYPE", "dataset"),
                    help="model channels must be verified private before and after publication")
    ap.add_argument("--receipt-outbox", default="~/salad-receipt-outbox",
                    help="local pending receipt bytes and verified publication acknowledgements")
    ap.add_argument("--heartbeat-seconds", type=int, default=300,
                    help="publish at the first Trainer step boundary after this interval")
    ap.add_argument("--publish-final-to", default="",
                    help="optional separate repo for the final adapter (safetensors/config/tokenizer only)")
    a = ap.parse_args()
    if min(a.max_steps, a.save_steps, a.max_examples, a.min_examples, a.heartbeat_seconds) < 1:
        ap.error("step, save, example and minimum-example bounds must be positive")
    if a.max_examples < a.min_examples:
        ap.error("max-examples cannot be smaller than min-examples")
    if not a.hf_token:
        print("FAIL-CLOSED: HF_TOKEN required (env or --hf-token)", file=sys.stderr)
        return 2

    api = HfApi(token=a.hf_token)
    global RECEIPTS
    try:
        RECEIPTS = ReceiptChannel(
            api, a.hf_token, a.receipts_repo, a.receipts_repo_type,
            os.path.expanduser(a.receipt_outbox), a.adapter_out,
        )
        return run_training(a, api)
    except ReceiptError as error:
        print(f"FAIL-CLOSED: {error}; receipt publication unverified", file=sys.stderr, flush=True)
        return 8
    except ArtifactError as error:
        print(f"FAIL-CLOSED: {error}; artifact publication unverified", file=sys.stderr, flush=True)
        return 9
    finally:
        if _ZEUS_STATE.get("window"):
            zeus_end_window()


def run_training(a, api) -> int:
    # The working repo holds resumable (pickle-bearing) checkpoints, so it is created PRIVATE.
    # create_repo(exist_ok=True) leaves an EXISTING public repo public, so verify explicitly.
    try:
        ensure_private_repo(api, a.adapter_out, a.hf_token)
    except VisibilityError as e:
        print(f"FAIL-CLOSED: {e}", file=sys.stderr)
        return 3
    try:
        resume = fetch_resume_checkpoint(api, a.adapter_out, os.path.expanduser("~/ckpt"))
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
             "target_steps": a.max_steps,
             "energy_start": energy_start,
             "zeus_window": zeus_start})

    cuda_arch_precheck()  # F4

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

    tok = AutoTokenizer.from_pretrained(a.base)
    model = AutoModelForCausalLM.from_pretrained(
        a.base, dtype=torch.bfloat16, device_map="auto")
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
        output_dir=os.path.expanduser("~/ckpt"),
        max_length=2048,
        per_device_train_batch_size=1,          # spot-metal safe: fits 24GB cards
        gradient_accumulation_steps=16,
        learning_rate=2e-4, lr_scheduler_type="cosine", warmup_steps=20,
        max_steps=a.max_steps,
        save_steps=a.save_steps, save_total_limit=2,
        bf16=True, optim="adamw_torch", report_to=[], logging_steps=10,
        gradient_checkpointing=False, packing=False)
    peft = LoraConfig(r=32, lora_alpha=64, lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
        task_type="CAUSAL_LM")

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, peft_config=peft)

    from transformers import TrainerCallback
    class ReceiptCallback(TrainerCallback):
        def __init__(self):
            self.last_heartbeat = time.monotonic()

        def publish_or_stop(self, event, detail, control, extra=None, request_save=False):
            try:
                receipt(event, detail, extra)
                return True
            except ReceiptError as error:
                STOP["flag"] = True
                STOP["receipts"] = True
                control.should_training_stop = True
                if request_save:
                    control.should_save = True
                print(f"FAIL-CLOSED: {error}; checkpoint requested before exit",
                      file=sys.stderr, flush=True)
                return False

        def on_step_end(self, args, state, control, **kw):
            if STOP["flag"]:
                control.should_training_stop = True
                control.should_save = True
                return
            now = time.monotonic()
            if now - self.last_heartbeat >= a.heartbeat_seconds:
                observed = state.global_step
                actual = observed if type(observed) is int and observed >= 0 else None
                self.publish_or_stop("heartbeat", "Trainer step boundary observed", control,
                    {"actual_steps": actual, "target_steps": a.max_steps,
                     "scope": "TRAINER_LIVENESS_NOT_CHECKPOINT_DURABILITY"}, request_save=True)
                self.last_heartbeat = now

        def on_save(self, args, state, control, **kw):
            # push the checkpoint to durable storage immediately
            ck = f"checkpoint-{state.global_step}"
            path = os.path.join(args.output_dir, ck)
            if os.path.exists(path):
                try:
                    # re-check every time: visibility can change while training runs
                    verify_private(api, a.adapter_out)
                except VisibilityError as e:
                    STOP["flag"] = True
                    STOP["visibility"] = True
                    control.should_training_stop = True
                    self.publish_or_stop("checkpoint-blocked", f"upload refused: {e}", control)
                    return
                try:
                    commit = api.upload_folder(folder_path=path, path_in_repo=ck,
                        repo_id=a.adapter_out, repo_type="model")
                    revision = hub_revision(commit)
                except Exception as e:
                    STOP["flag"] = True
                    STOP["checkpoint"] = True
                    control.should_training_stop = True
                    self.publish_or_stop("checkpoint-fail",
                        f"checkpoint publication unverified ({transport_label(e)})", control)
                else:
                    self.publish_or_stop("checkpoint", f"checkpoint {ck} commit acknowledged", control,
                        {"actual_steps": state.global_step, "checkpoint": ck,
                         "checkpoint_revision": revision,
                         "checkpoint_durability": "HUB_COMMIT_ACKNOWLEDGED_NOT_FULL_BYTE_READBACK"})
            else:
                STOP["flag"] = True
                STOP["checkpoint"] = True
                control.should_training_stop = True
                self.publish_or_stop("checkpoint-fail", "local checkpoint directory missing", control)
            if STOP["flag"]:
                control.should_training_stop = True
    trainer.add_callback(ReceiptCallback())

    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume) if resume else trainer.train()
    observed_steps = getattr(getattr(trainer, "state", None), "global_step", None)
    actual_steps = observed_steps if type(observed_steps) is int and observed_steps >= 0 else None
    if actual_steps != a.max_steps or STOP.get("receipts") or STOP.get("checkpoint"):
        # Stop signals, early exits, invalid state and overshoot are not completion.
        # Do not overwrite final/ with a partial adapter. Checkpoint preservation
        # remains the callback's responsibility; this branch cannot prove durability.
        energy_end = sample_energy_j()
        receipt("interrupted" if STOP["flag"] else "incomplete",
                "training did not reach the exact admitted target; final publication withheld",
                {"actual_steps": actual_steps, "target_steps": a.max_steps,
                 "completion_qualified": False, "final_publication_attempted": False,
                 "receipt_channel_failed": bool(STOP.get("receipts")),
                 "checkpoint_publication_failed": bool(STOP.get("checkpoint")),
                 "checkpoint_durability": "UNVERIFIED_BY_TERMINAL_GUARD",
                 "energy_start": energy_start, "energy_end": energy_end,
                 "zeus_window": zeus_end_window(),
                 "energy_j": energy_delta(energy_start, energy_end)})
        print("FAIL-CLOSED: incomplete or unqualified trainer state; final adapter not published",
              file=sys.stderr, flush=True)
        return 8 if STOP.get("receipts") else 7
    final = os.path.expanduser("~/ckpt/final")
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
    try:
        final_commit = api.upload_folder(folder_path=final, path_in_repo="final",
            repo_id=a.adapter_out, repo_type="model", ignore_patterns=PICKLE_PATTERNS)
    except Exception as error:
        raise ArtifactError(f"final adapter upload failed ({transport_label(error)})") from None
    final_revision = hub_revision(final_commit)
    published_revision = None
    if a.publish_final_to:
        try:
            ensure_private_repo(api, a.publish_final_to, a.hf_token)
        except VisibilityError as e:
            print(f"FAIL-CLOSED: {e}; final adapter not published", file=sys.stderr)
            return 3
        try:
            published_commit = api.upload_folder(folder_path=final, repo_id=a.publish_final_to, repo_type="model",
                              ignore_patterns=PICKLE_PATTERNS)  # visibility change stays a human decision
        except Exception as error:
            raise ArtifactError(f"optional adapter upload failed ({transport_label(error)})") from None
        published_revision = hub_revision(published_commit)
    energy_end = sample_energy_j()  # v3: cumulative counter at completion
    receipt("complete", f"training complete in {round(time.time()-t0,1)}s",
            {"steps": actual_steps, "actual_steps": actual_steps,
             "target_steps": a.max_steps, "completion_qualified": True,
             "completion_scope": "EXACT_TRAINER_STEP_TARGET_NOT_MODEL_QUALITY",
             "adapter": a.adapter_out,
             "final_artifact": {"repo_id": a.adapter_out, "repo_type": "model",
                                "revision": final_revision, "directory": "final",
                                "verification": "HUB_COMMIT_ACKNOWLEDGED_NOT_FULL_BYTE_READBACK"},
             "published_adapter": ({"repo_id": a.publish_final_to, "revision": published_revision}
                                   if a.publish_final_to else None),
             "energy_start": energy_start,
             "energy_end": energy_end,
             "zeus_window": zeus_end_window(),
             "energy_j": energy_delta(energy_start, energy_end)})
    return 0


if __name__ == "__main__":
    sys.exit(main())

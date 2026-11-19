"""One-shot bounded, source-bound model-loop *preflight*.

This is NOT a model trainer, route activator, Hatun client, or GPU dispatcher.
No networking, weights, credentials, provider mutations or scheduling. Evidence
is source-only and must be ratified by the actual source owners.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import hmac
import json
from pathlib import Path
import re
import stat
import zipfile

SCHEMA = "szl.multi-model-recipe/v0"
OBS_SCHEMA = "szl.model-recipe.hf-public-metadata/v1"
DEPENDENCY_SCHEMA = "szl.model-recipe.dependency-readonly/v1"
TRAIN_ZIP_SHA256 = "4216b3e1130d3cba49f05fe9456d9e2ee936a9eb87ab80639698f0b58caa3c4d"
TRAIN_SHA256 = "36048599efb5abc1b4ca0ef21ac1eae072c7412728f1dacd9cbe437b623df502"
MANIFEST_SHA256 = "d8cb77a517e69ded7314747df5e68db3481aaba3639a8529ed1e3eb47c448904"
TRAIN_NAMES = frozenset({
    "TRAINING_README.md", "TRAINING_ADMISSION.json", "SOURCE_CATALOG.json",
    "data-v1/train_sft.jsonl", "data-v1/manifest.json",
})
MAX_ZIP_BYTES = 1_000_000
MAX_UNPACKED_BYTES = 3_000_000
MAX_CYCLES = 3
MAX_STEPS = 12
MODEL_ID = re.compile(r"SZLHOLDINGS/[A-Za-z0-9_.-]{1,80}\Z")
SHA40 = re.compile(r"[a-f0-9]{40}\Z")
SHA64 = re.compile(r"[a-f0-9]{64}\Z")
ROLES = frozenset({"grounded", "offline", "proposal", "identity", "navigation",
                   "embedding", "scholar", "structure", "scholar-4b", "receipt"})
SOURCE_OWNERS = {
    "szl-holdings/szl-gpu-bridge": "23b45e1aa628b273108faeca59c02f6ef031b88c",
    "szl-holdings/szl-cram": "3f8894caad893961167af39c7f3d9f40383fe7cf",
    "szl-holdings/szl-cram:research-pr-11": "368b4179548a4962732a1f4c5569197641a1dd6b",
    "szl-holdings/szl-second-brain": "73e8e4388268d092c50eac849d2a6f22a18418f2",
    "szl-holdings/hatun-mcp": "1cec84742e07c151a34c329ae2fb50dd5f87c7ab",
}

class Hold(ValueError):
    pass

def require(ok: bool, code: str):
    if not ok:
        raise Hold(code)

def pairs_strict(pairs):
    result = {}
    for k, v in pairs:
        require(k not in result, "DUPLICATE_JSON_KEY")
        result[k] = v
    return result

def canonical(value):
    return (json.dumps(value, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False) + "\n").encode("ascii")

def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def read_json(path: Path, cap=200_000):
    require(path.is_file() and not path.is_symlink(), "INPUT_FILE_MISSING")
    require(0 < path.stat().st_size <= cap, "INPUT_BYTE_BUDGET")
    try:
        return json.loads(path.read_text(encoding="utf-8"),
                          object_pairs_hook=pairs_strict,
                          parse_constant=lambda _: require(False, "NONFINITE_JSON"))
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise Hold("INPUT_JSON_INVALID") from exc

def verify_plan(plan):
    require(type(plan) is dict and plan.get("schema") == SCHEMA, "PLAN_SCHEMA")
    require(plan.get("source_pins") == SOURCE_OWNERS, "SOURCE_OWNER_DRIFT")
    require(plan.get("training_dispatch_enabled") is False and
            plan.get("provider_write_enabled") is False and
            plan.get("private_graph_hydration") is False, "FORBIDDEN_EFFECT_FLAG")
    candidates = plan.get("models")
    require(type(candidates) is list and 1 <= len(candidates) <= MAX_STEPS, "MODEL_COUNT_BUDGET")
    ids=set()
    for m in candidates:
        require(type(m) is dict and set(m) ==
                {"id", "observed_sha", "role", "admission"}, "MODEL_SCHEMA")
        mid=m["id"]
        require(type(mid) is str and MODEL_ID.fullmatch(mid) is not None, "MODEL_ID")
        require(mid not in ids, "DUPLICATE_MODEL")
        ids.add(mid)
        require(type(m["observed_sha"]) is str and SHA40.fullmatch(m["observed_sha"]),
                "MODEL_SHA")
        require(type(m["role"]) is str and m["role"] in ROLES, "ROLE_UNSUPPORTED")
        require(m["admission"] == "REVIEW_REQUIRED", "UNAUTHORIZED_MODEL_ADMISSION")
    assert len(ids) == len(candidates)
    return candidates

def verify_training_zip(path: Path | None):
    """Content identity only. The separate owner CRAM verifier checks labels."""
    if path is None:
        return {"state": "NOT_STAGED", "training_admitted": False}
    require(path.is_file() and not path.is_symlink(), "BUNDLE_NOT_FOUND")
    require(path.stat().st_size <= MAX_ZIP_BYTES, "ZIP_BYTE_BUDGET")
    require(hmac.compare_digest(digest(path.read_bytes()), TRAIN_ZIP_SHA256),
            "TRAIN_ZIP_SHA_DRIFT")
    try:
        with zipfile.ZipFile(path) as z:
            infos=z.infolist()
            require({i.filename for i in infos} == TRAIN_NAMES and len(infos) == 5,
                    "TRAIN_ONLY_ENTRIES_MISMATCH")
            require(sum(i.file_size for i in infos) <= MAX_UNPACKED_BYTES, "ZIP_BOMB_BUDGET")
            require(all(not stat.S_ISLNK(i.external_attr >> 16) for i in infos),
                    "ZIP_LINK_REFUSED")
            manifest_raw=z.read("data-v1/manifest.json")
            train_raw=z.read("data-v1/train_sft.jsonl")
    except (zipfile.BadZipFile, RuntimeError, KeyError) as exc:
        raise Hold("ZIP_INVALID") from exc
    require(digest(manifest_raw) == MANIFEST_SHA256 and
            digest(train_raw) == TRAIN_SHA256, "TRAIN_BYTES_DRIFT")
    manifest=json.loads(manifest_raw, object_pairs_hook=pairs_strict)
    require(type(manifest) is dict and
            manifest.get("schema") == "szl.reasoning-compression.dataset-manifest/v0" and
            manifest.get("classification") == "AUTHOR_SYNTHETIC_SOFTWARE_FIXTURES_ONLY" and
            manifest.get("train_input_path") == "train_sft.jsonl" and
            manifest.get("dev_test_training_admission") == "DENY; evaluation only" and
            manifest.get("training_run_started") is False and
            manifest.get("upstream_paper_code_data_ingested") is False and
            manifest.get("receipt_signed") is False and
            manifest.get("sizes", {}).get("train") == 1320 and
            manifest.get("sha256", {}).get("train_sft.jsonl") == TRAIN_SHA256,
            "TRAIN_MANIFEST_POLICY")
    require(len(train_raw.splitlines()) == 1320, "TRAIN_ROW_COUNT")
    return {"state": "SOURCE_BYTES_HASHED_TRAIN_ONLY",
            "rows": 1320, "dataset_sha256": TRAIN_SHA256,
            "train_admitted_for_gpu": False,
            "independent_label_replay": "SEPARATE_SZL_CRAM_OWNER_GATE_REQUIRED"}

def _fresh(observed_at, hours=6):
    try:
        t=datetime.fromisoformat(observed_at.replace("Z","+00:00"))
        if t.tzinfo is None:
            return False
        seconds=(datetime.now(timezone.utc)-t).total_seconds()
        return 0 <= seconds <= hours*3600
    except (AttributeError, TypeError, ValueError):
        return False

def classify(candidates, observation, dependencies, dataset):
    require(type(observation) is dict and observation.get("schema") == OBS_SCHEMA,
            "HF_OBSERVATION_SCHEMA")
    require(type(dependencies) is dict and dependencies.get("schema") == DEPENDENCY_SCHEMA,
            "DEPENDENCY_SCHEMA")
    rows=observation.get("rows")
    require(type(rows) is list and len(rows) <= 100, "HF_OBSERVATION_SIZE")
    hfs={}
    for row in rows:
        require(type(row) is dict and type(row.get("id")) is str, "HF_OBSERVATION_ROW")
        require(row["id"] not in hfs, "HF_DUPLICATE_ID")
        hfs[row["id"]]=row
    deps=dependencies.get("rows")
    require(type(deps) is list and len(deps) <= 12, "DEPENDENCY_COUNT")
    brain = [r for r in deps if type(r) is dict and
             r.get("surface") == "SZLHOLDINGS/szl-second-brain" and
             r.get("path") in ("/health", "/readyz")]
    brain_ready = (len(brain) == 2 and len({r["path"] for r in brain}) == 2
                   and all(type(r.get("http_status")) is int and r["http_status"] == 200
                           for r in brain))
    hatun_runtime_verified = dependencies.get("hatun_runtime_tools_list_authenticated") == "AUTHENTICATED_SIGNED_SOURCE_BOUND"
    # Even if metadata claims runtime verification, this *reader* does not
    # verify credentials/DSSE or launch a model, hence all evaluations HOLD.
    fresh = _fresh(observation.get("captured_utc"))
    results=[]
    for model in candidates:
        reasons=[]
        row=hfs.get(model["id"])
        if not fresh:
            reasons.append("HF_OBSERVATION_STALE_OR_UNDATED")
        if row is None or row.get("state") != "PUBLIC_METADATA_OBSERVED_NOT_ARTIFACT_VERIFIED":
            reasons.append("MODEL_METADATA_UNAVAILABLE")
        elif row.get("sha") != model["observed_sha"]:
            reasons.append("MODEL_SOURCE_SHA_CHANGED")
        if row is not None and row.get("gated") not in (False, "False", "false", None, ""):
            reasons.append("MODEL_PROVIDER_ACCESS_NOT_PROVEN")
        if dataset["state"] != "SOURCE_BYTES_HASHED_TRAIN_ONLY":
            reasons.append("TRAIN_BUNDLE_NOT_VERIFIED")
        if not brain_ready:
            reasons.append("SECOND_BRAIN_READINESS_UNAVAILABLE")
        if not hatun_runtime_verified:
            reasons.append("HATUN_AUTH_SIGNER_UNVERIFIED")
        if model["role"] == "embedding":
            reasons.append("EMBEDDING_ONLY_NO_GENERATIVE_ROUTING")
        if model["role"] in {"scholar","structure","scholar-4b","receipt"}:
            reasons.append("ADAPTER_BASE_AND_CHECKPOINT_NOT_QUALIFIED")
        reasons.append("PAIRED_HELDOUT_MODEL_EVALUATION_NOT_RUN")
        reasons.append("OWNER_BUDGET_AND_RELEASE_ADMISSION_MISSING")
        results.append({"id":model["id"],"pinned_sha":model["observed_sha"],
                        "role":model["role"],"state":"HOLD",
                        "reasons":sorted(set(reasons)),
                        "training_or_inference_dispatched":False,
                        "source_metadata_only":True})
    return results, {"second_brain_health_ready":brain_ready,
                     "hatun_authenticated_signed_runtime_verified":False,
                     "hf_metadata_fresh":fresh}

def _exclusive_file(path, data):
    with path.open("xb") as f:
        f.write(data)

def run(plan, observation, dependencies, dataset, outdir: Path, *, cycles=1, max_steps=12):
    require(type(cycles) is int and 1 <= cycles <= MAX_CYCLES, "CYCLE_CAP")
    require(type(max_steps) is int and 1 <= max_steps <= MAX_STEPS, "STEP_CAP")
    candidates=verify_plan(plan)
    require(len(candidates) <= max_steps, "BUDGET_EXHAUSTED_NO_PARTIAL_CLAIM")
    results, gates=classify(candidates, observation, dependencies, dataset)
    require(not outdir.exists(), "EVIDENCE_OUTPUT_EXISTS")
    outdir.mkdir(parents=True, exist_ok=False)
    previous=None
    hashes=[]
    for cycle in range(1,cycles+1):
        # A cycle replays a single immutable metadata capture. It is not
        # a fresh provider check, training step, or evidence of model progress.
        receipt={
            "schema":"szl.model-loop.preflight-receipt/v0",
            "utc":datetime.now(timezone.utc).isoformat(),
            "cycle":cycle,"cycles_budget":cycles,"max_steps":max_steps,
            "steps_processed":len(results),
            "exit":"COMPLETED_BOUNDED_SCAN_NOT_MODEL_CONVERGENCE",
            "classification":"READ_ONLY_METADATA_REPLAY_NOT_TRAINING",
            "source_pins":SOURCE_OWNERS,
            "previous_receipt_sha256":previous,
            "dataset":dataset,"dependency_gates":gates,
            "candidates":results,
            "all_models_training_qualified":False,
            "training_jobs_created":0,"paid_gpu_jobs_created":0,
            "second_brain_private_graph_read":False,
            "hatun_mcp_tools_called":False,
            "receipt_signed":False,"production_changed":False,
        }
        raw=canonical(receipt)
        sha=digest(raw)
        _exclusive_file(outdir / f"cycle-{cycle:02d}.json", raw)
        _exclusive_file(outdir / f"cycle-{cycle:02d}.sha256",
                        f"{sha}  cycle-{cycle:02d}.json\n".encode("ascii"))
        previous=sha; hashes.append(sha)
    stats=Counter(r["state"] for r in results)
    svg=('<svg xmlns="http://www.w3.org/2000/svg" width="780" height="160">'
         '<rect width="780" height="160" fill="white"/>'
         '<text x="16" y="32" font-size="21">SZL model-loop preflight - metadata only</text>'
         '<text x="16" y="68" font-size="16">'
         + str(len(results)) + ' source-pinned candidates, '
         + str(stats["HOLD"]) + ' HOLD, 0 dispatched</text>'
         '<text x="16" y="107" font-size="13">No model inference, training, signed receipts or convergence claim</text>'
         '</svg>\n')
    _exclusive_file(outdir/"graph.svg", svg.encode("utf-8"))
    _exclusive_file(outdir/"graph.sha256",
                    (digest(svg.encode("utf-8"))+"  graph.svg\n").encode("ascii"))
    return {"state":"HOLD_ALL_MODEL_ADMISSIONS",
            "models_observed":len(results),"cycles_completed":cycles,
            "cycle_receipt_sha256":hashes,
            "graph_sha256":digest(svg.encode("utf-8")),
            "training_jobs_created":0,"production_authorized":False}

def main(argv=None):
    parser=argparse.ArgumentParser(description="Read-only multi-model roster and bounded receipt loop; never trains.")
    parser.add_argument("--plan",type=Path,required=True)
    parser.add_argument("--hf-observation",type=Path,required=True)
    parser.add_argument("--dependency-observation",type=Path,required=True)
    parser.add_argument("--train-only-zip",type=Path)
    parser.add_argument("--outdir",type=Path,required=True)
    parser.add_argument("--cycles",type=int,default=1)
    parser.add_argument("--max-steps",type=int,default=12)
    args=parser.parse_args(argv)
    try:
        dataset=verify_training_zip(args.train_only_zip)
        outcome=run(read_json(args.plan),read_json(args.hf_observation),
                    read_json(args.dependency_observation),
                    dataset,args.outdir,cycles=args.cycles,max_steps=args.max_steps)
        print(json.dumps(outcome,sort_keys=True))
        return 2  # Admission HOLD is visible even if scan itself succeeded.
    except (Hold, OSError, ValueError) as exc:
        print(json.dumps({"state":"HOLD","reason":
                          str(exc) if isinstance(exc,Hold) else type(exc).__name__,
                          "training_jobs_created":0,"production_authorized":False}))
        return 2

if __name__=="__main__":
    raise SystemExit(main())

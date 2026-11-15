#!/usr/bin/env python3
"""Fail-closed, single-lane checkpoint archiver and completion guard.

Does not train, perform inference, unpickle objects, or change repository visibility.
An artifact filename alone never authorizes scale-down. Credentials stay on-node.
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import sys
import time
import urllib.request
import uuid

VERSION = '1.0.1'
LANE = 'szl-train-flagship54-v1'
REPO = 'SZLHOLDINGS/salad-flagship54-v1'
BASE = 'https://api.salad.com/api/public/organizations/alloyscape/projects/default/containers/'
STOP = False


class Hold(RuntimeError):
    """Only fixed diagnostic codes are emitted, never remote exception bodies."""


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(value) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    fd = os.open(temp, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(fd, 'wb') as out:
            out.write(canonical(value) + b'\n')
            out.flush()
            os.fsync(out.fileno())
        os.replace(temp, path)
        if os.name == 'posix':
            directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
    finally:
        if temp.exists():
            temp.unlink()


def receipt(cfg, kind: str, details: dict) -> Path:
    body = {'kind': 'szl.flagship54-guard/v1', 'event': kind, 'observed_at': utc(),
            'lane': LANE, 'repo': REPO, 'guard_version': VERSION, 'guard_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), **details}
    body['payload_sha256'] = hashlib.sha256(canonical(body)).hexdigest()
    path = Path(cfg['state_dir']) / 'receipts' / (kind.lower() + '-' + uuid.uuid4().hex + '.json')
    write_json(path, body)
    return path


def secret(path: str) -> str:
    p = Path(path)
    if p.is_symlink() or not p.is_file():
        raise Hold('CREDENTIAL_FILE_INVALID')
    info = p.stat()
    if os.name == 'posix' and (stat.S_IMODE(info.st_mode) & 0o077 or info.st_uid != os.getuid()):
        raise Hold('CREDENTIAL_FILE_PERMISSIONS')
    value = p.read_text(encoding='utf-8-sig').strip()
    if not value or any(c.isspace() for c in value):
        raise Hold('CREDENTIAL_FILE_EMPTY_OR_WHITESPACE')
    return value


def validate_config(cfg: dict) -> None:
    if cfg.get('lane') != LANE or cfg.get('repo') != REPO:
        raise Hold('LANE_NOT_ALLOWLISTED')
    if type(cfg.get('max_steps')) is not int or cfg['max_steps'] <= 0:
        raise Hold('MAX_STEPS_INVALID')
    if not re.fullmatch(r'[0-9a-f]{64}', cfg.get('command_sha256', '')):
        raise Hold('COMMAND_BINDING_INVALID')
    if not cfg.get('group_id') or type(cfg.get('group_version')) is not int:
        raise Hold('GROUP_BINDING_MISSING')
    for name in ('state_dir', 'archive_dir', 'hf_token_file', 'salad_key_file'):
        if not Path(cfg[name]).is_absolute():
            raise Hold('ABSOLUTE_PATH_REQUIRED')


def check_group(group: dict, cfg: dict) -> None:
    if group.get('name') != LANE or group.get('id') != cfg['group_id']:
        raise Hold('GROUP_IDENTITY_CHANGED')
    command_hash = hashlib.sha256(canonical(group.get('container', {}).get('command', []))).hexdigest()
    if command_hash != cfg['command_sha256']:
        raise Hold('TRAINING_COMMAND_CHANGED')
    # A successful scale-down itself advances Salad's version. Version may rise
    # only after this guard has persisted its own scale-down intent.
    if group.get('version') != cfg['group_version']:
        intent = Path(cfg['state_dir']) / 'scale-down-intent.json'
        if not intent.exists() or group.get('replicas') != 0:
            raise Hold('GROUP_VERSION_CHANGED')
        bound = json.loads(intent.read_text())
        if bound.get('group_id') != cfg['group_id'] or bound.get('from_version') != cfg['group_version'] or group.get('version') != cfg['group_version'] + 1:
            raise Hold('SCALE_DOWN_INTENT_BINDING_MISMATCH')
    if group.get('scheduled-scaling-enabled') or group.get('scheduled_scaling_enabled'):
        raise Hold('SCHEDULED_SCALING_ACTIVE')
    if group.get('queue_autoscaler') or group.get('queue_based_autoscaler'):
        raise Hold('AUTOSCALER_ACTIVE')


def counts_drained(group: dict) -> bool:
    counts = group.get('current_state', {}).get('instance_status_counts', {})
    keys = ('allocating_count', 'creating_count', 'running_count', 'stopping_count')
    return group.get('replicas') == 0 and all(type(counts.get(k)) is int and counts[k] == 0 for k in keys)


class Salad:
    def __init__(self, key_file: str):
        self.key_file = key_file

    def request(self, method='GET', data=None):
        headers = {'Salad-Api-Key': secret(self.key_file), 'User-Agent': 'Mozilla/5.0',
                   'Content-Type': 'application/merge-patch+json'}
        request = urllib.request.Request(BASE + LANE, headers=headers,
                    data=None if data is None else canonical(data), method=method)
        with urllib.request.urlopen(request, timeout=40) as response:
            return json.load(response)


def metadata(api):
    info = api.model_info(REPO, files_metadata=True)
    if info.private is not True:
        raise Hold('WORKING_REPOSITORY_NOT_VERIFIABLY_PRIVATE')
    if not re.fullmatch(r'[0-9a-f]{40,64}', info.sha):
        raise Hold('PINNED_REVISION_INVALID')
    entries = {f.rfilename: f for f in info.siblings}
    steps = {int(m.group(1)) for name in entries if (m := re.match(r'^checkpoint-(\d+)/', name))}
    if not steps:
        raise Hold('NO_DURABLE_CHECKPOINT')
    latest = max(steps)
    state_path = f'checkpoint-{latest}/trainer_state.json'
    if state_path not in entries:
        raise Hold('LATEST_CHECKPOINT_INCOMPLETE')
    return info, entries, latest, state_path


def file_hashes(path: Path):
    size = path.stat().st_size
    sha = hashlib.sha256()
    git = hashlib.sha1(b'blob ' + str(size).encode() + b'\0')
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            sha.update(block)
            git.update(block)
    return size, sha.hexdigest(), git.hexdigest()


def validate_file(path: Path, entry) -> dict:
    size, digest, git_digest = file_hashes(path)
    if type(entry.size) is not int or entry.size <= 0 or size != entry.size:
        raise Hold('ARTIFACT_SIZE_MISMATCH')
    lfs = getattr(entry, 'lfs', None)
    lfs_digest = lfs.get('sha256') if isinstance(lfs, dict) else getattr(lfs, 'sha256', None)
    expected = lfs_digest or getattr(entry, 'blob_id', None)
    if not expected or (digest if lfs_digest else git_digest) != expected:
        raise Hold('ARTIFACT_HASH_MISMATCH')
    return {'size': size, 'sha256': digest, 'hub_oid': expected,
            'hub_oid_algorithm': 'sha256' if lfs_digest else 'git-blob-sha1'}


def safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        raise Hold('UNSAFE_ARTIFACT_PATH')
    return path


def download_checked(api, info, entries, name, destination=None):
    from huggingface_hub import hf_hub_download
    if name not in entries:
        raise Hold('REQUIRED_ARTIFACT_MISSING')
    safe_relative(name)
    cached = Path(hf_hub_download(REPO, name, revision=info.sha, token=api.token, etag_timeout=30))
    measure = validate_file(cached, entries[name])
    if destination is not None:
        target = Path(destination).joinpath(*safe_relative(name).parts)
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if target.is_symlink():
            raise Hold('ARCHIVE_SYMLINK')
        temp = target.with_name(target.name + '.' + uuid.uuid4().hex + '.tmp')
        with cached.open('rb') as src, temp.open('xb') as dst:
            os.chmod(temp, 0o600)
            shutil.copyfileobj(src, dst, length=1024 * 1024)
            dst.flush()
            os.fsync(dst.fileno())
        os.replace(temp, target)
        if validate_file(target, entries[name]) != measure:
            raise Hold('ARCHIVE_COPY_MISMATCH')
        return target, measure
    return cached, measure


def validate_state(state: dict, step: int, cfg: dict) -> None:
    if type(state.get('global_step')) is not int or state['global_step'] != step:
        raise Hold('CHECKPOINT_STEP_MISMATCH')
    if type(state.get('max_steps')) is not int or state['max_steps'] != cfg['max_steps']:
        raise Hold('TRAINING_TARGET_CHANGED')


def required_exports(entries, latest: int) -> list[str]:
    ck = f'checkpoint-{latest}/'
    minimum = ['final/adapter_model.safetensors', 'final/adapter_config.json',
               ck + 'adapter_model.safetensors', ck + 'adapter_config.json', ck + 'trainer_state.json']
    # Preserve a tokenizer from the same pinned final/checkpoint snapshot.
    for leaf in ('tokenizer.json', 'tokenizer_config.json'):
        minimum.append('final/' + leaf if 'final/' + leaf in entries else ck + leaf)
    if any(name not in entries for name in minimum):
        raise Hold('FINAL_EXPORT_INCOMPLETE')
    final_names = [name for name in entries if name.startswith('final/')]
    if any(PurePosixPath(name).suffix.lower() in {'.pt', '.pth', '.bin', '.pkl', '.pickle', '.ckpt', '.py'} for name in final_names):
        raise Hold('UNSAFE_FINAL_EXPORT')
    return sorted(set(final_names + [name for name in entries if name.startswith(ck)] + minimum))


def archive(api, info, entries, names, cfg: dict) -> tuple[Path, dict]:
    total = sum(entries[name].size or 0 for name in names)
    if total <= 0 or total > cfg.get('max_archive_bytes', 4 * 1024**3):
        raise Hold('ARCHIVE_SIZE_OUT_OF_BOUNDS')
    root = Path(cfg['archive_dir']) / info.sha
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < total + 2 * 1024**3:
        raise Hold('INSUFFICIENT_ARCHIVE_DISK')
    records = {}
    for name in names:
        _, records[name] = download_checked(api, info, entries, name, root)
    manifest = {'kind':'szl.pinned-artifact-archive/v1', 'repo':REPO, 'revision':info.sha,
                'observed_at':utc(), 'files':records, 'total_bytes':total,
                'validation':'fresh byte hashes checked against pinned Hub object identifiers; pickle files never loaded'}
    write_json(root / 'manifest.json', manifest)
    return root, manifest


def validate_export(root: Path, manifest: dict, latest: int, cfg: dict) -> None:
    records = manifest['files']
    if records['final/adapter_model.safetensors']['sha256'] != records[f'checkpoint-{latest}/adapter_model.safetensors']['sha256']:
        raise Hold('FINAL_WEIGHTS_DO_NOT_MATCH_COMPLETED_CHECKPOINT')
    if records['final/adapter_config.json']['sha256'] != records[f'checkpoint-{latest}/adapter_config.json']['sha256']:
        raise Hold('FINAL_CONFIG_DOES_NOT_MATCH_COMPLETED_CHECKPOINT')
    config = json.loads((root / 'final/adapter_config.json').read_text())
    if config.get('base_model_name_or_path') != cfg['base_model'] or config.get('peft_type') != 'LORA':
        raise Hold('ADAPTER_BASE_OR_TYPE_MISMATCH')
    from safetensors import safe_open
    with safe_open(root / 'final/adapter_model.safetensors', framework='pt', device='cpu') as tensors:
        if not list(tensors.keys()):
            raise Hold('EMPTY_SAFETENSORS')
        for name in tensors.keys():
            tensors.get_slice(name).get_shape()
    for name in records:
        if name.endswith('/tokenizer.json') or name.endswith('/tokenizer_config.json'):
            if not isinstance(json.loads(root.joinpath(*safe_relative(name).parts).read_text()), dict):
                raise Hold('TOKENIZER_JSON_INVALID')


def cycle(cfg, api, salad, *, apply=False, backup_latest=False) -> dict:
    validate_config(cfg)
    group = salad.request()
    check_group(group, cfg)
    info, entries, latest, state_name = metadata(api)
    state_file, _ = download_checked(api, info, entries, state_name)
    state = json.loads(state_file.read_text())
    validate_state(state, latest, cfg)
    result = {'observed_at':utc(), 'repo_revision':info.sha, 'latest_checkpoint':latest,
              'max_steps':cfg['max_steps'], 'replicas':group.get('replicas'),
              'cloud_state':group.get('current_state',{}).get('status'), 'cloud_counts':group.get('current_state',{}).get('instance_status_counts'),
              'private': True, 'mutation_performed':False}
    if backup_latest:
        names = sorted(name for name in entries if name.startswith(f'checkpoint-{latest}/'))
        root, manifest = archive(api, info, entries, names, cfg)
        path = receipt(cfg, 'CHECKPOINT_ARCHIVED', {'revision':info.sha,'checkpoint':latest,'file_count':len(names),'archive':str(root),'manifest_sha256':hashlib.sha256(canonical(manifest)+b'\n').hexdigest()})
        return {**result,'status':'CHECKPOINT_ARCHIVED','archive':str(root),'receipt':str(path),'files':len(names),'bytes':manifest['total_bytes']}
    if latest < cfg['max_steps']:
        counts = result['cloud_counts'] or {}
        keys = ('allocating_count', 'creating_count', 'running_count', 'stopping_count')
        if type(group.get('replicas')) is not int or group['replicas'] < 1:
            status = 'HOLD_INCOMPLETE_COMPUTE_NOT_RUNNING'
        elif not all(type(counts.get(k)) is int and counts[k] >= 0 for k in keys):
            status = 'COMPUTE_STATUS_UNKNOWN'
        elif counts['running_count'] > 0:
            status = 'TRAINING'
        elif counts['allocating_count'] > 0:
            status = 'ALLOCATING'
        elif counts['creating_count'] > 0:
            status = 'CREATING'
        elif counts['stopping_count'] > 0:
            status = 'STOPPING'
        else:
            status = 'NO_RUNNING_REPLICA'
        return {**result,'status':status,'final_present':'final/adapter_model.safetensors' in entries,
                'process_liveness_verified': False}
    names = required_exports(entries, latest)
    root, manifest = archive(api, info, entries, names, cfg)
    validate_export(root, manifest, latest, cfg)
    # Re-resolve both authorities immediately before any mutation.
    fresh = api.model_info(REPO)
    if fresh.private is not True or fresh.sha != info.sha:
        raise Hold('REPOSITORY_CHANGED_DURING_VERIFICATION')
    group = salad.request()
    check_group(group, cfg)
    verified = receipt(cfg, 'FINAL_VERIFIED', {'revision':info.sha,'checkpoint':latest,'archive':str(root),'manifest_sha256':hashlib.sha256(canonical(manifest)+b'\n').hexdigest(), 'scope':'artifact integrity and recorded training step; NOT inference quality or production qualification'})
    if not apply:
        return {**result,'status':'READY_DRY_RUN','verification_receipt':str(verified)}
    if group.get('replicas') != 0:
        write_json(Path(cfg['state_dir']) / 'scale-down-intent.json', {'observed_at':utc(),'group_id':cfg['group_id'],'from_version':group.get('version'),'revision':info.sha,'verification_receipt':str(verified)})
        salad.request('PATCH', {'replicas':0})
        result['mutation_performed'] = True
    group = salad.request()
    check_group(group, cfg)
    result.update(replicas=group.get('replicas'), cloud_state=group.get('current_state',{}).get('status'),cloud_counts=group.get('current_state',{}).get('instance_status_counts'))
    if counts_drained(group):
        path = receipt(cfg, 'DONE', {'revision':info.sha,'verification_receipt':str(verified),'replicas':0,'instance_status_counts':result['cloud_counts']})
        return {**result,'status':'DONE','receipt':str(path)}
    return {**result,'status':'DRAINING' if group.get('replicas') == 0 else 'HOLD_SCALE_DOWN_NOT_CONFIRMED','verification_receipt':str(verified)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--backup-latest', action='store_true')
    args = parser.parse_args()
    if args.backup_latest and args.apply:
        parser.error('--backup-latest cannot be combined with --apply')
    cfg = json.loads(Path(args.config).read_text())
    validate_config(cfg)
    Path(cfg['state_dir']).mkdir(mode=0o700, parents=True, exist_ok=True)
    lock_handle = None
    if args.apply:
        import fcntl
        lock_handle = (Path(cfg['state_dir']) / 'guard.lock').open('a')
        try:
            fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print(json.dumps({'status':'ALREADY_RUNNING','mutation_performed':False}))
            return 0
    os.environ.setdefault('HF_HUB_DISABLE_PROGRESS_BARS','1')
    os.environ.setdefault('HF_HUB_DOWNLOAD_TIMEOUT','120')
    from huggingface_hub import HfApi
    api = HfApi(token=secret(cfg['hf_token_file']))
    salad = Salad(cfg['salad_key_file'])
    global STOP
    def stop(_sig, _frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    last_progress, last_step = time.monotonic(), None
    while not STOP:
        try:
            result = cycle(cfg, api, salad, apply=args.apply, backup_latest=args.backup_latest)
            step = result.get('latest_checkpoint')
            if step != last_step:
                last_step, last_progress = step, time.monotonic()
            if result['status'] == 'TRAINING' and time.monotonic()-last_progress >= cfg.get('stale_after_seconds',2700):
                result['status'] = 'TRAINING_NO_NEW_CHECKPOINT'
                result['note'] = 'No new durable checkpoint within observation interval; not proof of a stalled process.'
        except Hold as exc:
            result = {'observed_at':utc(),'status':'HOLD','reason':str(exc),'mutation_performed':'not asserted; inspect scale-down intent if present'}
        except Exception as exc:
            result = {'observed_at':utc(),'status':'ERROR','error_type':type(exc).__name__,
                      'http_status':getattr(exc,'code',None),'mutation_performed':'unknown; inspect prior scale-down intent if present'}
        status_name = 'backup-latest-status.json' if args.backup_latest else ('status.json' if args.apply else 'dry-run-status.json')
        write_json(Path(cfg['state_dir']) / status_name, result)
        print(json.dumps(result,sort_keys=True),flush=True)
        if args.once or args.backup_latest or result['status'] == 'DONE':
            return 0 if result['status'] not in ('HOLD','ERROR') else 2
        for _ in range(cfg.get('poll_seconds',120)):
            if STOP:
                break
            time.sleep(1)
    return 0


if __name__ == '__main__':
    sys.exit(main())

import hashlib
import importlib.util
import os
import subprocess
import sys

HERE = os.path.dirname(__file__)
WORKER = os.path.abspath(
    os.path.join(HERE, os.pardir, "tools", "salad", "salad_mint_worker.py")
)


def _load():
    spec = importlib.util.spec_from_file_location("salad_mint_worker_test", WORKER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_module_imports():
    mod = _load()
    assert callable(mod.sha256_file)


def test_sha256_file_matches_hashlib(tmp_path):
    mod = _load()
    p = tmp_path / "blob.bin"
    data = os.urandom(1 << 16) + b"\x00" * 1000
    p.write_bytes(data)
    assert mod.sha256_file(str(p)) == hashlib.sha256(data).hexdigest()


def test_help_contract():
    r = subprocess.run(
        [sys.executable, WORKER, "--help"], capture_output=True, text=True, timeout=120
    )
    assert r.returncode == 0, r.stderr[-400:]
    for flag in [
        "--base",
        "--base-revision",
        "--adapter-repo",
        "--adapter-path",
        "--out-repo",
        "--receipts-repo",
        "--meta",
    ]:
        assert flag in r.stdout, "missing flag: " + flag


def test_doctrine_markers_present():
    src = open(WORKER, encoding="utf-8", errors="replace").read()
    assert "szl.mint/v1" in src, "receipt kind marker missing"
    assert "repo_exists" in src, "never-overwrite check missing"
    assert "never-overwrite" in src, "never-overwrite doctrine marker missing"
    assert "publication" in src.lower(), "publication verification missing"
    assert "HF_TOKEN" in src, "token discipline missing"

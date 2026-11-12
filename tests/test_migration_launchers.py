"""Exercise launcher boundaries without starting training or a serving process."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import MagicMock

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "gpu_preflight", ROOT / "laptop/gpu_preflight.py"
)
preflight = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(preflight)


class PreflightTests(unittest.TestCase):
    def torch(self):
        t = MagicMock()
        t.__version__ = "test"
        t.version.cuda = "12.1"
        t.cuda.is_available.return_value = True
        t.cuda.get_device_name.return_value = "test GPU"
        t.cuda.get_device_capability.return_value = (12, 0)
        t.cuda.get_arch_list.return_value = ["sm_86"]
        t.cuda.get_device_properties.return_value = types.SimpleNamespace(
            total_memory=32
        )
        t.cuda.is_bf16_supported.return_value = True
        return t

    def test_cuda_visibility_does_not_hide_execution_failure(self):
        t = self.torch()
        t.ones.side_effect = RuntimeError("no kernel image is available")
        r = preflight.probe(t, True)
        self.assertEqual(r["status"], "BLOCKED")
        self.assertIn("no kernel image", r["reason"])
        self.assertFalse(r["training_verified"])

    def test_missing_cuda_blocks_before_allocation(self):
        t = self.torch()
        t.cuda.is_available.return_value = False
        self.assertEqual(preflight.probe(t, True)["status"], "BLOCKED")
        t.ones.assert_not_called()

    def test_bf16_requirement_is_enforced(self):
        t = self.torch()
        t.cuda.is_bf16_supported.return_value = False
        self.assertEqual(preflight.probe(t, True)["status"], "BLOCKED")
        t.ones.assert_not_called()

    def test_nonfinite_result_blocks(self):
        t = self.torch()
        t.isfinite.return_value.all.return_value = False
        r = preflight.probe(t, True)
        self.assertEqual(r["status"], "BLOCKED")
        self.assertIn("nonfinite", r["reason"])

    def test_incorrect_result_blocks(self):
        t = self.torch()
        t.isfinite.return_value.all.return_value = True
        t.ones.return_value.__matmul__.return_value.__eq__.return_value = (
            types.SimpleNamespace(all=lambda: False)
        )
        self.assertEqual(preflight.probe(t, True)["status"], "BLOCKED")

    def test_correct_execution_is_not_training_verification(self):
        t = self.torch()
        t.isfinite.return_value.all.return_value = True
        x = t.ones.return_value
        y = x.__matmul__.return_value
        y.__eq__.return_value = types.SimpleNamespace(all=lambda: True)
        x.grad.__eq__.return_value = types.SimpleNamespace(all=lambda: True)
        r = preflight.probe(t, True)
        self.assertEqual(r["status"], "PASS")
        self.assertFalse(r["training_verified"])
        t.cuda.synchronize.assert_called_once_with(0)


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="launch space '")
        self.addCleanup(self.tmp.cleanup)
        self.home = Path(self.tmp.name)
        self.bin = self.home / "bin"
        self.bin.mkdir()
        self.log = self.home / "argv.json"
        self.env = {
            **os.environ,
            "HOME": str(self.home),
            "PATH": str(self.bin) + os.pathsep + os.environ["PATH"],
            "HF_TOKEN": "test-only-credential",
            "TEST_LOG": str(self.log),
        }
        self.stub(
            "screen",
            """import json,os,sys
from pathlib import Path
p=Path(os.environ['TEST_LOG'])
if sys.argv[1:] == ['-ls']:
    if p.exists():
        a=json.loads(p.read_text())['argv']; print('123.'+a[2]+' (Detached)')
    else: sys.exit(1)
else:
    p.write_text(json.dumps({'argv':sys.argv,'token_inherited':bool(os.environ.get('HF_TOKEN_INJECT'))}))
""",
        )
        self.stub("sleep", "")
        self.stub(
            "curl", "import os,sys; sys.exit(int(os.environ.get('HEALTH_EXIT','0')))"
        )
        self.stub("llama-server", "")
        self.stub(
            "python3",
            f"""import os,sys
if len(sys.argv)>1 and sys.argv[1].endswith('gpu_preflight.py'):
    print('{{"status":"fixture"}}'); sys.exit(int(os.environ.get('PREFLIGHT_EXIT','0')))
os.execv({sys.executable!r}, [{sys.executable!r}]+sys.argv[1:])
""",
        )

    def stub(self, name, code):
        p = self.bin / name
        p.write_text(f"#!{sys.executable}\n" + code)
        p.chmod(0o755)

    def run_script(self, name, *args):
        return subprocess.run(
            ["bash", str(ROOT / "tools/a6000-migration" / name), *args],
            env=self.env,
            cwd=self.home,
            text=True,
            capture_output=True,
            timeout=15,
        )

    def test_leg_launch_from_other_directory_does_not_put_token_in_argv(self):
        r = self.run_script("launch_leg.sh", "szl-mesh-bench-5090")
        self.assertEqual(r.returncode, 0, r.stderr)
        data = json.loads(self.log.read_text())
        self.assertTrue(data["token_inherited"])
        self.assertNotIn(self.env["HF_TOKEN"], json.dumps(data))
        self.assertTrue(data["argv"][-2].startswith(str(ROOT)))
        self.assertIn("training success remains unverified", r.stdout)

    def test_failed_preflight_never_launches(self):
        self.env["PREFLIGHT_EXIT"] = "2"
        r = self.run_script("launch_leg.sh", "szl-mesh-bench-5090")
        self.assertNotEqual(r.returncode, 0)
        self.assertFalse(self.log.exists())

    def test_existing_session_is_not_duplicated(self):
        self.run_script("launch_leg.sh", "szl-mesh-bench-5090")
        r = self.run_script("launch_leg.sh", "szl-mesh-bench-5090")
        self.assertEqual(r.returncode, 2)
        self.assertIn("already exists", r.stderr)

    def test_invalid_names_are_rejected(self):
        for script in ["serve_model.sh", "launch_leg.sh"]:
            r = self.run_script(script, "../escape;touch INJECTED", "repo/model")
            self.assertEqual(r.returncode, 2)
        self.assertFalse(self.log.exists())

    def test_missing_source_is_rejected(self):
        self.assertEqual(self.run_script("serve_model.sh", "model").returncode, 2)
        self.assertFalse(self.log.exists())

    def test_gguf_path_is_one_literal_argument(self):
        src = self.home / "weights ' $(touch INJECTED).gguf"
        src.touch()
        r = self.run_script("serve_model.sh", "model", str(src))
        self.assertEqual(r.returncode, 0, r.stderr)
        argv = json.loads(self.log.read_text())["argv"]
        self.assertIn(str(src), argv)
        self.assertIn('exec "$@"', argv[5])
        self.assertFalse((self.home / "INJECTED").exists())

    def test_health_failure_does_not_report_ready(self):
        self.env["HEALTH_EXIT"] = "22"
        r = self.run_script("serve_model.sh", "model", "org/checkpoint")
        self.assertEqual(r.returncode, 4, r.stderr)
        self.assertNotIn("READY:", r.stdout)
        self.assertIn("READINESS UNVERIFIED", r.stderr)


if __name__ == "__main__":
    unittest.main()

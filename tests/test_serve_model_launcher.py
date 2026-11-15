# SPDX-License-Identifier: Apache-2.0
"""Offline launcher contracts: screen and model binaries are always fake."""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools/a6000-migration/serve_model.sh"


@unittest.skipUnless(os.name == "posix" and shutil.which("bash"), "requires POSIX bash")
class ServeModelContracts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.home = Path(self.temp.name)
        self.bin = self.home / "bin"
        self.bin.mkdir()
        for command in ["mkdir", "grep"]:
            (self.bin / command).symlink_to(shutil.which(command))
        (self.bin / "python3").symlink_to(sys.executable)
        for command in ["sleep", "llama-server"]:
            stub = self.bin / command
            stub.write_text("#!/bin/sh\nexit 0\n")
            stub.chmod(0o755)
        screen = self.bin / "screen"
        screen.write_text(
            f"#!{sys.executable}\n"
            "import json, os, sys\nfrom pathlib import Path\n"
            "if sys.argv[1:] == ['-ls']:\n"
            "    print(os.environ['TEST_SCREEN_SESSION'])\n"
            "else:\n"
            "    Path(os.environ['TEST_SCREEN_ARGS']).write_text(json.dumps(sys.argv[1:]))\n"
        )
        screen.chmod(0o755)

    def run_script(self, model="fixture", source="fixture.gguf", **extra):
        env = {
            "HOME": str(self.home),
            "PATH": str(self.bin),
            "TEST_SCREEN_ARGS": str(self.home / "screen-args.json"),
            "TEST_SCREEN_SESSION": "serve-" + model,
            **extra,
        }
        return subprocess.run(
            [shutil.which("bash"), str(SCRIPT), model, source],
            env=env,
            check=False,
            cwd=self.home,
            capture_output=True,
            text=True,
            timeout=10,
        )

    def test_remote_bind_without_key_is_refused_before_any_boot_or_server(self):
        result = self.run_script(HOST="0.0.0.0")
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertFalse((self.home / "szl-serve").exists())
        self.assertFalse((self.home / "screen-args.json").exists())

    def test_quoted_values_are_data_in_the_boot_receipt_and_server_arguments(self):
        model = "fixture'quoted"
        source = "a 'quoted' checkpoint.gguf"
        result = self.run_script(model, source)
        self.assertEqual(result.returncode, 0, result.stderr)
        receipt = json.loads(
            (self.home / "szl-serve" / model / "boot-receipt.json").read_text()
        )
        self.assertEqual((receipt["model"], receipt["source"]), (model, source))
        args = json.loads((self.home / "screen-args.json").read_text())
        self.assertEqual(args[args.index("-m") + 1], source)
        self.assertEqual(args[args.index("--host") + 1], "127.0.0.1")
        self.assertNotIn("--api-key", args)

    def test_explicit_remote_key_is_one_literal_argument(self):
        key = "fixture key 'with quotes'"
        result = self.run_script(HOST="0.0.0.0", SZL_API_KEY=key)
        self.assertEqual(result.returncode, 0, result.stderr)
        args = json.loads((self.home / "screen-args.json").read_text())
        self.assertEqual(args[args.index("--host") + 1], "0.0.0.0")
        self.assertEqual(args[args.index("--api-key") + 1], key)


if __name__ == "__main__":
    unittest.main()

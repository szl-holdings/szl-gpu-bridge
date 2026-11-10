"""Release gates reject wrong versions, stale outputs, and unmerged tags."""

import io
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
import zipfile


TOOLS = Path(__file__).resolve().parents[1] / "tools"


class ArtifactContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "tools").mkdir()
        (self.root / "szl").mkdir()
        self.script = self.root / "tools/verify_release.py"
        shutil.copy(TOOLS / "verify_release.py", self.script)
        self.version = "9.8.7"
        self.source = b'__version__ = "9.8.7"\n'
        self.license = b"Synthetic release-contract license fixture\n"
        (self.root / "szl/__init__.py").write_bytes(self.source)
        (self.root / "LICENSE").write_bytes(self.license)
        self.dist = self.root / "dist"
        self.dist.mkdir()
        self.build_fixture()

    def build_fixture(self, metadata_version="9.8.7", license_bytes=None):
        metadata = (
            "Metadata-Version: 2.4\nName: szl\n"
            f"Version: {metadata_version}\nLicense-Expression: Apache-2.0\n"
            "License-File: LICENSE\n\n"
        ).encode()
        license_bytes = self.license if license_bytes is None else license_bytes
        prefix = f"szl-{self.version}"
        with zipfile.ZipFile(self.dist / f"{prefix}-py3-none-any.whl", "w") as archive:
            archive.writestr(f"{prefix}.dist-info/METADATA", metadata)
            archive.writestr(f"{prefix}.dist-info/licenses/LICENSE", license_bytes)
            archive.writestr("szl/__init__.py", self.source)
        with tarfile.open(self.dist / f"{prefix}.tar.gz", "w:gz") as archive:
            for name, content in {
                "PKG-INFO": metadata,
                "LICENSE": license_bytes,
                "szl/__init__.py": self.source,
                "tests/test_fail_closed.py": b"# regression fixture\n",
            }.items():
                info = tarfile.TarInfo(f"{prefix}/{name}")
                info.size = len(content)
                archive.addfile(info, io.BytesIO(content))

    def verify(self, *extra):
        return subprocess.run(
            [sys.executable, str(self.script), "--dist", str(self.dist), *extra],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_matching_release_passes(self):
        result = self.verify("--tag", "szl-v9.8.7")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_mismatched_tag_fails(self):
        result = self.verify("--tag", "szl-v9.8.6")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Tag must match", result.stderr)

    def test_stale_artifact_directory_fails(self):
        (self.dist / "szl-0.1.0.tar.gz").write_bytes(b"old artifact")
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("stale artifacts are forbidden", result.stderr)

    def test_correct_filename_cannot_hide_wrong_metadata(self):
        self.build_fixture(metadata_version="0.1.0")
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Artifact name/version mismatch", result.stderr)

    def test_embedded_license_must_match(self):
        self.build_fixture(license_bytes=b"incorrect license")
        result = self.verify()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Embedded license differs", result.stderr)


class ReleaseRefContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.git("init", "--initial-branch=main")
        self.git("commit", "--allow-empty", "-m", "main baseline")
        self.main = self.git("rev-parse", "HEAD").stdout.strip()
        self.git("update-ref", "refs/remotes/origin/main", self.main)

    def git(self, *args):
        return subprocess.run(
            [
                "git",
                "-c",
                "user.name=Release Test",
                "-c",
                "user.email=test@example.invalid",
                *args,
            ],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=True,
        )

    def verify(self, ref):
        return subprocess.run(
            ["bash", str(TOOLS / "check_release_ref.sh"), ref],
            cwd=self.root,
            capture_output=True,
            text=True,
            check=False,
        )

    def test_exact_main_tag_passes(self):
        self.git("tag", "szl-v9.8.7")
        result = self.verify("refs/tags/szl-v9.8.7")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_tag_on_unmerged_commit_fails(self):
        self.git("commit", "--allow-empty", "-m", "unmerged")
        self.git("tag", "szl-v9.8.7")
        result = self.verify("refs/tags/szl-v9.8.7")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("not in the fetched origin/main history", result.stderr)

    def test_tag_checkout_mismatch_fails(self):
        self.git("tag", "szl-v9.8.7")
        self.git("commit", "--allow-empty", "-m", "different checkout")
        result = self.verify("refs/tags/szl-v9.8.7")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("does not match", result.stderr)

    def test_branch_dispatch_only_validates(self):
        result = self.verify("refs/heads/candidate")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("publication is disabled", result.stdout)


if __name__ == "__main__":
    unittest.main()

"""Check exact artifact versions and embedded license before any publication."""

import argparse
import ast
from email.parser import BytesParser
from pathlib import Path
import tarfile
import zipfile


def source_version(content):
    for node in ast.parse(content).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        ):
            version = ast.literal_eval(node.value)
            if isinstance(version, str) and version:
                return version
    raise ValueError("Missing literal SDK version")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(dist, tag=None):
    sdk = Path(__file__).resolve().parents[1]
    version = source_version((sdk / "szl/__init__.py").read_text())
    if tag is not None:
        require(tag == f"szl-v{version}", f"Tag must match szl-v{version}")
    expected = {f"szl-{version}-py3-none-any.whl", f"szl-{version}.tar.gz"}
    require(
        {p.name for p in dist.iterdir()} == expected,
        "Expected exactly one matching wheel and sdist; stale artifacts are forbidden",
    )
    license_bytes = (sdk / "LICENSE").read_bytes()
    for artifact in sorted(dist.iterdir()):
        if artifact.suffix == ".whl":
            with zipfile.ZipFile(artifact) as archive:
                metadata = archive.read(f"szl-{version}.dist-info/METADATA")
                embedded_license = archive.read(
                    f"szl-{version}.dist-info/licenses/LICENSE"
                )
                embedded_source = archive.read("szl/__init__.py")
        else:
            with tarfile.open(artifact) as archive:
                prefix = f"szl-{version}"
                metadata = archive.extractfile(f"{prefix}/PKG-INFO").read()
                embedded_license = archive.extractfile(f"{prefix}/LICENSE").read()
                embedded_source = archive.extractfile(
                    f"{prefix}/szl/__init__.py"
                ).read()
                require(
                    f"{prefix}/tests/test_fail_closed.py" in archive.getnames(),
                    "sdist must include the fail-closed regression tests",
                )
        info = BytesParser().parsebytes(metadata)
        require(
            info["Name"] == "szl" and info["Version"] == version,
            "Artifact name/version mismatch",
        )
        require(
            source_version(embedded_source) == version,
            "Embedded module version mismatch",
        )
        require(info["License-Expression"] == "Apache-2.0", "License metadata mismatch")
        require(
            "LICENSE" in info.get_all("License-File", []), "Missing license metadata"
        )
        require(
            embedded_license == license_bytes, "Embedded license differs from source"
        )
    print(
        f"Verified szl {version}: one wheel, one sdist, matching versions and license"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--tag")
    args = parser.parse_args()
    validate(args.dist, args.tag)

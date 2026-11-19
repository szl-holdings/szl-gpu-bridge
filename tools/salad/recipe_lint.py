#!/usr/bin/env python3
"""recipe_lint.py — the Salad lane admission gate (Golden Recipe v2, fail-closed).

Every check here corresponds to a measured failure class recorded in estate
receipts (see golden_recipe_v2.md). The gate is deliberately STATIC and
DETERMINISTIC: it does not talk to Salad or the Hub, it never prints secrets,
and it does not prove a lane will train — it refuses to let known-fatal specs
burn a machine slot and a boot cycle in the first place.

Usage:
    recipe_lint.py --spec spec.json          # lint a spec file
    recipe_lint.py --command '...' --name lane-name --image pytorch:... --env HF_TOKEN
    recipe_lint.py ... --json out.json      # write the receipt to a file too

Exit codes: 0 = PASS, 1 = FAIL (reasons printed + receipted), 2 = usage/IO error.
A FAIL lint receipt is itself an action receipt: keep it (near-misses get
receipted too, per estate doctrine).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone

DISPLAY_NAME_RE = re.compile(r"^[ ,\-.0-9A-Za-z]+$")
AUTH_PATTERNS = [
    re.compile(r"authorization\s*:\s*bearer", re.I),
    re.compile(r"bearer\s+[A-Za-z0-9_\-\.=+/]{16,}", re.I),
    re.compile(r"salad[-_ ]?api[-_ ]?key", re.I),
    re.compile(r"(hf|gh|sk)_[A-Za-z0-9]{20,}"),
]
CURL_PATTERNS = [
    re.compile(r"\bcurl\b"),
]

PIN_SETS = {
    # MODERN: the f54v2-proven pin set (fleet lanes that train first-pass).
    "MODERN": [
        "transformers==5.22.2", "peft==0.21.2",
        "huggingface_hub==0.30.2", "pynvml==12.0.0",
    ],
    # LEGACY: the ring-family pin set (compatible with the worker after PR #279).
    "LEGACY": [
        "transformers==4.51.3", "datasets==3.5.0", "peft==0.15.2",
        "trl==0.17.0", "accelerate==1.6.0", "huggingface_hub==0.30.2",
        "safetensors==0.5.3", "hf_transfer==0.1.9", "pynvml==12.0.0",
    ],
}
IMAGE_ALLOWLIST_PREFIXES = ("pytorch:",)  # off-pin images (e.g. 2.14.1-cuda12.6) burned lanes


def lint(name: str, command: str, image: str, env: dict | None,
         pinset: str | None) -> list[dict]:
    findings: list[dict] = []
    env = env or {}

    def fail(rule: str, detail: str):
        findings.append({"severity": "FAIL", "rule": rule, "detail": detail})

    def warn(rule: str, detail: str):
        findings.append({"severity": "WARN", "rule": rule, "detail": detail})

    # R1 display name
    if not name or not DISPLAY_NAME_RE.fullmatch(name):
        fail("R1_DISPLAY_NAME", "name must match ^[ ,-.0-9A-Za-z]+$ (no parens/colons)")

    # R2 no curl USED FOR FETCHING in the command (CURL_FAIL class).
    # The honest presence probe (command -v curl ...) is doctrine-allowed and exempt.
    import re as _re
    command_minus_probes = _re.sub(r"command\s+-v\s+curl[^\n]*", " ", command)
    for pat in CURL_PATTERNS:
        if pat.search(command_minus_probes):
            fail("R2_NO_CURL",
                 "curl fetch usage in worker commands is CURL_FAIL-prone on "
                 "pytorch runtime images; fetch with "
                 "hf_hub_download(..., token=os.environ['HF_TOKEN']) "
                 "(command -v curl presence probes are allowed)")

    # R3 no auth literals in command (create-400 class); token via env
    for pat in AUTH_PATTERNS:
        m = pat.search(command)
        if m:
            fail("R3_NO_AUTH_LITERALS",
                 "auth-like literal in command (never put tokens/bearer strings "
                 "in a Salad create body; rule matches, VALUE NOT PRINTED)")
    if "HF_TOKEN" not in env and "HF_TOKEN" not in command:
        fail("R3_HF_TOKEN_ENV", "HF_TOKEN must be present (env preferred, export in command)")

    # R4 hf_hub fetch marker
    if "hf_hub_download" not in command and (".py" in command or "kit" in command.lower()):
        warn("R4_FETCH_MARKER",
             "command fetches files but no hf_hub_download marker found; "
             "confirm fetch path or add KIT_FETCH_OK/FAIL markers")

    # R5 pin-set completeness (off-pin create class)
    matched_sets = []
    for set_name, pins in PIN_SETS.items():
        if all(p in command for p in pins):
            matched_sets.append(set_name)
    if not matched_sets:
        missing = {s: [p for p in pins if p not in command] for s, pins in PIN_SETS.items()}
        fail("R5_PINSET", "no allowlisted pin-set fully present in command; "
             f"missing per set: {json.dumps(missing)}")
    elif pinset and pinset not in matched_sets:
        fail("R5_PINSET", f"declared pinset {pinset} not satisfied (found {matched_sets})")

    # R6 resilience block (boot fragility class)
    for marker, rule, hint in (
        ("1.1.1.1", "R6_RESOLV", "resolv.conf fixup marker missing"),
        ("8.8.8.8", "R6_RESOLV", "secondary resolver missing"),
        ("PIP_DEFAULT_TIMEOUT", "R6_PIP_RESILIENCE", "PIP_DEFAULT_TIMEOUT missing"),
        ("PIP_RETRIES", "R6_PIP_RESILIENCE", "PIP_RETRIES missing"),
    ):
        if marker not in command:
            if rule == "R6_RESOLV" and marker == "8.8.8.8":
                warn(rule, hint)
            else:
                fail(rule, hint)

    # R7 image allowlist
    if image and not image.startswith(IMAGE_ALLOWLIST_PREFIXES):
        warn("R7_IMAGE", f"image '{image}' outside allowlist prefixes; "
             "only proven pytorch images have first-pass receipts")

    # R8 evidence markers: receipts channel + heartbeats + exit markers
    if not re.search(r"(SZL_RECEIPTS_REPO|receipts[-_]repo|mesh/)", command):
        warn("R8_RECEIPTS", "no receipts-repo channel marker in command")
    if not re.search(r"(heartbeat-seconds|HB=\d|hb\d*\.py|stage0|STAGE0)", command):
        warn("R8_HEARTBEAT", "no heartbeat/stage0 evidence marker in command")
    if not re.search(r"(INNER_EXIT|EVAL_EXIT)", command):
        warn("R8_EXIT_MARKERS", "no INNER_EXIT/EVAL_EXIT marker; exit codes stay unverifiable")

    # R9 park discipline acknowledgement: a completed lane MUST be parked
    if "salad_worker.py" in command and "--max-steps" in command:
        warn("R9_PARK_DISCIPLINE",
             "training lane: ship a completion watcher that parks the group "
             "(POST stop -> merge-patch replicas 0 -> readback) after the final receipt")
    return findings


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", help="JSON spec: {name, image, command, env, pinset}")
    ap.add_argument("--name"), ap.add_argument("--image", default="")
    ap.add_argument("--command", default="")
    ap.add_argument("--env", default="", help="KEY=VALUE,KEY=VALUE (values NOT printed)")
    ap.add_argument("--pinset", choices=sorted(PIN_SETS), default=None)
    ap.add_argument("--json", dest="json_out", default="", help="write receipt JSON here")
    a = ap.parse_args()

    name, command, image, env, pinset = a.name or "", a.command, a.image, {}, a.pinset
    if a.spec:
        try:
            with open(a.spec, encoding="utf-8") as f:
                spec = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            print(f"LINT_ERROR: cannot read spec: {type(e).__name__}", file=sys.stderr)
            return 2
        name = spec.get("name", name)
        command = spec.get("command", command)
        image = spec.get("image", image)
        env = spec.get("env", env) or {}
        pinset = spec.get("pinset", pinset)
    if a.env and not a.spec:
        for kv in a.env.split(","):
            if "=" in kv:
                k, _ = kv.split("=", 1)
                env[k.strip()] = "REDACTED"

    findings = lint(name, command, image, env, pinset)
    hard = [f for f in findings if f["severity"] == "FAIL"]
    verdict = "PASS" if not hard else "FAIL"
    receipt = {
        "schema": "szl.recipe-lint/v1",
        "verdict": verdict,
        "lane": name,
        "fail_count": len(hard),
        "warn_count": len(findings) - len(hard),
        "findings": findings,
        "ts": datetime.now(timezone.utc).isoformat(),
        "scope": "STATIC_ADMISSION_GATE_NOT_TRAINING_EVIDENCE",
    }
    out = json.dumps(receipt, indent=2)
    print(out)
    if a.json_out:
        with open(a.json_out, "w", encoding="utf-8") as f:
            f.write(out + "\n")
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

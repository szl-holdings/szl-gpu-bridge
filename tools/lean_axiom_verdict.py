#!/usr/bin/env python3
"""lean_axiom_verdict.py - kernel-level verdict for one Lean 4 declaration.

Replaces "grep the theorem's name in a file whose build succeeded", which a
`sorry` proof or an added axiom also satisfies. Input is the output of
`#print axioms <name>` produced by `lake env lean`. Fail closed: anything that
cannot be read is UNAVAILABLE, never PROVEN. Standard library only.

Verdicts
  PROVEN           build ok, audit ok, axioms within {propext, Classical.choice, Quot.sound}
  CONJECTURE       the declaration is itself an axiom
  UNPROVEN         depends on sorryAx (a `sorry` somewhere in the proof)
  AXIOM_DEPENDENT  depends on an axiom outside the Lean core set
  UNAVAILABLE      build failed, audit failed, or no `#print axioms` line found
"""

import json
import re
import sys

ALLOWED = frozenset({"propext", "Classical.choice", "Quot.sound"})


def parse_axioms(output, name):
    """Return the axiom list for `name`, [] for none, None if no line found."""
    q = re.escape(name)
    if re.search(r"'%s' does not depend on any axioms" % q, output):
        return []
    m = re.search(r"'%s' depends on axioms:\s*\[(.*?)\]" % q, output, re.S)
    if not m:
        return None
    return [a.strip() for a in re.split(r"[,\n]", m.group(1)) if a.strip()]


def verdict(build_rc, audit_rc, output, name):
    if build_rc != 0:
        return {"state": "UNAVAILABLE", "reason": "build failed", "axioms": None}
    if audit_rc != 0:
        return {
            "state": "UNAVAILABLE",
            "reason": "axiom audit exited non-zero",
            "axioms": None,
        }
    ax = parse_axioms(output, name)
    if ax is None:
        return {
            "state": "UNAVAILABLE",
            "reason": "no #print axioms line for declaration",
            "axioms": None,
        }
    if name in ax:
        return {"state": "CONJECTURE", "reason": "declared as an axiom", "axioms": ax}
    if "sorryAx" in ax:
        return {"state": "UNPROVEN", "reason": "proof depends on sorry", "axioms": ax}
    extra = sorted(set(ax) - ALLOWED)
    if extra:
        return {
            "state": "AXIOM_DEPENDENT",
            "reason": "non-core axioms: " + ", ".join(extra),
            "axioms": ax,
        }
    return {
        "state": "PROVEN",
        "reason": "kernel-checked; axioms within Lean core set",
        "axioms": ax,
    }


def _selftest():
    n = "A.b"
    cases = [
        (
            (
                0,
                0,
                "'A.b' depends on axioms: [propext, Classical.choice, Quot.sound]",
                n,
            ),
            "PROVEN",
        ),
        ((0, 0, "'A.b' does not depend on any axioms", n), "PROVEN"),
        (
            (
                0,
                0,
                "'A.b' depends on axioms: [propext,\n Classical.choice,\n Quot.sound]",
                n,
            ),
            "PROVEN",
        ),
        ((0, 0, "'A.b' depends on axioms: [propext, sorryAx]", n), "UNPROVEN"),
        ((0, 0, "'A.b' depends on axioms: [propext, A.cheat]", n), "AXIOM_DEPENDENT"),
        ((0, 0, "'A.b' depends on axioms: [A.b]", n), "CONJECTURE"),
        ((1, 0, "'A.b' depends on axioms: [propext]", n), "UNAVAILABLE"),
        ((0, 1, "'A.b' depends on axioms: [propext]", n), "UNAVAILABLE"),
        ((0, 0, "", n), "UNAVAILABLE"),
        ((0, 0, "'A.bb' depends on axioms: [propext]", n), "UNAVAILABLE"),
        ((0, 0, "'A.c' does not depend on any axioms", n), "UNAVAILABLE"),
    ]
    bad = [
        (a, want, verdict(*a)["state"])
        for a, want in cases
        if verdict(*a)["state"] != want
    ]
    print(json.dumps({"selftest_cases": len(cases), "failed": len(bad), "detail": bad}))
    return 1 if bad else 0


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "--selftest":
        sys.exit(_selftest())
    if len(sys.argv) != 5:
        print(
            "usage: lean_axiom_verdict.py BUILD_RC AUDIT_RC AUDIT_LOG DECL_NAME | --selftest",
            file=sys.stderr,
        )
        sys.exit(2)
    try:
        out = open(sys.argv[3], encoding="utf-8", errors="replace").read()
    except OSError:
        out = ""
    v = verdict(int(sys.argv[1]), int(sys.argv[2]), out, sys.argv[4])
    print(json.dumps(v, sort_keys=True))
    sys.exit(0 if v["state"] in ("PROVEN", "CONJECTURE") else 1)

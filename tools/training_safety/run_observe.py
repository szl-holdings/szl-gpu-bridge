"""Observe-only launcher for watch_safe.py (2026-11-19, Yachay).

Belt and braces on top of watch_safe's default: refuses --apply, disables
Salad PATCH entirely, and disables Hub evidence collection so that no model
weights are downloaded or copied while owner config fields are blank.
Credentials are read from the files named in the config by watch_safe/salad_io
at request time; nothing is put on the command line or printed.
"""

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

if "--apply" in sys.argv:
    print("run_observe.py refuses --apply (observe-only launcher)")
    sys.exit(3)

import salad_io  # noqa: E402
import watch_safe  # noqa: E402


def _no_patch(self, *a, **k):
    raise RuntimeError("PATCH disabled in observe-only launcher")


def _no_collect(group_cfg, token_file, **k):
    return None, "OBSERVE_ONLY_NO_DOWNLOAD"


salad_io.Salad.patch = _no_patch
_orig_cycle = watch_safe.cycle


def _cycle(*a, **k):
    k["apply"] = False
    k["collect"] = _no_collect
    return _orig_cycle(*a, **k)


watch_safe.cycle = _cycle

if __name__ == "__main__":
    sys.exit(watch_safe.main(sys.argv[1:]))

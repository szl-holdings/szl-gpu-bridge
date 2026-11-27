"""CPU-only simulation of the multi-model swap pool. No GPU, no R2, no spend.

Fake runs are tiny Python children that sleep, append eval losses from a fixed
curve and honor the pool's checkpoint-request contract. R2 is a recording mock.
"""

from datetime import datetime, timedelta
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import time
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]


def _load(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


pool = _load("szl_gpu_pool_runner", "tools/gpu/pool_runner.py")
r2 = _load("szl_r2_checkpoint_store", "tools/gpu/r2_checkpoint_store.py")

REVISION = "a" * 40
SECRET = "r2-secret-value-must-never-leak"

FAKE_CHILD = r"""
import json, os, sys, time
cfg = json.loads(sys.argv[1])
run_dir = os.environ["SZL_POOL_RUN_DIR"]
with open(os.path.join(run_dir, "env_keys.json"), "w") as f:
    json.dump(sorted(os.environ), f)
metrics = os.environ["SZL_POOL_METRICS_FILE"]
request = os.environ["SZL_POOL_CHECKPOINT_REQUEST"]
seen = set()
curve = cfg["curve"]
i = 0
while True:
    time.sleep(cfg.get("step_s", 0.05))
    if i < len(curve):
        loss = curve[i]
    elif cfg.get("forever"):
        loss = curve[-1]
    else:
        break
    i += 1
    with open(metrics, "a") as f:
        f.write(json.dumps({"eval_loss": loss, "step": i}) + "\n")
    if not cfg.get("ignore_checkpoint") and os.path.exists(request):
        try:
            req = json.load(open(request))
        except ValueError:
            continue
        if req["seq"] not in seen:
            seen.add(req["seq"])
            os.makedirs(req["dir"], exist_ok=True)
            with open(os.path.join(req["dir"], "adapter.bin"), "wb") as f:
                f.write(b"w" * 16)
            open(os.path.join(req["dir"], "COMPLETE"), "w").close()
sys.exit(cfg.get("exit", 0))
"""


class RecordingStore:
    def __init__(self, fail=False):
        self.uploads = []
        self.fail = fail

    def upload_checkpoint(self, run_id, label, checkpoint_dir, work_dir):
        if self.fail:
            raise r2.R2Error("R2_UPLOAD_FAILED:ConnectTimeout")
        files = sorted(p.name for p in Path(checkpoint_dir).iterdir())
        self.uploads.append((run_id, label, files))
        key = f"pool/{run_id}/{label}.tar"
        return {"uri": f"r2://szl-mesh-artifacts/{key}", "sha256": "0" * 64, "bytes": 1}


class PoolHarness(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.base = Path(self.tmp.name)
        self.repo = self.base / "repo"
        (self.repo / "jobspecs").mkdir(parents=True)
        (self.repo / "queue" / "pending").mkdir(parents=True)
        (self.repo / "laptop").mkdir()
        (self.repo / "laptop" / "dispatcher.py").write_text("raise SystemExit(3)\n")
        self.child = self.base / "fake_child.py"
        self.child.write_text(FAKE_CHILD)
        self.ledger = self.base / "spend.jsonl"
        self.ledger.write_text("")
        self.work = self.base / "work"
        self.configs = {}
        self.max_seen = 0

    def node(self, **overrides):
        deadline = datetime.now(pool.ET) + timedelta(hours=72)
        node = {
            "schema": pool.NODE_SCHEMA,
            "node_id": "massed-h100-test",
            "provider": "massed",
            "gpu_type": "H100-80GB",
            "node_deadline": deadline.isoformat(),
            "deadline_margin_s": 3,
            "checkpoint_timeout_s": 2,
            "checkpoint_interval_s": 14400,
            "terminate_grace_s": 0.5,
            "vram_headroom_fraction": 0.1,
            "max_concurrent": 4,
            "devices": [
                {"id": "0", "vram_gb": 80, "isolation": "device", "usd_per_hour": 2.0},
                {"id": "1", "vram_gb": 80, "isolation": "device", "usd_per_hour": 2.0},
            ],
            "caps": {"per_run_usd": 100, "per_day_usd": 500, "per_month_usd": 5000},
            "spend_ledger": str(self.ledger),
            "r2": {"bucket": "szl-mesh-artifacts", "prefix": "pool"},
        }
        node.update(overrides)
        return node

    def entry(
        self, run_id, curve, *, vram=30, max_hours=10, k=3, threshold=0.05, **cfg
    ):
        jobspec = self.repo / "jobspecs" / f"{run_id}-reviewed.json"
        envelope = self.repo / "queue" / "pending" / f"job-{run_id}.json"
        jobspec.write_text(json.dumps({"run": run_id}))
        envelope.write_text(json.dumps({"payload": run_id}))
        self.configs[run_id] = dict(cfg, curve=curve)
        return {
            "run_id": run_id,
            "model": f"model-{run_id}",
            "entrypoint": "laptop/dispatcher.py",
            "reviewed_jobspec": f"jobspecs/{run_id}-reviewed.json",
            "signed_envelope": f"queue/pending/job-{run_id}.json",
            "reviewed_binding": {
                "jobspec_sha256": hashlib.sha256(jobspec.read_bytes()).hexdigest(),
                "envelope_sha256": hashlib.sha256(envelope.read_bytes()).hexdigest(),
                "execution_bridge_revision": REVISION,
            },
            "vram_gb": vram,
            "max_hours": max_hours,
            "plateau": {"k": k, "threshold": threshold},
        }

    def runner(self, node, entries, store=None, dispatch_caps=(), **kw):
        harness = self

        class Counting(pool.PoolRunner):
            def _launch(self, *a, **k):
                super()._launch(*a, **k)
                harness.max_seen = max(harness.max_seen, len(self.running))

        self.log = io.StringIO()
        self.store = store or RecordingStore()
        env = dict(os.environ, R2_ACCESS_KEY_ID="kid", R2_SECRET_ACCESS_KEY=SECRET)
        return Counting(
            node,
            entries,
            repo_root=self.repo,
            work_dir=self.work,
            store=self.store,
            dispatch_caps=dispatch_caps,
            execution_revision=kw.pop("revision", REVISION),
            command_factory=lambda run: [
                sys.executable,
                str(self.child),
                json.dumps(self.configs[run.run_id]),
            ],
            poll_s=0.05,
            environ=env,
            log_stream=self.log,
            **kw,
        )

    def lines(self):
        return [json.loads(x) for x in self.log.getvalue().splitlines()]

    def swaps(self):
        return [x for x in self.lines() if x["event"] == "swap"]


class ConcurrencyAndQueue(PoolHarness):
    def test_exclusive_devices_cap_concurrency_and_queue_advances(self):
        entries = [self.entry(f"r{i}", [1.0, 0.9, 0.8]) for i in range(5)]
        summary = self.runner(self.node(), entries).run()
        self.assertEqual(self.max_seen, 2)  # two exclusive devices
        self.assertEqual(len(summary["finished"]), 5)
        self.assertEqual({r["reason"] for r in summary["finished"]}, {"done"})
        admits = [x["run_id"] for x in self.lines() if x["event"] == "admit"]
        self.assertEqual(admits, ["r0", "r1", "r2", "r3", "r4"])  # FIFO
        self.assertTrue(any(s["next_run_id"] for s in self.swaps()))

    def test_shared_gpu_admits_only_while_estimated_vram_fits(self):
        node = self.node(
            devices=[
                {"id": "0", "vram_gb": 80, "isolation": "shared", "usd_per_hour": 3}
            ],
            allow_unenforced_shared_memory=True,
        )
        entries = [self.entry(f"s{i}", [1.0, 0.9, 0.8], vram=30) for i in range(4)]
        self.runner(node, entries).run()
        # 30 GB x 1.1 headroom: two fit in 80 GB, a third (99 GB) does not.
        self.assertEqual(self.max_seen, 2)
        env = json.loads((self.work / "runs/s0/env_keys.json").read_text())
        self.assertIn("SZL_POOL_CUDA_MEMORY_FRACTION", env)

    def test_max_concurrent_is_a_hard_limit(self):
        node = self.node(
            devices=[
                {"id": "0", "vram_gb": 80, "isolation": "shared", "usd_per_hour": 3}
            ],
            allow_unenforced_shared_memory=True,
            max_concurrent=1,
        )
        entries = [self.entry(f"m{i}", [1.0, 0.9], vram=5) for i in range(3)]
        self.runner(node, entries).run()
        self.assertEqual(self.max_seen, 1)

    def test_run_that_can_never_fit_is_refused_not_launched(self):
        entries = [self.entry("huge", [1.0], vram=79), self.entry("ok", [1.0])]
        summary = self.runner(self.node(), entries).run()
        self.assertEqual(
            summary["refused"], [{"run_id": "huge", "reason": "VRAM_NEVER_FITS"}]
        )
        self.assertFalse((self.work / "runs/huge").exists())


class PlateauAndCheckpoints(PoolHarness):
    def test_plateau_evicts_with_r2_checkpoint_and_loads_next(self):
        node = self.node(max_concurrent=1)
        flat = [2.0, 1.5, 1.2, 1.19, 1.19, 1.19, 1.19]
        entries = [
            self.entry("flat", flat, forever=True, k=3, threshold=0.05),
            self.entry("next", [1.0, 0.5]),
        ]
        self.runner(node, entries).run()
        swap = self.swaps()[0]
        self.assertEqual(swap["run_id"], "flat")
        self.assertEqual(swap["reason"], "plateau")
        self.assertEqual(swap["next_run_id"], "next")
        self.assertTrue(
            swap["checkpoint_uri"].startswith("r2://szl-mesh-artifacts/pool/flat/")
        )
        run_id, label, files = self.store.uploads[0]
        self.assertEqual((run_id, files), ("flat", ["COMPLETE", "adapter.bin"]))
        self.assertTrue(label.endswith("-plateau"))
        receipt = json.loads((self.work / "receipts/flat.evicted.json").read_text())
        self.assertEqual(receipt["checkpoint"]["mode"], "acknowledged")
        self.assertEqual(receipt["entrypoint"], "laptop/dispatcher.py")

    def test_plateau_detector(self):
        self.assertFalse(pool.plateaued([1.0, 0.9, 0.8], 0.05, 3, 4))
        self.assertTrue(pool.plateaued([1.0, 0.99, 0.99, 0.99], 0.05, 3, 4))
        self.assertFalse(pool.plateaued([1.0, 0.9, 0.8, 0.7], 0.05, 3, 4))
        self.assertTrue(pool.plateaued([1.0, 0.5, 0.6, 0.55, 0.52], 0.0, 3, 4))

    def test_periodic_checkpoint_uploads_without_evicting(self):
        node = self.node(checkpoint_interval_s=0.3)
        entries = [self.entry("p", [1.0 - i * 0.05 for i in range(14)])]
        self.runner(node, entries).run()
        events = [x for x in self.lines() if x["event"] == "checkpoint"]
        self.assertGreaterEqual(len(events), 1)
        self.assertEqual(events[0]["reason"], "periodic")
        self.assertEqual(self.swaps()[0]["reason"], "done")

    def test_unacknowledged_checkpoint_with_nothing_on_disk_evicts(self):
        node = self.node(checkpoint_interval_s=0.2, checkpoint_timeout_s=0.4)
        entries = [
            self.entry(
                "deaf",
                [1.0, 0.8, 0.6],
                forever=True,
                ignore_checkpoint=True,
                k=50,
                threshold=0,
            )
        ]
        self.runner(node, entries).run()
        swap = self.swaps()[0]
        self.assertEqual(swap["reason"], "checkpoint_failed")
        self.assertIsNone(swap["checkpoint_uri"])

    def test_upload_failure_is_recorded_not_hidden(self):
        node = self.node(max_concurrent=1)
        entries = [self.entry("flat", [1.0, 1.0, 1.0, 1.0, 1.0], forever=True)]
        self.runner(node, entries, store=RecordingStore(fail=True)).run()
        swap = self.swaps()[0]
        self.assertIsNone(swap["checkpoint_uri"])
        self.assertTrue(swap["checkpoint_error"].startswith("R2_UPLOAD_FAILED"))


class DeadlineStop(PoolHarness):
    def test_hard_stop_checkpoints_every_run_before_node_deadline(self):
        deadline = datetime.now(pool.ET) + timedelta(seconds=4.5)
        node = self.node(node_deadline=deadline.isoformat(), max_concurrent=2)
        entries = [
            self.entry(
                f"long{i}", [1.0, 0.9, 0.8, 0.7], forever=True, k=500, threshold=0
            )
            for i in range(3)
        ]
        started = time.time()
        summary = self.runner(node, entries).run()
        self.assertLess(time.time(), deadline.timestamp())
        self.assertLess(time.time() - started, 4.5)
        reasons = sorted(r["reason"] for r in summary["finished"])
        self.assertEqual(reasons, ["node_deadline", "node_deadline"])
        self.assertTrue(all(s["checkpoint_uri"] for s in self.swaps()))
        self.assertEqual(summary["refused"][0]["run_id"], "long2")
        self.assertIn("NOT_ADMITTED_POOL_STOPPED", summary["refused"][0]["reason"])

    def test_passed_deadline_refuses_the_whole_node(self):
        past = (datetime.now(pool.ET) - timedelta(minutes=1)).isoformat()
        with self.assertRaisesRegex(pool.GateError, "NODE_DEADLINE_PASSED"):
            self.runner(self.node(node_deadline=past), [self.entry("x", [1.0])])

    def test_deadline_needs_offset_and_margin_above_checkpoint_timeout(self):
        naive = datetime.now().replace(microsecond=0).isoformat()
        with self.assertRaisesRegex(pool.GateError, "NODE_DEADLINE_NEEDS_OFFSET"):
            self.runner(self.node(node_deadline=naive), [])
        with self.assertRaisesRegex(pool.GateError, "DEADLINE_MARGIN_BELOW"):
            self.runner(self.node(checkpoint_timeout_s=5), [])

    def test_run_timebox_is_clamped_to_the_node_hard_stop(self):
        deadline = datetime.now(pool.ET) + timedelta(hours=2)
        r = self.runner(
            self.node(node_deadline=deadline.isoformat()),
            [self.entry("t", [1.0], max_hours=10)],
        )
        decision = r.plan()[0]
        self.assertLessEqual(decision["hard_timebox_h"], 2.0)


class CostCaps(PoolHarness):
    def test_projected_run_cost_above_cap_is_refused(self):
        # 2 USD/h x 60 h = 120 USD > 100 USD per-run cap.
        summary = self.runner(
            self.node(), [self.entry("big", [1.0], max_hours=60)]
        ).run()
        self.assertEqual(
            summary["refused"][0]["reason"], "PROJECTED_COST_ABOVE_RUN_CAP"
        )
        self.assertFalse((self.work / "runs/big").exists())

    def test_day_and_month_caps_count_ledger_and_running_reservations(self):
        now = datetime.now(pool.ET).isoformat()
        self.ledger.write_text(json.dumps({"at_et": now, "usd": 470}) + "\n")
        entries = [
            self.entry("a", [1.0, 0.9], max_hours=10),
            self.entry("b", [1.0], max_hours=10),
        ]
        summary = self.runner(self.node(), entries).run()
        # 470 + 20 fits 500; 470 + 20 reserved + 20 does not.
        self.assertEqual([r["run_id"] for r in summary["finished"]], ["a"])
        self.assertEqual(
            summary["refused"][0]["reason"], "PROJECTED_COST_ABOVE_DAY_CAP"
        )

    def test_month_cap(self):
        earlier = datetime.now(pool.ET).replace(day=1, hour=0, minute=0, second=1)
        if earlier.date() == datetime.now(pool.ET).date():
            earlier = earlier.replace(second=0)
        self.ledger.write_text(
            json.dumps({"at_et": earlier.isoformat(), "usd": 4990}) + "\n"
        )
        node = self.node(
            caps={"per_run_usd": 100, "per_day_usd": 10000, "per_month_usd": 5000}
        )
        summary = self.runner(node, [self.entry("m", [1.0], max_hours=10)]).run()
        self.assertEqual(
            summary["refused"][0]["reason"], "PROJECTED_COST_ABOVE_MONTH_CAP"
        )

    def test_stricter_of_node_dispatch_and_run_caps(self):
        caps, provenance = pool.effective_caps(
            [
                (
                    "node",
                    {"per_run_usd": 100, "per_day_usd": 500, "per_month_usd": 5000},
                ),
                (
                    "dispatch:caps.json",
                    {"per_run_usd": 25, "per_day_usd": 100, "per_month_usd": 1500},
                ),
                ("run:x", {"per_run_usd": 40}),
            ]
        )
        self.assertEqual(
            caps, {"per_run_usd": 25.0, "per_day_usd": 100.0, "per_month_usd": 1500.0}
        )
        self.assertEqual(provenance["per_run_usd"], "dispatch:caps.json")

    def test_dispatch_caps_file_tightens_admission(self):
        path = self.base / "caps.json"
        path.write_text(
            json.dumps(
                {
                    "schema": "szl.dispatch.org-caps/v1",
                    "fail_closed": True,
                    "per_run_usd": 25,
                    "per_day_usd": 100,
                    "per_month_usd": 1500,
                }
            )
        )
        caps = [("dispatch:caps.json", pool.load_dispatch_caps(path))]
        # 2 USD/h x 15 h = 30 USD: fits the node's 100 but not dispatch's 25.
        summary = self.runner(
            self.node(), [self.entry("d", [1.0], max_hours=15)], dispatch_caps=caps
        ).run()
        self.assertEqual(
            summary["refused"][0]["reason"], "PROJECTED_COST_ABOVE_RUN_CAP"
        )

    def test_caps_come_from_config_not_constants(self):
        source = (ROOT / "tools/gpu/pool_runner.py").read_text()
        for literal in ("25.0", "100.0", "500.0", "5000", "1500", "375"):
            self.assertNotIn(literal, source)
        node = self.node(
            caps={"per_run_usd": 200, "per_day_usd": 500, "per_month_usd": 5000}
        )
        summary = self.runner(node, [self.entry("cfg", [1.0], max_hours=60)]).run()
        self.assertEqual(summary["finished"][0]["run_id"], "cfg")

    def test_missing_or_invalid_cap_refuses(self):
        with self.assertRaisesRegex(pool.GateError, "CAP_MISSING:per_month_usd"):
            self.runner(self.node(caps={"per_run_usd": 1, "per_day_usd": 1}), [])
        with self.assertRaisesRegex(pool.GateError, "CAP_INVALID"):
            self.runner(
                self.node(
                    caps={"per_run_usd": -1, "per_day_usd": 1, "per_month_usd": 1}
                ),
                [],
            )

    def test_runtime_run_cap_watchdog_evicts(self):
        node = self.node(
            caps={"per_run_usd": 0.002, "per_day_usd": 500, "per_month_usd": 5000}
        )
        # Rate 36 USD/h = 0.01 USD/s: max_hours tiny so admission fits, cap hits at 0.2 s.
        node["devices"][0]["usd_per_hour"] = 36
        entries = [
            self.entry(
                "burn",
                [1.0, 0.9, 0.8, 0.7],
                forever=True,
                max_hours=0.00005,
                k=500,
                threshold=0,
            )
        ]
        self.runner(node, entries).run()
        self.assertIn(self.swaps()[0]["reason"], ("cap_run", "run_timebox"))


class BindingAndNodeGates(PoolHarness):
    def test_envelope_drift_is_refused_before_launch(self):
        entry = self.entry("drift", [1.0])
        (self.repo / entry["signed_envelope"]).write_text("tampered")
        summary = self.runner(self.node(), [entry]).run()
        self.assertEqual(summary["refused"][0]["reason"], "RUN_ENVELOPE_DRIFT")
        self.assertFalse((self.work / "runs/drift").exists())

    def test_execution_revision_must_match(self):
        summary = self.runner(
            self.node(), [self.entry("rev", [1.0])], revision="b" * 40
        ).run()
        self.assertEqual(
            summary["refused"][0]["reason"], "RUN_EXECUTION_REVISION_MISMATCH"
        )
        summary = self.runner(
            self.node(), [self.entry("rev2", [1.0])], revision=None
        ).run()
        self.assertEqual(
            summary["refused"][0]["reason"], "RUN_EXECUTION_REVISION_MISMATCH"
        )

    def test_only_the_dispatcher_entrypoint_and_reviewed_jobspecs(self):
        bad = self.entry("ep", [1.0])
        bad["entrypoint"] = "laptop/runjob_nemo_v3.py"
        draft = self.entry("draft", [1.0])
        draft["reviewed_jobspec"] = "queue/pending/job-draft.json"
        missing = self.entry("nobind", [1.0])
        del missing["reviewed_binding"]
        summary = self.runner(self.node(), [bad, draft, missing]).run()
        self.assertEqual(
            [r["reason"] for r in summary["refused"]],
            [
                "RUN_ENTRYPOINT_NOT_DISPATCHER",
                "RUN_JOBSPEC_NOT_REVIEWED",
                "RUN_BINDING_MISSING",
            ],
        )
        self.assertEqual(summary["finished"], [])

    def test_node_provider_host_and_shared_isolation_gates(self):
        with self.assertRaisesRegex(pool.GateError, "NODE_PROVIDER_NOT_ALLOWED"):
            self.runner(self.node(provider="owner"), [])
        with self.assertRaisesRegex(pool.GateError, "NODE_FORBIDDEN_HOST"):
            self.runner(self.node(node_id="betterwithage-laptop"), [])
        shared = [{"id": "0", "vram_gb": 80, "isolation": "shared", "usd_per_hour": 2}]
        with self.assertRaisesRegex(pool.GateError, "SHARED_GPU_MEMORY_UNENFORCED"):
            self.runner(self.node(devices=shared), [])
        with self.assertRaisesRegex(pool.GateError, "MIG_DEVICE_NEEDS_MIG_UUID"):
            self.runner(
                self.node(
                    devices=[
                        {
                            "id": "0",
                            "vram_gb": 40,
                            "isolation": "mig",
                            "usd_per_hour": 1,
                        }
                    ]
                ),
                [],
            )
        with self.assertRaisesRegex(pool.GateError, "CHECKPOINT_INTERVAL_ABOVE_4H"):
            self.runner(self.node(checkpoint_interval_s=14401), [])

    def test_missing_ledger_refuses(self):
        self.ledger.unlink()
        with self.assertRaisesRegex(pool.GateError, "SPEND_LEDGER_MISSING"):
            self.runner(self.node(), [])

    def test_child_gate_refusal_is_reported(self):
        summary = self.runner(self.node(), [self.entry("refused", [1.0], exit=3)]).run()
        self.assertEqual(summary["finished"][0]["reason"], "child_gate_refused")


class LogFormatAndSecrets(PoolHarness):
    def test_swap_line_format(self):
        node = self.node(max_concurrent=1)
        entries = [
            self.entry("f1", [1.0, 1.0, 1.0, 1.0], forever=True),
            self.entry("f2", [1.0]),
        ]
        self.runner(node, entries).run()
        swap = self.swaps()[0]
        for key in (
            "schema",
            "event",
            "model",
            "run_id",
            "at_et",
            "gpu_hours",
            "reason",
            "checkpoint_uri",
            "next_run_id",
        ):
            self.assertIn(key, swap)
        self.assertEqual(swap["schema"], "szl.gpu-pool.swap/v1")
        self.assertEqual(swap["model"], "model-f1")
        stamp = datetime.fromisoformat(swap["at_et"])
        self.assertIn(stamp.utcoffset(), (timedelta(hours=-5), timedelta(hours=-4)))
        self.assertIsInstance(swap["gpu_hours"], float)
        self.assertGreater(swap["gpu_hours"], 0)
        file_lines = (self.work / "swaps.jsonl").read_text().splitlines()
        self.assertEqual(len(file_lines), len(self.log.getvalue().splitlines()))
        for line in file_lines:
            self.assertEqual(line.count("\n"), 0)
            json.loads(line)

    def test_children_and_receipts_never_see_r2_secrets(self):
        self.runner(
            self.node(), [self.entry("sec", [1.0, 1.0, 1.0, 1.0], forever=True)]
        ).run()
        env = json.loads((self.work / "runs/sec/env_keys.json").read_text())
        self.assertFalse([k for k in env if k.startswith(("R2_", "AWS_"))])
        self.assertEqual(env.count("CUDA_VISIBLE_DEVICES"), 1)
        for path in self.work.rglob("*"):
            if path.is_file():
                self.assertNotIn(SECRET, path.read_text(errors="ignore"))
        self.assertNotIn(SECRET, self.log.getvalue())

    def test_receipts_are_hash_chained(self):
        self.runner(
            self.node(), [self.entry("c1", [1.0]), self.entry("c2", [1.0])]
        ).run()
        pool_receipt = json.loads((self.work / "receipts/pool.json").read_text())
        self.assertRegex(pool_receipt["prev_receipt_sha256"], "^[0-9a-f]{64}$")


class PlanModeAndCli(PoolHarness):
    def test_plan_mode_launches_nothing(self):
        r = self.runner(
            self.node(),
            [self.entry("p1", [1.0]), self.entry("p2", [1.0], max_hours=60)],
        )
        decisions = r.plan()
        self.assertTrue(decisions[0]["admissible"])
        self.assertEqual(decisions[1]["reason"], "PROJECTED_COST_ABOVE_RUN_CAP")
        self.assertFalse((self.work / "runs/p1").exists())

    def test_cli_defaults_to_plan_and_execute_needs_r2_credentials(self):
        node_path = self.base / "node.json"
        node_path.write_text(json.dumps(self.node()))
        queue_path = self.base / "queue.json"
        queue_path.write_text(
            json.dumps(
                {"schema": pool.QUEUE_SCHEMA, "runs": [self.entry("cli", [1.0])]}
            )
        )
        argv = [
            "--node",
            str(node_path),
            "--queue",
            str(queue_path),
            "--work-dir",
            str(self.work),
            "--repo-root",
            str(self.repo),
        ]
        clean = {k: v for k, v in os.environ.items() if not k.startswith("R2_")}
        out = io.StringIO()
        with (
            mock.patch.dict(
                os.environ,
                dict(clean, SZL_EXECUTION_BRIDGE_REVISION=REVISION),
                clear=True,
            ),
            mock.patch("sys.stdout", out),
        ):
            self.assertEqual(pool.main(argv), 0)
            self.assertEqual(json.loads(out.getvalue())["status"], "PLAN_ONLY")
            out.truncate(0)
            out.seek(0)
            self.assertEqual(pool.main(argv + ["--execute"]), 3)
        self.assertIn("R2_CREDENTIALS_MISSING", out.getvalue())
        self.assertFalse((self.work / "runs/cli").exists())


class R2Store(unittest.TestCase):
    def test_requires_credentials_and_account_endpoint(self):
        with self.assertRaisesRegex(r2.R2Error, "R2_CREDENTIALS_MISSING"):
            r2.R2CheckpointStore("szl-mesh-artifacts", "pool", environ={})
        env = {
            "R2_ACCESS_KEY_ID": "k",
            "R2_SECRET_ACCESS_KEY": SECRET,
            "R2_ENDPOINT": "https://evil.example.com",
        }
        with self.assertRaisesRegex(r2.R2Error, "R2_ENDPOINT_NOT_ACCOUNT_ENDPOINT"):
            r2.R2CheckpointStore("szl-mesh-artifacts", "pool", environ=env)
        with self.assertRaisesRegex(r2.R2Error, "R2_BUCKET_INVALID"):
            r2.R2CheckpointStore("Bad_Bucket", "pool", environ=env)

    def test_upload_reads_back_size_and_sha_and_never_echoes_secrets(self):
        env = {
            "R2_ACCESS_KEY_ID": "k",
            "R2_SECRET_ACCESS_KEY": SECRET,
            "R2_ACCOUNT_ID": "0" * 32,
        }
        store = r2.R2CheckpointStore("szl-mesh-artifacts", "pool", environ=env)
        stored = {}

        class FakeS3:
            def upload_file(self, filename, bucket, key, ExtraArgs):
                stored.update(
                    size=os.path.getsize(filename), meta=ExtraArgs["Metadata"], key=key
                )

            def head_object(self, Bucket, Key):
                return {"ContentLength": stored["size"], "Metadata": stored["meta"]}

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "adapter.bin").write_bytes(b"x" * 10)
            store._client = FakeS3()
            out = store.upload_checkpoint("run-1", "0001-plateau", ckpt, tmp)
            self.assertEqual(
                out["uri"], "r2://szl-mesh-artifacts/pool/run-1/0001-plateau.tar"
            )
            self.assertEqual(out["sha256"], stored["meta"]["sha256"])
            self.assertEqual(list(Path(tmp).glob("*.tar")), [])

            class Broken(FakeS3):
                def upload_file(self, *a, **k):
                    raise RuntimeError(f"denied for secret {SECRET}")

            store._client = Broken()
            with self.assertRaises(r2.R2Error) as caught:
                store.upload_checkpoint("run-1", "0002-final", ckpt, tmp)
            self.assertNotIn(SECRET, str(caught.exception))
            self.assertEqual(str(caught.exception), "R2_UPLOAD_FAILED:RuntimeError")


if __name__ == "__main__":
    unittest.main()

"""Tests for the robotics motion gate (Kamachikuna on physical actuation)."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from szl import robotics  # noqa: E402

SECRET = "s"
ENV = robotics.SafetyEnvelope(
    bounds=([0.0, 0.0, 0.0], [1.0, 1.0, 0.5]),
    max_speed=0.5,
    forbidden_zones=[([0.8, 0.8, 0.0], [1.0, 1.0, 0.5])],
)


class MotionGateTests(unittest.TestCase):
    def setUp(self):
        self.gate = robotics.MotionGate(ENV, secret=SECRET)

    def test_safe_move_allowed_and_receipted(self):
        d = self.gate.command({"action": "move", "target": [0.5, 0.5, 0.2], "speed": 0.3})
        self.assertEqual(d["decision"], "ALLOWED")
        self.assertIsNotNone(d["receipt"])

    def test_estop_always_allowed(self):
        d = self.gate.command({"action": "estop"})
        self.assertEqual(d["decision"], "ALLOWED")
        self.assertIn(1, d["laws"])

    def test_out_of_bounds_blocked(self):
        d = self.gate.command({"action": "move", "target": [2.0, 0.5, 0.2], "speed": 0.3})
        self.assertEqual(d["decision"], "BLOCKED")
        self.assertIn(1, d["laws"])

    def test_forbidden_zone_blocked(self):
        d = self.gate.command({"action": "move", "target": [0.9, 0.9, 0.2], "speed": 0.3})
        self.assertEqual(d["decision"], "BLOCKED")
        self.assertIn("forbidden", d["reason"])

    def test_overspeed_blocked(self):
        d = self.gate.command({"action": "move", "target": [0.5, 0.5, 0.2], "speed": 9.0})
        self.assertEqual(d["decision"], "BLOCKED")

    def test_human_only_action_blocked(self):
        d = self.gate.command({"action": "brake.release"})
        self.assertEqual(d["decision"], "BLOCKED")
        self.assertIn(7, d["laws"])

    def test_malformed_commands_fail_closed(self):
        for cmd in ({}, {"action": "move", "target": [0.1, 0.1], "speed": 0.1},
                    {"action": "move", "target": ["x", 0.1, 0.1], "speed": 0.1},
                    {"action": "move", "target": [0.1, 0.1, 0.1], "speed": -1}):
            with self.subTest(cmd=cmd):
                self.assertEqual(self.gate.command(cmd)["decision"], "BLOCKED")

    def test_command_log_chain_verifies_and_detects_tamper(self):
        gate = robotics.MotionGate(ENV, secret=SECRET)
        rs = [gate.command({"action": "move", "target": [0.1 * i, 0.1, 0.1],
                            "speed": 0.1})["receipt"] for i in range(1, 4)]
        self.assertTrue(gate.verify_log(rs))
        rs[1]["claim"]["decision"] = "BLOCKED"  # forge a decision post-hoc
        self.assertFalse(gate.verify_log(rs))

    def test_no_secret_means_no_log_verification(self):
        gate = robotics.MotionGate(ENV, secret=None)
        d = gate.command({"action": "move", "target": [0.1, 0.1, 0.1], "speed": 0.1})
        self.assertEqual(d["decision"], "ALLOWED")
        self.assertIsNone(d["receipt"])
        self.assertFalse(gate.verify_log([]))

    def test_bad_envelope_rejected(self):
        with self.assertRaises(ValueError):
            robotics.SafetyEnvelope(bounds=([1.0], [0.0]), max_speed=1.0)
        with self.assertRaises(ValueError):
            robotics.SafetyEnvelope(bounds=([0.0], [1.0]), max_speed=0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Exercise admission boundaries without allocating a GPU or importing training."""

import ast
from pathlib import Path
import types
import unittest
from unittest.mock import Mock

WORKER = Path(__file__).resolve().parents[1] / "tools" / "salad" / "salad_worker.py"
MIB = 1024**2


class SharedGpuBudget(unittest.TestCase):
    def setUp(self):
        self.cuda = types.SimpleNamespace(
            is_available=Mock(return_value=True),
            device_count=Mock(return_value=1),
            mem_get_info=Mock(return_value=(38_000 * MIB, 49_152 * MIB)),
            set_per_process_memory_fraction=Mock(),
        )
        source = ast.parse(WORKER.read_text(encoding="utf-8"))
        function = next(
            node
            for node in source.body
            if isinstance(node, ast.FunctionDef)
            and node.name == "configure_shared_gpu_budget"
        )
        self.receipt = Mock()
        namespace = {
            "torch": types.SimpleNamespace(cuda=self.cuda),
            "receipt": self.receipt,
        }
        exec(
            compile(ast.Module(body=[function], type_ignores=[]), str(WORKER), "exec"),
            namespace,
        )
        self.admit = namespace["configure_shared_gpu_budget"]

    def test_sets_allocator_cap_after_headroom_check(self):
        evidence = self.admit(24576, 8192)
        self.cuda.set_per_process_memory_fraction.assert_called_once_with(0.5, device=0)
        self.assertEqual(evidence["free_before_mib"], 38_000)
        self.assertEqual(evidence["allocator_budget_mib"], 24576)
        self.assertIn("NOT_HARDWARE_PARTITION", evidence["scope"])

    def test_exact_headroom_boundary_is_admitted(self):
        self.cuda.mem_get_info.return_value = (32768 * MIB, 49152 * MIB)
        self.admit(24576, 8192)
        self.cuda.set_per_process_memory_fraction.assert_called_once()

    def test_one_byte_below_reserve_refuses_before_setting_allocator(self):
        self.cuda.mem_get_info.return_value = (32768 * MIB - 1, 49152 * MIB)
        with self.assertRaisesRegex(RuntimeError, "insufficient"):
            self.admit(24576, 8192)
        self.cuda.set_per_process_memory_fraction.assert_not_called()
        self.receipt.assert_not_called()

    def test_zero_or_multiple_visible_gpus_are_not_admitted(self):
        for count in (0, 2):
            with self.subTest(count=count):
                self.cuda.device_count.return_value = count
                with self.assertRaisesRegex(RuntimeError, "exactly one"):
                    self.admit(24576, 8192)
        self.cuda.mem_get_info.assert_not_called()

    def test_invalid_limits_refuse_before_gpu_queries(self):
        for budget, reserve in (
            (0, 8192),
            (-1, 8192),
            (24576, -1),
            (True, 0),
            (24576, 0.5),
        ):
            with self.subTest(budget=budget, reserve=reserve):
                with self.assertRaises(ValueError):
                    self.admit(budget, reserve)
        self.cuda.is_available.assert_not_called()

    def test_driver_and_allocator_failures_never_emit_admission_receipt(self):
        self.cuda.set_per_process_memory_fraction.side_effect = RuntimeError(
            "driver lost"
        )
        with self.assertRaisesRegex(RuntimeError, "driver lost"):
            self.admit(24576, 8192)
        self.receipt.assert_not_called()

    def test_cuda_unavailable_does_not_silently_use_cpu(self):
        self.cuda.is_available.return_value = False
        with self.assertRaises(RuntimeError):
            self.admit(24576, 8192)
        self.cuda.mem_get_info.assert_not_called()


if __name__ == "__main__":
    unittest.main()

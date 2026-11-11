"""Contract tests for tools/lean_axiom_verdict.py and the proving-ground Lean stage."""

import importlib.util
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "lean_axiom_verdict", ROOT / "tools" / "lean_axiom_verdict.py"
)
lav = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lav)
N = "Lutar.Lambda.lambda_two_axis_schur_concave"
CORE = "'%s' depends on axioms: [propext, Classical.choice, Quot.sound]" % N


class VerdictTests(unittest.TestCase):
    def state(self, build_rc, audit_rc, out, name=N):
        return lav.verdict(build_rc, audit_rc, out, name)["state"]

    def test_core_axioms_only_is_proven(self):
        self.assertEqual(self.state(0, 0, CORE), "PROVEN")

    def test_no_axioms_is_proven(self):
        self.assertEqual(
            self.state(0, 0, "'%s' does not depend on any axioms" % N), "PROVEN"
        )

    def test_wrapped_axiom_list_is_parsed(self):
        out = "'%s' depends on axioms: [propext,\n Classical.choice,\n Quot.sound]" % N
        self.assertEqual(self.state(0, 0, out), "PROVEN")

    def test_sorry_is_never_proven(self):
        out = (
            "'%s' depends on axioms: [propext, sorryAx, Classical.choice, Quot.sound]"
            % N
        )
        self.assertEqual(self.state(0, 0, out), "UNPROVEN")

    def test_added_axiom_is_never_proven(self):
        out = (
            "'%s' depends on axioms: [propext, Classical.choice, Quot.sound, Lutar.Lambda.cheat]"
            % N
        )
        self.assertEqual(self.state(0, 0, out), "AXIOM_DEPENDENT")

    def test_axiom_declaration_is_conjecture(self):
        n = "Lutar.Lambda.lambda_schur_concave_n_axis"
        out = "'%s' depends on axioms: [propext, Classical.choice, Quot.sound, %s]" % (
            n,
            n,
        )
        self.assertEqual(self.state(0, 0, out, n), "CONJECTURE")

    def test_failed_build_withholds_verdict(self):
        self.assertEqual(self.state(1, 0, CORE), "UNAVAILABLE")

    def test_failed_audit_withholds_verdict(self):
        self.assertEqual(self.state(0, 1, CORE), "UNAVAILABLE")

    def test_missing_line_withholds_verdict(self):
        self.assertEqual(self.state(0, 0, ""), "UNAVAILABLE")

    def test_other_declaration_does_not_count(self):
        out = "'%s_renamed' depends on axioms: [propext]" % N
        self.assertEqual(self.state(0, 0, out), "UNAVAILABLE")

    def test_selftest_passes(self):
        self.assertEqual(lav._selftest(), 0)


class ProvingGroundScriptTests(unittest.TestCase):
    text = (ROOT / "tools" / "aeroff_proving_ground.sh").read_text(encoding="utf-8")

    def test_verdicts_do_not_come_from_grep(self):
        self.assertNotIn('grep -q "theorem ', self.text)
        self.assertNotIn('grep -q "axiom ', self.text)
        self.assertIn("#print axioms", self.text)
        self.assertIn("lean_axiom_verdict.py", self.text)

    def test_no_fallback_test_runner(self):
        self.assertNotIn("make test 2>&1 || pytest", self.text)

    def test_rayleigh_ratio_has_no_square_root_of_viktorov_form(self):
        self.assertNotIn("math.sqrt((0.87 + 1.12*nu)/(1 + nu))", self.text)

    def test_hourly_probe_is_not_labelled_measured(self):
        self.assertNotIn('STATE="MEASURED commits=', self.text)


if __name__ == "__main__":
    unittest.main()

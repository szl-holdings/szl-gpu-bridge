"""Synthetic dependency-pair contract tests, no package/model imports."""

import unittest

from tools.salad.numpy_pyarrow_preflight import (
    MAX_LOCK_BYTES,
    review_lock_contents,
    review_pair,
)


class ArrowNumpyPreflightTests(unittest.TestCase):
    def review(self, np, pa, python="3.12"):
        return review_pair(numpy=np, pyarrow=pa, python=python)

    def test_original_crash_loop_pair_is_denied(self):
        result = self.review("1.26.4", "26.0.1")
        self.assertEqual(result["reason"], "PYARROW26_REQUIRES_NUMPY2")
        self.assertFalse(result["cloud_mutation_authorized"])

    def test_arrow_26_0_0_rejects_numpy_1(self):
        result = self.review("1.26.4", "26.0.0")
        self.assertEqual(result["reason"], "PYARROW26_REQUIRES_NUMPY2")

    def test_owner_proposed_historical_pin_is_source_candidate_only(self):
        result = self.review("1.26.4", "19.0.1")
        self.assertEqual(result["verdict"], "SOURCE_PAIR_FOR_OWNER_REVIEW")
        self.assertFalse(result["runtime_import_verified"])
        self.assertFalse(result["training_complete_verified"])

    def test_v3_numpy2_and_current_image_pair_are_source_candidates(self):
        for numpy in ("2.1.3", "2.5.5"):
            with self.subTest(numpy=numpy):
                result = self.review(numpy, "26.0.1")
                self.assertEqual(result["verdict"], "SOURCE_PAIR_FOR_OWNER_REVIEW")
                self.assertFalse(result["model_qualified"])
                self.assertFalse(result["dependency_install_verified"])

    def test_python_310_cannot_admit_arrow_26(self):
        result = self.review("2.1.3", "26.0.1", python="3.10")
        self.assertEqual(result["reason"], "PYARROW26_PYTHON_TOO_OLD")

    def test_numpy3_or_unrecognized_arrow_rejected(self):
        result = self.review("3.0.0", "26.0.1")
        self.assertEqual(result["reason"], "PYARROW26_REQUIRES_NUMPY2")
        result = self.review("1.26.4", "27.0.0")
        reason = result["reason"]
        self.assertEqual(reason, "PYARROW_VERSION_NOT_IN_QUALIFIED_SOURCE_SET")

    def test_numpy2_not_assumed_compatible_with_older_arrow(self):
        result = self.review("2.1.3", "19.0.1")
        self.assertEqual(result["reason"], "HISTORICAL_NUMPY1_PAIR_UNVERIFIED")

    def test_unpinned_or_malformed_versions_hold(self):
        for numpy in ("numpy>=2", ">=2.0", "2.1", None, True):
            with self.subTest(numpy=numpy):
                result = self.review(numpy, "26.0.1")
                self.assertEqual(result["reason"], "NUMPY_EXACT_PIN_MISSING")
        self.assertEqual(
            self.review("2.1.3", "pyarrow>=26")["reason"],
            "PYARROW_EXACT_PIN_MISSING",
        )

    def test_unknown_python_target_holds(self):
        self.assertEqual(
            self.review("2.1.3", "26.0.1", python="local")["reason"],
            "PYTHON_RUNTIME_UNVERIFIED",
        )

    def test_hash_lock_format_and_actual_versions_are_supported(self):
        sample = (
            "numpy==2.5.5 " + chr(92) + "\n"
            "  --hash=sha256:" + "a" * 64 + "\n"
            "pyarrow==26.0.1 " + chr(92) + "\n"
            "  --hash=sha256:" + "b" * 64 + "\n"
        )
        result = review_lock_contents(sample, python="3.12")
        self.assertEqual(result["verdict"], "SOURCE_PAIR_FOR_OWNER_REVIEW")

    def test_hash_lock_rejects_broken_original_recipe(self):
        sample = "numpy==1.26.4\npyarrow==26.0.1\n"
        result = review_lock_contents(sample, python="3.12")
        self.assertEqual(result["reason"], "PYARROW26_REQUIRES_NUMPY2")

    def test_transitive_unpinned_pyarrow_is_denied(self):
        sample = "numpy==1.26.4\ndatasets==3.5.1\n"
        self.assertEqual(
            review_lock_contents(sample, python="3.12")["reason"],
            "LOCK_NUMPY_PYARROW_PIN_MISSING",
        )
        sample = "numpy==1.26.4\npyarrow>=15\n"
        self.assertEqual(
            review_lock_contents(sample, python="3.12")["reason"],
            "LOCK_EXACT_PIN_UNVERIFIED",
        )

    def test_duplicate_or_repeated_pin_denied(self):
        sample = "numpy==2.1.3\nnumpy==2.1.3\npyarrow==26.0.1\n"
        self.assertEqual(
            review_lock_contents(sample, python="3.12")["reason"],
            "LOCK_DUPLICATE_PIN",
        )

    def test_locked_input_is_bounded_and_typed(self):
        self.assertEqual(
            review_lock_contents("x" * (MAX_LOCK_BYTES + 1), python="3.12")["reason"],
            "LOCK_BYTES_INVALID",
        )
        self.assertEqual(
            review_lock_contents(None, python="3.12")["reason"],
            "LOCK_BYTES_INVALID",
        )

    def test_output_never_authorizes_training_or_exposes_recipe(self):
        for result in (
            self.review("1.26.4", "26.0.1"),
            self.review("2.1.3", "26.0.1"),
        ):
            self.assertFalse(result["cloud_mutation_authorized"])
            self.assertFalse(result["training_complete_verified"])
            self.assertNotIn("HF_TOKEN", str(result))
            self.assertNotIn("Salad-Api-Key", str(result))


if __name__ == "__main__":
    unittest.main()

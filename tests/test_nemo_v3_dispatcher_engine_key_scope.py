from __future__ import annotations

import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "laptop"))

from dispatcher import validate_payload_key_scope  # noqa: E402
from frontier_contract import (  # noqa: E402
    ContractError,
    V1_PAYLOAD_TYPE,
    V2_PAYLOAD_TYPE,
)
from nemo_v3_contract import (  # noqa: E402
    LEGACY_ENGINE_KEY_ID,
    NEMO_V3_PAYLOAD_TYPE,
)


class DispatcherEngineKeyScopeTests(unittest.TestCase):
    def test_legacy_key_remains_valid_for_legacy_payload_types(self) -> None:
        for payload_type in (V1_PAYLOAD_TYPE, V2_PAYLOAD_TYPE):
            validate_payload_key_scope(payload_type, LEGACY_ENGINE_KEY_ID)

    def test_recovery_key_is_refused_for_legacy_payload_types(self) -> None:
        for payload_type in (V1_PAYLOAD_TYPE, V2_PAYLOAD_TYPE):
            with self.assertRaisesRegex(ContractError, "cannot authorize"):
                validate_payload_key_scope(payload_type, "815714c8d4ae3e4d")

    def test_recovery_key_can_reach_nemo_authorization_checks(self) -> None:
        validate_payload_key_scope(NEMO_V3_PAYLOAD_TYPE, "815714c8d4ae3e4d")


if __name__ == "__main__":
    unittest.main(verbosity=2)

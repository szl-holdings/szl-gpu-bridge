from __future__ import annotations

import base64
import json
import pathlib
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]

sys.path.insert(0, str(ROOT / "laptop"))

from frontier_contract import (  # noqa: E402
    ContractError,
    derive_key_id,
    load_engine_pin_for_envelope,
)


class EngineKeyringTests(unittest.TestCase):
    def setUp(self) -> None:
        try:
            from nacl.signing import SigningKey
        except ImportError as exc:  # pragma: no cover - CI installs PyNaCl
            self.skipTest(f"PyNaCl unavailable: {exc}")
        self.tmp = tempfile.TemporaryDirectory()
        self.keys = pathlib.Path(self.tmp.name)
        self.pins: dict[str, dict[str, str]] = {}
        for suffix in ("old", "new"):
            signing_key = SigningKey.generate()
            spki = b"\x30\x2a\x30\x05\x06\x03\x2b\x65\x70\x03\x21\x00" + bytes(
                signing_key.verify_key
            )
            key_id = derive_key_id(spki)
            self.pins[suffix] = {
                "keyId": key_id,
                "publicKeySpkiBase64": base64.b64encode(spki).decode(),
            }
            (self.keys / f"engine_pubkey_{key_id}.json").write_text(
                json.dumps(self.pins[suffix]),
                encoding="utf-8",
            )
        (self.keys / "engine_keyring.json").write_text(
            json.dumps(
                {
                    "kind": "szl-quant-engine-keyring",
                    "v": 1,
                    "keys": {
                        self.pins["old"]["keyId"]: {
                            "file": (f"engine_pubkey_{self.pins['old']['keyId']}.json"),
                            "status": "VERIFY_ONLY",
                        },
                        self.pins["new"]["keyId"]: {
                            "file": (f"engine_pubkey_{self.pins['new']['keyId']}.json"),
                            "status": "ACTIVE",
                        },
                    },
                }
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    @staticmethod
    def envelope(key_id: str) -> dict[str, object]:
        return {"signatures": [{"keyid": key_id, "sig": "unused"}]}

    def test_active_and_verify_only_keys_are_both_verifiable(self) -> None:
        for pin in self.pins.values():
            selected = load_engine_pin_for_envelope(
                self.keys,
                self.envelope(pin["keyId"]),
            )
            self.assertEqual(selected, pin)

    def test_verify_only_key_cannot_authorize_new_execution(self) -> None:
        with self.assertRaisesRegex(ContractError, "verification-only"):
            load_engine_pin_for_envelope(
                self.keys,
                self.envelope(self.pins["old"]["keyId"]),
                require_active=True,
            )
        selected = load_engine_pin_for_envelope(
            self.keys,
            self.envelope(self.pins["new"]["keyId"]),
            require_active=True,
        )
        self.assertEqual(selected, self.pins["new"])

    def test_unenrolled_key_fails_closed(self) -> None:
        with self.assertRaisesRegex(ContractError, "not enrolled"):
            load_engine_pin_for_envelope(
                self.keys,
                self.envelope("0" * 16),
            )

    def test_unsafe_pin_filename_fails_closed(self) -> None:
        keyring_path = self.keys / "engine_keyring.json"
        keyring = json.loads(keyring_path.read_text(encoding="utf-8"))
        keyring["keys"][self.pins["new"]["keyId"]]["file"] = "../private.pem"
        keyring_path.write_text(json.dumps(keyring), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "unsafe pin file"):
            load_engine_pin_for_envelope(
                self.keys,
                self.envelope(self.pins["new"]["keyId"]),
            )

    def test_mislabeled_public_bytes_fail_closed(self) -> None:
        path = self.keys / f"engine_pubkey_{self.pins['new']['keyId']}.json"
        path.write_text(json.dumps(self.pins["old"]), encoding="utf-8")
        with self.assertRaisesRegex(ContractError, "differs from its pin"):
            load_engine_pin_for_envelope(
                self.keys,
                self.envelope(self.pins["new"]["keyId"]),
            )

    def test_repository_keyring_has_one_exact_active_administrative_key(self) -> None:
        keyring = json.loads(
            (ROOT / "keys" / "engine_keyring.json").read_text(encoding="utf-8")
        )
        self.assertEqual(
            {key_id: entry["status"] for key_id, entry in keyring["keys"].items()},
            {
                "5c6cf59741ade920": "VERIFY_ONLY",
                "815714c8d4ae3e4d": "VERIFY_ONLY",
                "b8041281c81c4caa": "ACTIVE",
            },
        )
        active = [
            key_id
            for key_id, entry in keyring["keys"].items()
            if entry["status"] == "ACTIVE"
        ]
        self.assertEqual(active, ["b8041281c81c4caa"])
        selected = load_engine_pin_for_envelope(
            ROOT / "keys",
            self.envelope("b8041281c81c4caa"),
            require_active=True,
        )
        spki = base64.b64decode(selected["publicKeySpkiBase64"], validate=True)
        self.assertEqual(derive_key_id(spki), "b8041281c81c4caa")
        self.assertIn(
            "No cryptographic continuity",
            str(selected.get("note")),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

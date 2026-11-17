# SPDX-License-Identifier: Apache-2.0
"""Synthetic negative controls; never claim these are estate observations."""

import hashlib
import json

import pytest

from tools.salad import honesty_energy_frontier as frontier


def inputs():
    identity = {
        "model": "SZLHOLDINGS/synthetic-do-not-publish",
        "model_revision": "a" * 40,
        "run_id": "synthetic_eval_001",
        "fixture_sha256": "b" * 64,
        "gpu_uuid": "GPU-00000000-0000-0000-0000-000000000000",
    }
    battery = {
        "schema": frontier.BATTERY_SCHEMA,
        **identity,
        "receipt_sha256": "c" * 64,
        "state": "MEASURED",
        "score": "3/6",
        "rows": [
            {
                "id": name,
                "kind": "answerable" if name.startswith("a") else "unanswerable",
                "abstained": False,
                "verdict": "PASS" if name.startswith("a") else "FAIL",
            }
            for name in frontier.IDS
        ],
    }
    energy = {
        "schema": frontier.ENERGY_SCHEMA,
        **identity,
        "receipt_sha256": "d" * 64,
        "measurement_scope": "EVALUATION_SIX_PROBES",
        "method": "NVML_DEVICE_TOTAL_MJ",
        "counter_unit": "mJ",
        "rows": [
            {
                "id": name,
                "counter_start_mj": i * 1_000,
                "counter_end_mj": (i + 1) * 1_000,
            }
            for i, name in enumerate(frontier.IDS)
        ],
    }
    return battery, energy


def assert_hold(battery, energy, reason):
    report = frontier.assess(battery, energy)
    assert report["status"] == "HOLD"
    assert report["reason_code"] == reason
    assert report["metric"] is None
    assert report["source_identity"] is None
    assert report["reported_receipts"] is None
    assert report["signature_verified"] is False
    assert report["production_authorization"] is False
    assert report["ranking_authorized"] is False


def test_same_run_synthetic_diagnostic_reports_zero_honesty_not_energy_claim():
    battery, energy = inputs()
    result = frontier.assess(battery, energy)
    assert result["status"] == "DERIVED_UNATTESTED"
    metric = result["metric"]
    assert metric["correct_answerable"] == 3
    assert metric["correct_abstentions"] == 0
    assert metric["measured_device_joules"] == 6.0
    assert metric["device_joules_per_valid_outcome"] == 2.0
    assert metric["joint_quality_geomean"] == 0
    assert metric["joint_quality_per_kilojoule"] == 0
    assert len(metric["per_probe_device_joules"]) == 6
    assert result["source_identity"]["run_id"] == "synthetic_eval_001"
    assert (
        result["reported_receipts"]["verification"] == "UNVERIFIED_REPORTED_IDENTIFIERS"
    )
    assert result["signature_verified"] is False
    assert result["ranking_authorized"] is False
    assert result["production_authorization"] is False
    assert result["promotion_authorized"] is False
    assert result["receipt_integrity"] == "REPORTED_DIGEST_UNVERIFIED"
    digest = result.pop("report_sha256")
    assert digest == hashlib.sha256(frontier._canonical(result)).hexdigest()


def test_correct_abstentions_make_joint_quality_one_without_certifying_model():
    battery, energy = inputs()
    for row in battery["rows"][3:]:
        row["abstained"] = True
        row["verdict"] = "PASS"
    battery["score"] = "6/6"
    report = frontier.assess(battery, energy)
    assert report["metric"]["joint_quality_geomean"] == 1.0
    assert report["metric"]["device_joules_per_valid_outcome"] == 1.0
    assert report["metric"]["joint_quality_per_kilojoule"] == pytest.approx(
        1000 / 6, abs=1e-6
    )
    assert report["ranking_authorized"] is False


@pytest.mark.parametrize(
    "key,code",
    [
        ("model", "MODEL_ID_MISMATCH"),
        ("model_revision", "MODEL_REVISION_MISMATCH"),
        ("run_id", "RUN_ID_MISMATCH"),
        ("fixture_sha256", "FIXTURE_DIGEST_MISMATCH"),
        ("gpu_uuid", "GPU_UUID_MISMATCH"),
    ],
)
def test_independent_identity_axes_cannot_cross_join(key, code):
    battery, energy = inputs()
    energy[key] = (
        "z" * 40
        if key == "model_revision"
        else (
            "d" * 64
            if key == "fixture_sha256"
            else "GPU-11111111-2222-3333-4444-555555555555"
            if key == "gpu_uuid"
            else "SZLHOLDINGS/other-model"
            if key == "model"
            else "unrelated_run"
        )
    )
    if key == "model_revision":
        energy[key] = "1" * 40
    assert_hold(battery, energy, code)


@pytest.mark.parametrize(
    "change,code",
    [
        ("wrong_score", "SCORE_CONTRADICTS_ROWS"),
        ("wrong_abstention", "VERDICT_ABSTENTION_CONFLICT"),
        ("bool_as_abstention", "ABSTENTION_FLAG_INVALID"),
        ("duplicate_probe", "PROBE_ID_INVALID"),
        ("missing_probe", "PROBE_COVERAGE_INVALID"),
        ("unknown_probe", "PROBE_ID_INVALID"),
        ("wrong_kind", "PROBE_KIND_INVALID"),
        ("not_measured", "BATTERY_NOT_MEASURED"),
    ],
)
def test_battery_negative_controls(change, code):
    battery, energy = inputs()
    if change == "wrong_score":
        battery["score"] = "6/6"
    elif change == "wrong_abstention":
        battery["rows"][0]["abstained"] = True
    elif change == "bool_as_abstention":
        battery["rows"][0]["abstained"] = 1
    elif change == "duplicate_probe":
        battery["rows"][1]["id"] = "a1"
    elif change == "missing_probe":
        battery["rows"].pop()
    elif change == "unknown_probe":
        battery["rows"][1]["id"] = "x2"
    elif change == "wrong_kind":
        battery["rows"][1]["kind"] = "unanswerable"
    else:
        battery["state"] = "UNMEASURED"
    assert_hold(battery, energy, code)


@pytest.mark.parametrize(
    "change,code",
    [
        ("training_scope", "ENERGY_SCOPE_MISMATCH"),
        ("not_nvml", "ENERGY_METHOD_UNSUPPORTED"),
        ("wrong_unit", "ENERGY_UNIT_INVALID"),
        ("missing_probe", "ENERGY_COVERAGE_INVALID"),
        ("duplicate_probe", "ENERGY_PROBE_ID_INVALID"),
        ("counter_rollback", "COUNTER_RANGE_INVALID"),
        ("counter_reset", "COUNTER_NONMONOTONE"),
        ("zero_delta", "COUNTER_RANGE_INVALID"),
        ("bool_counter", "COUNTER_RANGE_INVALID"),
        ("wrong_order", "ENERGY_ORDER_INVALID"),
    ],
)
def test_energy_negative_controls(change, code):
    battery, energy = inputs()
    if change == "training_scope":
        energy["measurement_scope"] = "TRAINING"
    elif change == "not_nvml":
        energy["method"] = "ASSUMED_WATTS"
    elif change == "wrong_unit":
        energy["counter_unit"] = "J"
    elif change == "missing_probe":
        energy["rows"].pop()
    elif change == "duplicate_probe":
        energy["rows"][1]["id"] = "a1"
    elif change == "counter_rollback":
        energy["rows"][0]["counter_end_mj"] = -5
    elif change == "counter_reset":
        energy["rows"][1]["counter_start_mj"] = 0
    elif change == "zero_delta":
        energy["rows"][0]["counter_end_mj"] = 0
    elif change == "bool_counter":
        energy["rows"][0]["counter_start_mj"] = True
    elif change == "wrong_order":
        energy["rows"][0]["id"], energy["rows"][1]["id"] = (
            energy["rows"][1]["id"],
            energy["rows"][0]["id"],
        )
    assert_hold(battery, energy, code)


def test_legacy_energy_and_other_model_battery_cannot_be_laundered():
    battery, energy = inputs()
    battery.pop("schema")
    battery["kind"] = "szl.estate-battery/base-baseline"
    energy["schema"] = "szl.gpu-mesh-benchmark/v1"
    assert_hold(battery, energy, "SCHEMA_INVALID")


@pytest.mark.parametrize(
    "field,code",
    [
        ("model_revision", "MODEL_REVISION_UNAVAILABLE"),
        ("run_id", "RUN_ID_UNAVAILABLE"),
        ("fixture_sha256", "FIXTURE_DIGEST_UNAVAILABLE"),
        ("gpu_uuid", "GPU_UUID_UNAVAILABLE"),
        ("receipt_sha256", "RECEIPT_DIGEST_UNAVAILABLE"),
    ],
)
def test_missing_verifiable_pins_never_become_proof(field, code):
    battery, energy = inputs()
    battery.pop(field)
    assert_hold(battery, energy, code)


def test_claimed_signature_and_approval_never_transfer_from_input():
    battery, energy = inputs()
    battery["signature_verified"] = True
    battery["production_authorization"] = True
    energy["ranking_authorized"] = True
    report = frontier.assess(battery, energy)
    assert report["status"] == "DERIVED_UNATTESTED"
    assert report["signature_verified"] is False
    assert report["production_authorization"] is False
    assert report["ranking_authorized"] is False
    assert "signature_verified" not in report["source_identity"]


def test_input_json_duplicate_keys_fail_closed(tmp_path):
    path = tmp_path / "duplicate.json"
    path.write_text('{"model":"a","model":"b"}', encoding="utf-8")
    with pytest.raises(frontier.Hold, match="DUPLICATE_JSON_KEY"):
        frontier.read_input(path)


@pytest.mark.parametrize("text", ['{"x":NaN}', '{"x":Infinity}', '{"x":1e999}'])
def test_nonfinite_json_rejected(tmp_path, text):
    path = tmp_path / "bad.json"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(frontier.Hold, match="NONFINITE_JSON"):
        frontier.read_input(path)


def test_oversized_and_missing_file_rejected(tmp_path):
    huge = tmp_path / "huge.json"
    huge.write_bytes(b" " * (frontier.MAX_BYTES + 1))
    with pytest.raises(frontier.Hold, match="INPUT_BYTE_LIMIT"):
        frontier.read_input(huge)
    with pytest.raises(frontier.Hold, match="INPUT_UNAVAILABLE_OR_INVALID"):
        frontier.read_input(tmp_path / "missing.json")


def test_symlink_does_not_escape_bounded_read(tmp_path):
    target = tmp_path / "target.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("host lacks symlink creation privilege")
    with pytest.raises(frontier.Hold, match="INPUT_NOT_REGULAR"):
        frontier.read_input(link)


def test_cli_returns_bounded_hold_without_printing_private_input(tmp_path, capsys):
    battery, energy = inputs()
    battery["run_id"] = "sensitive-local-opaque-token"
    p1 = tmp_path / "battery.json"
    p2 = tmp_path / "energy.json"
    p1.write_text(json.dumps(battery), encoding="utf-8")
    p2.write_text(json.dumps(energy), encoding="utf-8")
    status = frontier.main(["--battery", str(p1), "--energy", str(p2)])
    result = json.loads(capsys.readouterr().out)
    assert status == 3
    assert result["status"] == "HOLD"
    assert "sensitive-local-opaque-token" not in json.dumps(result)


def test_cli_derived_report_is_read_only_and_deterministic(tmp_path, capsys):
    battery, energy = inputs()
    p1, p2 = tmp_path / "battery.json", tmp_path / "energy.json"
    p1.write_text(json.dumps(battery), encoding="utf-8")
    p2.write_text(json.dumps(energy), encoding="utf-8")
    assert frontier.main(["--battery", str(p1), "--energy", str(p2)]) == 0
    one = capsys.readouterr().out
    assert frontier.main(["--battery", str(p1), "--energy", str(p2)]) == 0
    two = capsys.readouterr().out
    assert one == two
    assert json.loads(one)["external_writes"] is False

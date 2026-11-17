"""Adversarial fixtures for a read-only finite evidence-obligation advisor."""

import json
import subprocess
import sys

import pytest

from tools.evidence import obligation_planner as advice


def request(*, authority="ALLOW", budget=15, obligations=None, observations=None):
    return {
        "schema": advice.REQUEST_SCHEMA,
        "operation_id": "fixture:promotion-review",
        "authority_state": authority,
        "budget_units": budget,
        "obligations": obligations
        if obligations is not None
        else [
            {"id": label, "state": "OPEN", "evidence_digest": None}
            for label in ("source", "device", "window")
        ],
        "observations": observations
        if observations is not None
        else [
            {
                "id": "inspect-source",
                "mode": "READ_ONLY",
                "admission_state": "ALLOW",
                "cost_units": 2,
                "may_inform": ["source"],
            },
            {
                "id": "inspect-device-window",
                "mode": "READ_ONLY",
                "admission_state": "ALLOW",
                "cost_units": 6,
                "may_inform": ["device", "window"],
            },
            {
                "id": "inspect-all",
                "mode": "READ_ONLY",
                "admission_state": "ALLOW",
                "cost_units": 11,
                "may_inform": ["source", "device", "window"],
            },
        ],
    }


def test_exact_low_cost_selection_with_noncompensating_status():
    result = advice.advise(request())
    assert result["state"] == "ADVISORY_CANDIDATE"
    assert result["selected_read_observations"] == [
        "inspect-device-window",
        "inspect-source",
    ]
    assert result["proposed_cost_units"] == 8
    assert result["authority_granted"] is False
    assert result["requirements_satisfied_by_advice"] == 0
    assert result["quality_or_promotion_qualified"] is False
    assert result["remote_effects_performed"] == 0


@pytest.mark.parametrize(
    "state,expected",
    [
        ("DENY", "BLOCKED_POLICY"),
        ("UNKNOWN", "HOLD_AUTHORITY"),
    ],
)
def test_denied_or_unknown_operation_never_gets_a_plan(state, expected):
    result = advice.advise(request(authority=state))
    assert result["state"] == expected
    assert result["selected_read_observations"] == []
    assert result["authority_granted"] is False


def test_stale_obligation_is_not_silently_satisfied():
    spec = request()
    spec["obligations"][0]["state"] = "STALE"
    result = advice.advise(spec)
    assert "source" in result["evidence_obligations_open"]
    assert result["requirements_satisfied_by_advice"] == 0


def test_caller_declared_observed_is_not_release_authorization():
    result = advice.advise(
        request(
            obligations=[
                {"id": "source", "state": "OBSERVED", "evidence_digest": "a" * 64}
            ],
            observations=[],
        )
    )
    assert result["state"] == "NO_MISSING_INPUT_OBLIGATIONS"
    assert result["authority_granted"] is False
    assert result["quality_or_promotion_qualified"] is False
    assert not result["selected_read_observations"]


def test_untrusted_input_claim_of_observation_needs_a_content_digest():
    spec = request()
    spec["obligations"][0]["state"] = "OBSERVED"
    with pytest.raises(advice.EvidenceHold, match="OBSERVED_DIGEST_UNAVAILABLE"):
        advice.advise(spec)


@pytest.mark.parametrize(
    "change",
    [
        {"mode": "WRITE", "admission_state": "ALLOW"},
        {"mode": "READ_ONLY", "admission_state": "DENY"},
        {"mode": "READ_ONLY", "admission_state": "UNKNOWN"},
    ],
)
def test_unadmitted_reads_or_any_write_are_excluded(change):
    spec = request(
        obligations=[{"id": "source", "state": "OPEN", "evidence_digest": None}],
        observations=[
            {
                "id": "candidate",
                "mode": change["mode"],
                "admission_state": change["admission_state"],
                "cost_units": 0,
                "may_inform": ["source"],
            }
        ],
    )
    result = advice.advise(spec)
    assert result["state"] == "HOLD_NO_READ_COVERAGE"
    assert "candidate" in result["excluded_observations"]
    assert result["selected_read_observations"] == []


def test_zero_cost_is_valid_but_not_measured_evidence():
    spec = request(
        budget=0,
        obligations=[{"id": "source", "state": "OPEN", "evidence_digest": None}],
        observations=[
            {
                "id": "free-local-read",
                "mode": "READ_ONLY",
                "admission_state": "ALLOW",
                "cost_units": 0,
                "may_inform": ["source"],
            }
        ],
    )
    value = advice.advise(spec)
    assert value["proposed_cost_units"] == 0
    assert value["state"] == "ADVISORY_CANDIDATE"
    assert value["signature"] == "UNSIGNED"


def test_insufficient_budget_never_uses_a_denied_cheaper_write():
    spec = request(budget=5)
    spec["observations"].append(
        {
            "id": "unsafe-skip",
            "mode": "WRITE",
            "admission_state": "ALLOW",
            "cost_units": 0,
            "may_inform": ["device", "window", "source"],
        }
    )
    result = advice.advise(spec)
    assert result["state"] == "HOLD_UNCOVERED_OR_OVER_BUDGET"
    assert result["selected_read_observations"] == []
    assert (
        result["excluded_observations"]["unsafe-skip"] == "NOT_A_READ_ONLY_OBSERVATION"
    )


def test_equal_cost_prefers_fewer_reads_then_lexical_ids():
    spec = request(
        obligations=[{"id": "source", "state": "OPEN", "evidence_digest": None}],
        observations=[
            {
                "id": name,
                "mode": "READ_ONLY",
                "admission_state": "ALLOW",
                "cost_units": 3,
                "may_inform": ["source"],
            }
            for name in ("z-last", "a-first")
        ],
    )
    assert advice.advise(spec)["selected_read_observations"] == ["a-first"]


def test_search_bound_abstains_instead_of_claiming_optimality():
    spec = request(
        budget=0,
        obligations=[
            {"id": "o" + str(i), "state": "OPEN", "evidence_digest": None}
            for i in range(16)
        ],
        observations=[
            {
                "id": "read" + str(i),
                "mode": "READ_ONLY",
                "admission_state": "ALLOW",
                "cost_units": 0,
                "may_inform": ["o" + str(i)],
            }
            for i in range(16)
        ],
    )
    result = advice.advise(spec)
    assert result["state"] == "ABSTAIN_SEARCH_BOUND"
    assert result["selected_read_observations"] == []
    assert result["proposed_cost_units"] is None


@pytest.mark.parametrize("bad", [True, -1, 1.5, 10**20, "1", None])
def test_cost_type_and_bounds_are_strict(bad):
    spec = request(budget=bad)
    with pytest.raises(advice.EvidenceHold, match="INVALID_COST_UNITS"):
        advice.advise(spec)


def test_duplicate_ids_or_unknown_targets_are_rejected():
    spec = request()
    spec["obligations"].append(spec["obligations"][0])
    with pytest.raises(advice.EvidenceHold, match="DUPLICATE_OBLIGATION"):
        advice.advise(spec)
    spec = request()
    spec["observations"][0]["may_inform"] = ["unknown"]
    with pytest.raises(advice.EvidenceHold, match="INVALID_OBSERVATION_TARGETS"):
        advice.advise(spec)


def test_strict_json_refuses_duplicate_members_and_nonfinite():
    for raw, reason in [
        (b'{"schema":1,"schema":2}', "DUPLICATE_JSON_MEMBER"),
        (b'{"x":NaN}', "NONFINITE_JSON"),
        (b'{"x":Infinity}', "NONFINITE_JSON"),
    ]:
        with pytest.raises(advice.EvidenceHold, match=reason):
            advice.load_strict(raw)


def test_cli_creates_new_unsigned_advisory_not_overwriting_input_or_output(tmp_path):
    source = tmp_path / "request.json"
    target = tmp_path / "result.json"
    raw = json.dumps(request(), sort_keys=True).encode()
    source.write_bytes(raw)
    script = advice.__file__
    cmd = [
        sys.executable,
        "-B",
        script,
        "--input",
        str(source),
        "--output",
        str(target),
    ]
    first = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    assert first.returncode == 0, first.stderr + first.stdout
    result = json.loads(target.read_text())
    assert result["input_bytes_sha256"]
    assert result["signature"] == "UNSIGNED"
    assert result["authority_granted"] is False
    assert source.read_bytes() == raw
    second = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    assert second.returncode == 2
    assert json.loads(second.stdout)["state"] == "HOLD"
    assert target.read_text() == json.dumps(result, indent=2, allow_nan=False) + "\n"


def test_invalid_schema_and_unknown_fields_hold():
    spec = request()
    spec["admit_everything"] = True
    with pytest.raises(
        advice.EvidenceHold, match="QUERY_FIELDS_UNAVAILABLE_OR_UNKNOWN"
    ):
        advice.advise(spec)
    del spec["admit_everything"]
    spec["schema"] = "unsupported"
    with pytest.raises(advice.EvidenceHold, match="UNSUPPORTED_QUERY_SCHEMA"):
        advice.advise(spec)

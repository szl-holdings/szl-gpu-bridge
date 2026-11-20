import json
from datetime import datetime, timezone

from tools.salad import admission

NOW = datetime(2026, 11, 20, 20, 45, tzinfo=timezone.utc)

WITNESS_FRESH = {
    'state': 'REVIEW_CANDIDATE_ONLY',
    'reason': 'FRESH_NO_PRIOR_STEPS_UNATTESTED',
    'replay_steps_conditional': 0,
}

WITNESS_RESTART = {
    'state': 'REVIEW_CANDIDATE_ONLY',
    'reason': 'FULL_STATE_DECLARED_UNATTESTED',
    'replay_steps_conditional': 0,
}

GATES_PERMITTED = {'decision': 'RESTART_PERMITTED', 'restart_permitted': True}

GROUP = 'szl-train-unified-corpus-v1'


def _intent(op, intent_id):
    return json.dumps({
        'schema': 'szl.salad-admission-intent/v1',
        'intent_id': intent_id,
        'actor': 'cto-lane',
        'op': op,
        'lane_id': 'lane-budget-floor',
        'group': GROUP,
        'prev_intent_sha256': None,
    }).encode()


def _budget(**over):
    b = {
        'remaining': 0,
        'start': 0,
        'replay_step': 0,
        'next_checkpoint': 0,
        'finalize': 0,
        'reserve': 0,
    }
    b.update(over)
    return b


def _registry(budget, desired_state):
    return {
        'schema': 'szl.desired-state-registry/v1',
        'last_intent_sha256': None,
        'lanes': {
            'lane-budget-floor': {
                'group': GROUP,
                'desired_state': desired_state,
                'actors': ['cto-lane'],
                'budget': budget,
            }
        },
    }


def test_all_zero_budget_denies_create():
    r = admission.admit(
        intent_bytes=_intent('create', 'budget-floor-create-1'),
        registry=_registry(_budget(), 'NEW'),
        now=NOW,
        lease_ok=True,
        witness_verdict=dict(WITNESS_FRESH),
    )
    assert r['decision'] == 'DENY'
    assert r['reason'] == 'BUDGET_FLOOR_UNPINNED'
    assert r['salad_mutation_authorized'] is False


def test_all_zero_budget_denies_restart():
    r = admission.admit(
        intent_bytes=_intent('restart', 'budget-floor-restart-1'),
        registry=_registry(_budget(), 'STOPPED'),
        now=NOW,
        lease_ok=True,
        witness_verdict=dict(WITNESS_RESTART),
        gates_decision=dict(GATES_PERMITTED),
    )
    assert r['decision'] == 'DENY'
    assert r['reason'] == 'BUDGET_FLOOR_UNPINNED'
    assert r['salad_mutation_authorized'] is False


def test_pinned_budget_create_still_admits():
    r = admission.admit(
        intent_bytes=_intent('create', 'budget-floor-create-2'),
        registry=_registry(_budget(remaining=5000000, start=100), 'NEW'),
        now=NOW,
        lease_ok=True,
        witness_verdict=dict(WITNESS_FRESH),
    )
    assert r['decision'] == 'ADMIT'
    assert r['salad_mutation_authorized'] is True


def test_zero_budget_park_still_admits():
    r = admission.admit(
        intent_bytes=_intent('park', 'budget-floor-park-1'),
        registry=_registry(_budget(), 'STOPPED'),
        now=NOW,
        lease_ok=True,
    )
    assert r['decision'] == 'ADMIT'


def test_zero_budget_stop_still_admits():
    r = admission.admit(
        intent_bytes=_intent('stop', 'budget-floor-stop-1'),
        registry=_registry(_budget(), 'RUNNING'),
        now=NOW,
        lease_ok=True,
    )
    assert r['decision'] == 'ADMIT'
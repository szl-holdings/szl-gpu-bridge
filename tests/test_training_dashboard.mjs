// SPDX-License-Identifier: Apache-2.0
import test from "node:test";
import assert from "node:assert/strict";
import {readFile} from "node:fs/promises";

const source = await readFile(new URL("../tools/dashboard/training_command.js", import.meta.url), "utf8");
const {validateReport, snapshotView, loadStatus, createController, renderDashboard, unavailable} =
  await import("data:text/javascript;base64," + Buffer.from(source).toString("base64"));
const AT = Date.parse("2026-11-16T23:30:00Z");
const fixture = () => ({
  schema: "szl.training-dashboard/v1", status: "CURRENT",
  generated_at_utc: new Date(AT).toISOString(), scope: "SELECTED_CANDIDATES",
  source: {schema: "szl.training-observation/v1", sha256: "a".repeat(64),
    observed_at_utc: new Date(AT - 10000).toISOString(), age_seconds: 10, stale_after_seconds: 60},
  candidates: [{repo: "SZLHOLDINGS/test-candidate", state: "CHECKPOINT_PROGRESS_OBSERVED",
    observed_state: "CHECKPOINT_PROGRESS_OBSERVED", freshness: "CURRENT",
    observed_at_utc: new Date(AT - 10000).toISOString(), age_seconds: 10, revision: "b".repeat(40),
    latest_checkpoint: 100, target_steps: 500, global_step: 100, trainer_max_steps: 500,
    trainer_state_integrity: "PINNED_OBJECT_BYTES_VERIFIED", trainer_state_sha256: "d".repeat(64),
    provider: {state: "OBSERVED", observed_state: "OBSERVED", instance_count: 1, ready_count: 1,
      instance_states: ["running"], trainer_process_liveness_verified: false},
    training_complete_verified: false, promotion_authorized: false, trainer_process_liveness_verified: false}],
  training_complete_verified: false, promotion_authorized: false, trainer_process_liveness_verified: false, reason: null
});
const response = data => ({ok: true, headers: {get: () => "application/json; charset=utf-8"}, json: async () => data});
const RELEASE_BLOCKERS = [
  "TRAINING_PROVENANCE_AND_SIGNATURE_NOT_VERIFIED",
  "HELDOUT_QUALITY_AND_CONTAMINATION_NOT_ASSESSED",
  "SAFETY_AND_NEUTRAL_CAPABILITY_NOT_ASSESSED",
  "SERVING_AUTH_ISOLATION_LATENCY_ROLLBACK_NOT_ASSESSED",
  "OWNER_PROMOTION_APPROVAL_NOT_BOUND"
];
const qualificationFixture = () => {
  const report = fixture();
  const row = report.candidates[0];
  row.state = row.observed_state = "TERMINAL_METADATA_PRESENT";
  row.latest_checkpoint = row.global_step = row.target_steps;
  row.qualification = {
    state: "ARTIFACT_OBSERVATION_MATCHED", revision: row.revision,
    observed_at_utc: "2026-11-01T12:00:00Z", qualification_sha256: "e".repeat(64),
    artifact_sha256: "f".repeat(64), target_steps: row.target_steps, base_repo: "Example/base-model",
    hold_reasons: [], release_blockers: [...RELEASE_BLOCKERS],
    artifact_observation_bound: true, authenticity_verified: false,
    quality_qualification: "NOT_ESTABLISHED", compute_authorized: false, promotion_authorized: false
  };
  return report;
};

test("only the same-origin status endpoint supplies evidence", async () => {
  const requests = [];
  const data = await loadStatus(async (url, options) => {
    requests.push([url, options]); return response(fixture());
  });
  assert.equal(data.source.sha256, "a".repeat(64));
  assert.equal(requests.length, 1);
  assert.equal(requests[0][0], "/api/training/status");
  assert.equal(requests[0][1].cache, "no-store");
  assert.equal(requests[0][1].credentials, "same-origin");
});

test("schema, scope, revisions and unauthorized authority fail closed", () => {
  for (const mutate of [
    r => { r.schema = "unknown/v1"; },
    r => { r.scope = "WHOLE_FLEET"; },
    r => { r.promotion_authorized = true; },
    r => { r.trainer_process_liveness_verified = true; },
    r => { r.candidates[0].observed_state = "COMPLETE"; },
    r => { r.candidates[0].trainer_state_integrity = "VERIFIED"; },
    r => { r.candidates[0].trainer_state_sha256 = null; },
    r => { r.candidates[0].trainer_state_sha256 = "bad-digest"; },
    r => { r.candidates[0].provider.instance_count = 2; },
    r => { r.candidates[0].provider.ready_count = 2; },
    r => { r.candidates[0].provider.observed_state = "TRAINING"; },
    r => { r.candidates[0].provider.instance_states = ["not-a-state"]; },
    r => { r.candidates[0].provider.trainer_process_liveness_verified = true; },
    r => { r.candidates[0].training_complete_verified = true; },
    r => { r.candidates[0].trainer_process_liveness_verified = true; },
    r => { r.candidates[0].revision = "main"; },
    r => { r.source.age_seconds = -1; },
    r => { r.candidates.push(structuredClone(r.candidates[0])); }
  ]) {
    const data = fixture(); mutate(data);
    assert.throws(() => validateReport(data), /RESPONSE_CONTRACT_INVALID/);
  }
});

test("exact forty- and sixty-four-character revisions remain admissible", () => {
  const data = fixture(); data.candidates[0].revision = "c".repeat(64);
  assert.equal(validateReport(data).candidates[0].revision, "c".repeat(64));
  data.candidates[0].revision = "c".repeat(63);
  assert.throws(() => validateReport(data), /RESPONSE_CONTRACT_INVALID/);
});

test("a displayed observation expires even without another successful fetch", () => {
  const data = fixture();
  const current = snapshotView(validateReport(data), AT, AT + 49000);
  const stale = snapshotView(data, AT, AT + 51000);
  assert.equal(current.status, "CURRENT");
  assert.equal(stale.status, "STALE");
  assert.equal(stale.candidates[0].state, "STALE");
  assert.equal(stale.candidates[0].provider.state, "STALE");
  assert.equal(stale.candidates[0].observed_state, "CHECKPOINT_PROGRESS_OBSERVED");
  assert.equal(stale.candidates[0].provider.observed_state, "OBSERVED");
  assert.equal(stale.promotion_authorized, false);
  assert.equal(stale.candidates[0].training_complete_verified, false);
  assert.equal(data.candidates[0].freshness, "CURRENT");
});

test("an earlier candidate expires the aggregate while newer rows retain their own freshness", () => {
  const data = fixture();
  data.source.stale_after_seconds = 660;
  const recent = structuredClone(data.candidates[0]); recent.repo = "SZLHOLDINGS/recent";
  data.candidates[0].age_seconds = 650;
  data.candidates.push(recent);
  const view = snapshotView(validateReport(data), AT, AT + 20000);
  assert.equal(view.source.age_seconds, 30);
  assert.equal(view.status, "STALE");
  assert.equal(view.candidates[0].freshness, "STALE");
  assert.equal(view.candidates[1].freshness, "CURRENT");
  assert.equal(view.candidates[1].provider.state, "OBSERVED");
});

test("failed refresh clears prior current evidence and does not show zero candidates", async () => {
  let fail = false; let shown;
  const controller = createController({now: () => AT, render: data => { shown = data; },
    fetcher: async () => { if (fail) throw new Error("offline"); return response(fixture()); }});
  await controller.refresh();
  assert.equal(shown.status, "CURRENT");
  fail = true; await controller.refresh();
  assert.equal(shown.status, "UNAVAILABLE");
  assert.equal(shown.source, null);
  assert.deepEqual(shown.candidates, []);
});

test("HTML fallback or malformed evidence cannot become a current observation", async () => {
  await assert.rejects(loadStatus(async () => ({
    ok: true, headers: {get: () => "text/html"}, json: async () => fixture()
  })), /STATUS_ENDPOINT_UNAVAILABLE/);
  let shown; const malformed = fixture(); malformed.schema = "wrong";
  const controller = createController({now: () => AT, render: data => { shown = data; },
    fetcher: async () => response(malformed)});
  await controller.refresh();
  assert.equal(shown.status, "UNAVAILABLE");
  assert.equal(shown.reason, "RESPONSE_CONTRACT_INVALID");
});

test("an old in-flight success cannot overwrite a newer failure", async () => {
  let release; let calls = 0; let shown;
  const old = new Promise(resolve => { release = resolve; });
  const controller = createController({now: () => AT, render: data => { shown = data; },
    fetcher: async () => { if (++calls === 1) return old; throw new Error("offline"); }});
  const first = controller.refresh();
  await controller.refresh();
  release(response(fixture())); await first;
  assert.equal(shown.status, "UNAVAILABLE");
});

function fakeDocument() {
  const elements = new Map();
  const create = tag => ({
    tag, children: [], textContent: "", className: "", id: "", open: false,
    querySelectorAll(selector) {
      const descendants = this.children.flatMap(child => [child, ...child.querySelectorAll("*")]);
      return selector === "*" ? descendants : descendants.filter(child => child.tag === "details" &&
        (selector === "details" || selector === "details[open]" && child.open));
    },
    set innerHTML(_) { throw new Error("untrusted HTML used"); },
    append(...items) { this.children.push(...items); },
    replaceChildren(...items) { this.children = items; }
  });
  return {createElement: create, getElementById(id) {
    if (!elements.has(id)) elements.set(id, create("div"));
    return elements.get(id);
  }};
}
const collect = node => node.textContent + node.children.map(collect).join(" ");

test("rendering treats hostile strings as text and keeps exact evidence identifiers", () => {
  const document = fakeDocument();
  const hostile = '<img src=x onerror="throw 1">';
  renderDashboard(document, unavailable(hostile, AT));
  assert.match(document.getElementById("status-message").textContent, /<img src=x/);
  assert.equal(document.getElementById("candidate-count").textContent, "Unavailable");
  const data = fixture(); data.candidates[0].provider.instance_states = [hostile];
  assert.throws(() => validateReport(data), /RESPONSE_CONTRACT_INVALID/);
  renderDashboard(document, snapshotView(data, AT, AT));
  assert.ok(collect(document.getElementById("candidates")).includes(hostile));
  assert.ok(collect(document.getElementById("candidates")).includes("b".repeat(40)));
  assert.ok(collect(document.getElementById("candidates")).includes("d".repeat(64)));
  assert.equal(document.getElementById("source-hash").textContent, "a".repeat(64));
});

test("a browser clock reversal removes current evidence", () => {
  const view = snapshotView(fixture(), AT, AT - 1000);
  assert.equal(view.status, "UNAVAILABLE");
  assert.equal(view.source, null);
});


test("older projections without qualification are explicitly unconfigured", () => {
  const original = fixture();
  const report = validateReport(original);
  assert.equal(original.candidates[0].qualification, undefined);
  assert.equal(report.candidates[0].qualification.state, "NOT_CONFIGURED");
  assert.equal(report.candidates[0].qualification.artifact_observation_bound, false);
  assert.deepEqual(report.candidates[0].qualification.release_blockers, RELEASE_BLOCKERS);
  const document = fakeDocument();
  renderDashboard(document, report);
  assert.match(collect(document.getElementById("candidates")), /Artifact check not configured/);
  assert.doesNotMatch(collect(document.getElementById("candidates")), /Bound to recorded revision/);
});

test("an exact artifact match remains separate from checkpoint evidence and never grants release authority", async () => {
  const document = fakeDocument();
  let shown;
  const controller = createController({now: () => AT,
    fetcher: async () => response(qualificationFixture()),
    render: report => { shown = report; renderDashboard(document, report); }});
  await controller.refresh();
  const q = shown.candidates[0].qualification;
  assert.equal(q.state, "ARTIFACT_OBSERVATION_MATCHED");
  assert.equal(q.artifact_observation_bound, true);
  assert.equal(q.authenticity_verified, false);
  assert.equal(q.quality_qualification, "NOT_ESTABLISHED");
  assert.equal(q.compute_authorized, false);
  assert.equal(q.promotion_authorized, false);
  assert.equal(shown.promotion_authorized, false);
  assert.equal(shown.candidates[0].training_complete_verified, false);
  const row = document.getElementById("candidates").children[0];
  const artifact = collect(row.children[0]);
  const checkpoint = collect(row.children[1]);
  assert.match(artifact, /Recorded artifact observation matched/);
  assert.match(artifact, /Bound to recorded revision/);
  assert.match(artifact, /Quality not established · compute and promotion unauthorized/);
  assert.match(artifact, /Authenticity remains unverified/);
  assert.match(checkpoint, /TERMINAL METADATA PRESENT/);
  assert.doesNotMatch(checkpoint, /Recorded artifact check/);
  assert.ok(artifact.includes("e".repeat(64)));
  assert.ok(artifact.includes("f".repeat(64)));
  for (const code of RELEASE_BLOCKERS) assert.ok(artifact.includes(code));
  const requirements = row.querySelectorAll("details").find(item => item.id.startsWith("qualification-"));
  assert.equal(requirements.open, false);
});

test("a changed revision renders the existing artifact record as unbound", () => {
  const data = qualificationFixture();
  data.candidates[0].revision = "c".repeat(40);
  Object.assign(data.candidates[0].qualification, {
    state: "REVISION_MISMATCH", artifact_observation_bound: false,
    hold_reasons: ["QUALIFICATION_REVISION_MISMATCH"]
  });
  const report = validateReport(data);
  const document = fakeDocument();
  renderDashboard(document, report);
  const display = collect(document.getElementById("candidates"));
  assert.match(display, /Artifact revision mismatch/);
  assert.match(display, /Unbound to this candidate observation/);
  assert.doesNotMatch(display, /Bound to recorded revision/);
  assert.ok(display.includes("b".repeat(40)));
  assert.ok(display.includes("c".repeat(40)));
  assert.ok(display.includes("f".repeat(64)));
  assert.equal(report.candidates[0].qualification.promotion_authorized, false);
});

test("a changed target cannot retain an artifact binding", () => {
  const data = qualificationFixture();
  data.candidates[0].target_steps = 600;
  assert.throws(() => validateReport(data), /RESPONSE_CONTRACT_INVALID/);
  Object.assign(data.candidates[0].qualification, {
    state: "INVALID", artifact_observation_bound: false,
    hold_reasons: ["QUALIFICATION_TARGET_MISMATCH"]
  });
  const document = fakeDocument();
  renderDashboard(document, validateReport(data));
  const display = collect(document.getElementById("candidates"));
  assert.match(display, /Artifact record invalid/);
  assert.match(display, /recorded target differs/);
  assert.doesNotMatch(display, /Bound to recorded revision/);
});

test("a false binding or elevated authority clears a prior matched record", async () => {
  for (const mutate of [
    r => { r.candidates[0].revision = "c".repeat(40); },
    r => { r.candidates[0].revision = null; },
    r => { r.candidates[0].qualification.artifact_observation_bound = false; },
    r => { r.candidates[0].qualification.state = "ARTIFACT_HOLD"; },
    r => { r.candidates[0].qualification.authenticity_verified = true; },
    r => { r.candidates[0].qualification.quality_qualification = "ESTABLISHED"; },
    r => { r.candidates[0].qualification.compute_authorized = true; },
    r => { r.candidates[0].qualification.promotion_authorized = true; },
    r => { r.candidates[0].qualification.release_blockers.pop(); },
    r => { r.candidates[0].qualification.release_blockers[0] = RELEASE_BLOCKERS[1]; },
    r => { r.candidates[0].qualification.hold_reasons = ["ARBITRARY_PROVIDER_ERROR"]; },
    r => { r.candidates[0].qualification.artifact_sha256 = null; },
    r => { r.candidates[0].qualification.qualification_sha256 = "bad-digest"; },
    r => { r.candidates[0].qualification.observed_at_utc = null; }
  ]) {
    const document = fakeDocument();
    let data = qualificationFixture(); let shown;
    const controller = createController({now: () => AT,
      fetcher: async () => response(data),
      render: report => { shown = report; renderDashboard(document, report); }});
    await controller.refresh();
    assert.match(collect(document.getElementById("candidates")), /Bound to recorded revision/);
    data = qualificationFixture(); mutate(data);
    await controller.refresh();
    assert.equal(shown.status, "UNAVAILABLE");
    assert.equal(shown.reason, "RESPONSE_CONTRACT_INVALID");
    assert.doesNotMatch(collect(document.getElementById("candidates")), /Bound to recorded revision/);
    assert.ok(!collect(document.getElementById("candidates")).includes("f".repeat(64)));
  }
});

test("artifact failures remain explicit while checkpoint observations remain available", () => {
  for (const [state, reason, message] of [
    ["UNAVAILABLE", "QUALIFICATION_UNAVAILABLE", "Artifact check unavailable"],
    ["INVALID", "QUALIFICATION_INVALID", "Artifact record invalid"],
    ["NOT_ASSESSED", "QUALIFICATION_CANDIDATE_NOT_ASSESSED", "Candidate not assessed"],
    ["ARTIFACT_HOLD", "QUALIFICATION_ARTIFACT_HELD", "Artifact check held"]
  ]) {
    const data = fixture();
    data.candidates[0].qualification = {
      ...validateReport(fixture()).candidates[0].qualification, state, hold_reasons: [reason]
    };
    const document = fakeDocument();
    const report = validateReport(data);
    renderDashboard(document, report);
    const display = collect(document.getElementById("candidates"));
    assert.equal(report.status, "CURRENT");
    assert.ok(display.includes(message));
    assert.match(display, /CHECKPOINT PROGRESS OBSERVED/);
    assert.doesNotMatch(display, /Bound to recorded revision/);
    for (const code of RELEASE_BLOCKERS) assert.ok(display.includes(code));
  }
});

test("source expiry retains the immutable artifact record as a historical revision match", () => {
  const report = validateReport(qualificationFixture());
  const document = fakeDocument();
  renderDashboard(document, report);
  const detailId = "qualification-" + report.candidates[0].repo;
  document.getElementById("candidates").querySelectorAll("details")
    .find(item => item.id === detailId).open = true;
  const snapshot = structuredClone(report.candidates[0].qualification);
  const aged = snapshotView(report, AT, AT + 86400000);
  renderDashboard(document, aged);
  assert.equal(aged.status, "STALE");
  assert.deepEqual(aged.candidates[0].qualification, snapshot);
  assert.equal(aged.candidates[0].qualification.artifact_observation_bound, true);
  const display = collect(document.getElementById("candidates"));
  assert.match(display, /Bound to recorded revision/);
  assert.match(display, /Historical candidate observation; current revision is unestablished/);
  assert.match(display, /Quality not established · compute and promotion unauthorized/);
  assert.equal(document.getElementById("candidates").querySelectorAll("details")
    .find(item => item.id === detailId).open, true);
});

test("an endpoint failure removes the previously rendered artifact match", async () => {
  const document = fakeDocument();
  let fail = false;
  const controller = createController({now: () => AT,
    fetcher: async () => { if (fail) throw new Error("offline"); return response(qualificationFixture()); },
    render: report => renderDashboard(document, report)});
  await controller.refresh();
  assert.match(collect(document.getElementById("candidates")), /Bound to recorded revision/);
  fail = true;
  await controller.refresh();
  assert.equal(document.getElementById("report-status").textContent, "UNAVAILABLE");
  assert.doesNotMatch(collect(document.getElementById("candidates")), /Bound to recorded revision/);
  assert.ok(!collect(document.getElementById("candidates")).includes("f".repeat(64)));
});

test("artifact metadata uses text rendering even for hostile base labels", () => {
  const report = qualificationFixture();
  const hostile = '<img src=x onerror="throw 1">';
  report.candidates[0].qualification.base_repo = hostile;
  const document = fakeDocument();
  renderDashboard(document, validateReport(report));
  assert.ok(collect(document.getElementById("candidates")).includes(hostile));
});

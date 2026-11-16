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
    tag, children: [], textContent: "", className: "",
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

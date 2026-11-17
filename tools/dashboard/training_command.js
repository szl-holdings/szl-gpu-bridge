// SPDX-License-Identifier: Apache-2.0
// This view reads one local projection. It has no training or publication actions.
const SCHEMA = "szl.training-dashboard/v1";
const SOURCE_SCHEMA = "szl.training-observation/v1";
const STATES = new Set(["CURRENT", "STALE", "UNAVAILABLE", "INVALID"]);
const OBSERVATIONS = new Set(["UNAVAILABLE", "NO_CHECKPOINT_OBSERVED", "TARGET_CONTRACT_MISMATCH",
  "TERMINAL_STEP_OVERSHOOT", "CHECKPOINT_PROGRESS_OBSERVED", "TERMINAL_METADATA_PRESENT",
  "TARGET_CHECKPOINT_WITHOUT_FINAL_BUNDLE"]);
const PROVIDER_OBSERVATIONS = new Set(["OBSERVED", "UNAVAILABLE"]);
const INSTANCE_STATES = new Set(["allocating", "allocated", "creating", "running", "stopping",
  "stopped", "failed", "error", "downloading", "starting", "pending", "queued",
  "provisioning", "succeeded", "unknown"]);
const object = value => value !== null && typeof value === "object" && !Array.isArray(value);
const text = (value, max = 160) => typeof value === "string" && value.length <= max;
const count = value => Number.isSafeInteger(value) && value >= 0 && value <= 1_000_000_000;
const maybeCount = value => value === null || count(value);
const age = value => Number.isFinite(value) && value >= 0 && value <= Number.MAX_SAFE_INTEGER;
const utc = value => text(value, 64) &&
  /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,9})?(?:Z|\+00:00)$/.test(value) &&
  Number.isFinite(Date.parse(value));
const authority = value => value.training_complete_verified === false &&
  value.promotion_authorized === false && value.trainer_process_liveness_verified === false;

export function unavailable(reason = "STATUS_READ_FAILED", now = Date.now()) {
  return {schema: SCHEMA, status: "UNAVAILABLE", generated_at_utc: new Date(now).toISOString(),
    source: null, scope: "SELECTED_CANDIDATES", candidates: [],
    training_complete_verified: false, promotion_authorized: false, trainer_process_liveness_verified: false, reason};
}

export function validateReport(report) {
  const fail = () => { throw new Error("RESPONSE_CONTRACT_INVALID"); };
  if (!object(report) || report.schema !== SCHEMA || !STATES.has(report.status) ||
      report.scope !== "SELECTED_CANDIDATES" || !authority(report) ||
      !utc(report.generated_at_utc) || !Array.isArray(report.candidates) ||
      !(report.reason === null || text(report.reason, 300))) fail();
  if (report.status === "UNAVAILABLE" || report.status === "INVALID") {
    if (report.candidates.length !== 0) fail();
    return report;
  }
  const source = report.source;
  if (!object(source) || source.schema !== SOURCE_SCHEMA ||
      !text(source.sha256, 64) || !/^[0-9a-f]{64}$/.test(source.sha256) || !utc(source.observed_at_utc) ||
      !age(source.age_seconds) ||
      !count(source.stale_after_seconds) || source.stale_after_seconds < 1 || source.stale_after_seconds > 7260 ||
      report.candidates.length < 1 || report.candidates.length > 6) fail();
  const repos = new Set();
  for (const row of report.candidates) {
    if (!object(row) || !text(row.repo, 140) || !/^SZLHOLDINGS\/[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$/.test(row.repo) ||
        repos.has(row.repo) || !OBSERVATIONS.has(row.observed_state) ||
        !["CURRENT", "STALE"].includes(row.freshness) ||
        row.state !== (row.freshness === "STALE" ? "STALE" : row.observed_state) ||
        !utc(row.observed_at_utc) || !age(row.age_seconds) ||
        !(row.revision === null || (text(row.revision, 64) && /^(?:[0-9a-f]{40}|[0-9a-f]{64})$/.test(row.revision))) ||
        !maybeCount(row.latest_checkpoint) || !count(row.target_steps) || row.target_steps < 1 ||
        !maybeCount(row.global_step) || !maybeCount(row.trainer_max_steps) ||
        ![null, "PINNED_OBJECT_BYTES_VERIFIED"].includes(row.trainer_state_integrity) ||
        !(row.trainer_state_integrity === null ? row.trainer_state_sha256 === null :
          text(row.trainer_state_sha256, 64) && /^[0-9a-f]{64}$/.test(row.trainer_state_sha256)) ||
        !authority(row)) fail();
    repos.add(row.repo);
    const provider = row.provider;
    if (!object(provider) || !PROVIDER_OBSERVATIONS.has(provider.observed_state) ||
        provider.state !== (row.freshness === "STALE" ? "STALE" : provider.observed_state) ||
        provider.trainer_process_liveness_verified !== false ||
        !Array.isArray(provider.instance_states) || provider.instance_states.length > 1000 ||
        !provider.instance_states.every(state => INSTANCE_STATES.has(state))) fail();
    if (provider.observed_state === "OBSERVED") {
      if (!count(provider.instance_count) || provider.instance_count > 1000 ||
          provider.instance_count !== provider.instance_states.length ||
          !count(provider.ready_count) || provider.ready_count > provider.instance_count) fail();
    } else if (provider.instance_count !== null || provider.ready_count !== null ||
        provider.instance_states.length !== 0) fail();
  }
  return report;
}

export function snapshotView(report, receivedAt, now = Date.now()) {
  if (now < receivedAt) return unavailable("BROWSER_CLOCK_MOVED_BACKWARDS", now);
  if (report.status === "UNAVAILABLE" || report.status === "INVALID") {
    return {...report, source: null, candidates: []};
  }
  const elapsed = (now - receivedAt) / 1000;
  const age = report.source.age_seconds + elapsed;
  const limit = report.source.stale_after_seconds;
  const sourceStale = age > limit;
  const candidates = report.candidates.map(row => {
    const rowAge = row.age_seconds + elapsed;
    const rowStale = sourceStale || row.freshness === "STALE" || rowAge > limit;
    return {...row, age_seconds: rowAge, freshness: rowStale ? "STALE" : "CURRENT",
      state: rowStale ? "STALE" : row.state,
      provider: {...row.provider, state: rowStale ? "STALE" : row.provider.state}};
  });
  const stale = report.status === "STALE" || sourceStale || candidates.some(row => row.freshness === "STALE");
  return {...report, status: stale ? "STALE" : "CURRENT",
    source: {...report.source, age_seconds: age}, candidates};
}

export async function loadStatus(fetcher, signal) {
  const response = await fetcher("/api/training/status", {
    cache: "no-store", credentials: "same-origin", headers: {Accept: "application/json"}, signal
  });
  if (!response.ok || !(response.headers.get("content-type") || "").toLowerCase().startsWith("application/json")) {
    throw new Error("STATUS_ENDPOINT_UNAVAILABLE");
  }
  return validateReport(await response.json());
}

export function createController({fetcher, render, now = Date.now}) {
  let report = unavailable("WAITING_FOR_STATUS", now());
  let receivedAt = now();
  let request = 0;
  const tick = () => render(snapshotView(report, receivedAt, now()));
  async function refresh() {
    const number = ++request;
    const abort = new AbortController();
    const timeout = setTimeout(() => abort.abort(), 8000);
    try {
      const result = await loadStatus(fetcher, abort.signal);
      if (number !== request) return;
      report = result;
    } catch (error) {
      if (number !== request) return;
      report = unavailable(error.message === "RESPONSE_CONTRACT_INVALID" ?
        "RESPONSE_CONTRACT_INVALID" : "STATUS_READ_FAILED", now());
    } finally {
      clearTimeout(timeout);
      if (number === request) { receivedAt = now(); tick(); }
    }
  }
  return {refresh, tick};
}

function ageLabel(seconds) {
  if (!Number.isFinite(seconds)) return "Unavailable";
  const age = Math.max(0, Math.floor(seconds));
  if (age < 60) return age + " sec";
  if (age < 3600) return Math.floor(age / 60) + " min " + (age % 60) + " sec";
  if (age < 86400) return Math.floor(age / 3600) + " hr " + Math.floor(age % 3600 / 60) + " min";
  return Math.floor(age / 86400) + " days " + Math.floor(age % 86400 / 3600) + " hr";
}
const label = value => String(value ?? "UNAVAILABLE").replaceAll("_", " ");
const value = number => number === null || number === undefined ? "Unavailable" : String(number);

export function renderDashboard(document, report) {
  const set = (id, content) => { document.getElementById(id).textContent = content; };
  const node = (tag, content, className = "") => {
    const element = document.createElement(tag);
    element.textContent = content;
    element.className = className;
    return element;
  };
  const status = document.getElementById("report-status");
  status.textContent = report.status;
  status.className = "value " + report.status.toLowerCase();
  const hasSource = report.source !== null;
  set("candidate-count", hasSource ? String(report.candidates.length) : "Unavailable");
  set("source-age", hasSource ? ageLabel(report.source.age_seconds) : "Unavailable");
  set("freshness-window", hasSource ? "Freshness window: " + ageLabel(report.source.stale_after_seconds) : "Freshness window unavailable");
  set("observed-at", hasSource ? report.source.observed_at_utc : "Unavailable");
  set("generated-at", report.generated_at_utc);
  set("source-schema", hasSource ? report.source.schema : "Unavailable");
  set("source-hash", hasSource ? report.source.sha256 : "Unavailable");
  const message = report.status === "CURRENT"
    ? "Current observation for the selected candidates. Completion and liveness remain unverified."
    : report.status === "STALE"
      ? "Stale observation. Candidate and provider details below describe the recorded observation."
      : "Evidence unavailable. " + label(report.reason) + ". Refresh evidence to retry.";
  set("status-message", message);
  const body = document.getElementById("candidates");
  body.replaceChildren();
  if (!hasSource || !report.candidates.length) {
    const row = node("tr", "");
    const cell = node("td", "Candidate evidence unavailable. No previous observation is presented as current.");
    cell.colSpan = 5; row.append(cell); body.append(row); return;
  }
  for (const item of report.candidates) {
    const row = node("tr", "");
    const candidate = node("td", "");
    candidate.append(node("div", item.repo, "repo"));
    candidate.append(node("div", "Revision: " + value(item.revision), "detail mono"));
    const checkpoint = node("td", "");
    checkpoint.append(node("div", label(item.observed_state)));
    checkpoint.append(node("div", "Checkpoint: " + value(item.latest_checkpoint) + " · target: " + value(item.target_steps), "detail"));
    checkpoint.append(node("div", "Trainer step: " + value(item.global_step) + " / " + value(item.trainer_max_steps), "detail"));
    checkpoint.append(node("div", "Reported trainer-state integrity: " + label(item.trainer_state_integrity), "detail"));
    if (item.trainer_state_sha256 !== null) {
      const digest = node("details", "", "detail");
      digest.append(node("summary", "Trainer-state SHA-256"));
      digest.append(node("p", item.trainer_state_sha256, "mono"));
      checkpoint.append(digest);
    }
    const provider = node("td", "");
    provider.append(node("div", label(item.provider.observed_state)));
    provider.append(node("div", "Instances: " + value(item.provider.instance_count) + " · ready: " + value(item.provider.ready_count), "detail"));
    provider.append(node("div", item.provider.instance_states.length ? item.provider.instance_states.join(", ") : "Instance states unavailable", "detail"));
    const observed = node("td", "");
    observed.append(node("div", item.observed_at_utc, "mono"));
    observed.append(node("div", ageLabel(item.age_seconds) + " ago", "detail"));
    const fresh = node("td", "");
    fresh.append(node("span", item.freshness, "badge " + item.freshness.toLowerCase()));
    if (item.freshness === "STALE") fresh.append(node("div", "Historical observation", "detail"));
    row.append(candidate, checkpoint, provider, observed, fresh);
    body.append(row);
  }
}

if (typeof document !== "undefined") {
  const refresh = document.getElementById("refresh-button");
  const controller = createController({
    fetcher: (...args) => fetch(...args),
    render: report => renderDashboard(document, report)
  });
  const read = async () => {
    refresh.disabled = true;
    try { await controller.refresh(); } finally { refresh.disabled = false; }
  };
  refresh.addEventListener("click", read);
  controller.tick();
  read();
  setInterval(() => { if (!refresh.disabled) read(); }, 30000);
  setInterval(controller.tick, 1000);
}

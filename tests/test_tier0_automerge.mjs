import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';

// Execute the actual inline evaluator against an in-memory GitHub boundary.
// No checkout, credential, provider call, or real merge is used by these tests.
const workflow = readFileSync(new URL('../.github/workflows/auto-merge-tier0.yml', import.meta.url), 'utf8');
const source = workflow.split('          script: |\n')[1]
  .split('\n').map(line => line.replace(/^            /, '')).join('\n');
const AsyncFunction = Object.getPrototypeOf(async function () {}).constructor;
const evaluate = new AsyncFunction('github', 'context', 'console', source);
const HEAD = 'a'.repeat(40);

function fixture(options = {}) {
  const state = {
    files: [{ filename: 'requirements/ci/contract.txt' }],
    suites: [
      { id: 1, status: 'completed', conclusion: 'success' },
      { id: 2, status: 'in_progress', conclusion: null },
    ],
    statuses: [], mergeCalls: [], comments: [], reads: 0,
    protected: true, final: {}, ...options,
  };
  const pr = {
    number: 42, state: 'open', draft: false, user: { login: 'dependabot[bot]' },
    head: { sha: HEAD, repo: { full_name: 'owner/repo' } },
    base: { ref: 'main', sha: 'c'.repeat(40), repo: { full_name: 'owner/repo' } },
    changed_files: state.changedFiles ?? state.files.length,
    mergeable: true, mergeable_state: 'unstable',
  };
  const page = (rows, params) => rows.slice(((params.page ?? 1) - 1) * params.per_page,
    (params.page ?? 1) * params.per_page);
  const github = {
    rest: {
      pulls: {
        get: async () => {
          state.reads += 1;
          return { data: structuredClone(state.reads === 1 ? pr : { ...pr, ...state.final }) };
        },
        listFiles: async params => ({ data: page(state.files, params) }),
        merge: async params => {
          state.mergeCalls.push(params);
          assert.equal(params.sha, HEAD, 'merge must bind the evaluated SHA');
          if (state.mergeError) throw state.mergeError;
          return { data: { merged: state.mergeResult ?? true } };
        },
      },
      checks: {
        listSuitesForRef: async params => ({ data: {
          check_suites: page(state.suites, params),
          total_count: state.suiteCount ?? state.suites.length,
        } }),
      },
      actions: { getWorkflowRun: async () => ({ data: { check_suite_id: 2 } }) },
      repos: {
        listCommitStatusesForRef: async params => ({ data: page(state.statuses, params) }),
        getBranch: async () => ({ data: { protected: state.protected } }),
      },
      issues: { createComment: async params => { state.comments.push(params); } },
    },
    paginate: async (method, params) => {
      const rows = [];
      for (let index = 1; index < 100; index += 1) {
        const { data } = await method({ ...params, page: index });
        const chunk = Array.isArray(data) ? data : data.check_suites;
        rows.push(...chunk);
        if (chunk.length < params.per_page) return rows;
      }
      throw new Error('test pagination unexpectedly unbounded');
    },
  };
  const context = { repo: { owner: 'owner', repo: 'repo' }, runId: 123,
    eventName: state.workflowRun ? 'workflow_run' : 'pull_request_target',
    payload: state.workflowRun ? { workflow_run: state.workflowRun }
      : { pull_request: { number: 42 } } };
  state.run = () => evaluate(github, context, { log() {} });
  return state;
}

test('qualifies an unchanged PR without waiting for its own running evaluator', async () => {
  const state = fixture();
  await state.run();
  assert.equal(state.mergeCalls.length, 1);
  assert.equal(state.comments.length, 1);
});

test('inspects unsafe changed files beyond page one', async () => {
  const files = Array.from({ length: 100 }, (_, i) => ({ filename: `pkg${i}/package-lock.json` }));
  files.push({ filename: 'src/unsafe.py' });
  const state = fixture({ files });
  await state.run();
  assert.equal(state.mergeCalls.length, 0);
});

test('accepts complete safe inventories longer than one page', async () => {
  const files = Array.from({ length: 101 }, (_, i) => ({ filename: `pkg${i}/package-lock.json` }));
  const state = fixture({ files });
  await state.run();
  assert.equal(state.mergeCalls.length, 1);
});

test('declines an empty or truncated changed-file inventory', async () => {
  for (const options of [{ files: [] }, { changedFiles: 3001 }]) {
    const state = fixture(options);
    await state.run();
    assert.equal(state.mergeCalls.length, 0);
  }
});

test('inspects failed check suites beyond page one', async () => {
  const suites = Array.from({ length: 101 }, (_, i) => ({
    id: i + 10, status: 'completed', conclusion: i === 100 ? 'failure' : 'success',
  }));
  const state = fixture({ suites });
  await state.run();
  assert.equal(state.mergeCalls.length, 0);
});

test('declines incomplete check-suite inventory and absence of independent checks', async () => {
  for (const options of [{ suiteCount: 99 }, { suites: [{ id: 2, status: 'in_progress' }] }]) {
    const state = fixture(options);
    await state.run();
    assert.equal(state.mergeCalls.length, 0);
  }
});

test('inspects latest commit statuses beyond page one', async () => {
  const statuses = Array.from({ length: 101 }, (_, i) => ({
    context: `check-${i}`, state: i === 100 ? 'pending' : 'success',
  }));
  const state = fixture({ statuses });
  await state.run();
  assert.equal(state.mergeCalls.length, 0);
});

test('uses the newest status per context, not a superseded failure', async () => {
  const state = fixture({ statuses: [
    { context: 'build', state: 'success' }, { context: 'build', state: 'failure' },
  ] });
  await state.run();
  assert.equal(state.mergeCalls.length, 1);
});

test('anchors path components and checks the source of a rename', async () => {
  for (const file of [
    { filename: 'src/evilpackage-lock.json' },
    { filename: 'src/notrequirements.in' },
    { filename: 'package-lock.json', previous_filename: 'src/executable.py' },
  ]) {
    const state = fixture({ files: [file] });
    await state.run();
    assert.equal(state.mergeCalls.length, 0);
  }
});

test('declines a head that advanced after evidence was read', async () => {
  const state = fixture({ final: { head: { sha: 'b'.repeat(40) } } });
  await state.run();
  assert.equal(state.mergeCalls.length, 0);
});

test('server rejects a head race after the final read without a success receipt', async () => {
  const state = fixture({ mergeError: new Error('409 head changed') });
  await assert.rejects(state.run(), /409 head changed/);
  assert.equal(state.mergeCalls.length, 1);
  assert.equal(state.comments.length, 0);
});

test('declines draft, blocked, behind, unknown, or unprotected states', async () => {
  for (const options of [
    { final: { draft: true } }, { final: { mergeable_state: 'blocked' } },
    { final: { mergeable_state: 'behind' } }, { final: { mergeable: null } },
    { protected: false },
  ]) {
    const state = fixture(options);
    await state.run();
    assert.equal(state.mergeCalls.length, 0);
  }
});

test('honors protected-merge rejection without bypass or success receipt', async () => {
  const state = fixture({ mergeError: new Error('405 required review missing') });
  await assert.rejects(state.run(), /405 required review missing/);
  assert.equal(state.mergeCalls.length, 1);
  assert.equal(state.comments.length, 0);
});

test('does not claim a merge when the API returns merged=false', async () => {
  const state = fixture({ mergeResult: false });
  await state.run();
  assert.equal(state.comments.length, 0);
});

function completion(overrides = {}) {
  return { name: 'GPU bridge contract CI', event: 'pull_request',
    status: 'completed', conclusion: 'success', head_sha: HEAD,
    head_repository: { full_name: 'owner/repo' },
    pull_requests: [{ number: 42 }], ...overrides };
}

test('known verifier completion wakes evaluation and can qualify the current head', async () => {
  for (const name of ['GPU bridge contract CI', 'CodeQL']) {
    const state = fixture({ workflowRun: completion({ name }) });
    await state.run();
    assert.equal(state.mergeCalls.length, 1);
    assert.equal(state.mergeCalls[0].pull_number, 42);
  }
});

test('workflow completion refuses self-loops, failed runs, forks and unrelated events', async () => {
  for (const fields of [
    { name: 'Auto-merge safe dependencies' }, { conclusion: 'failure' },
    { status: 'in_progress' }, { event: 'push' },
    { head_repository: { full_name: 'other/repo' } }, { pull_requests: [] },
  ]) {
    const state = fixture({ workflowRun: completion(fields) });
    await state.run();
    assert.equal(state.mergeCalls.length, 0);
  }
});

test('stale verifier completion does not qualify a newer head', async () => {
  const state = fixture({ workflowRun: completion({ head_sha: 'b'.repeat(40) }) });
  await state.run();
  assert.equal(state.mergeCalls.length, 0);
});

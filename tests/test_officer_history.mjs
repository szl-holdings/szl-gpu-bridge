// Static-only UI regression tests. Never import officer Python apps: importing
// the dreamer creates a key and starts its signing/uploading loop.
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import test from 'node:test';
import vm from 'node:vm';

const root = process.env.OFFICER_SOURCE_ROOT || path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const roles = [
  {name: 'eternal-auditor', sha: 'body_sha256', sig: 'signature', timestamp: 'ts'},
  {name: 'estate-gardener', sha: 'row_sha256', sig: null, timestamp: 'ts'},
  {name: 'szl-dream-architect', sha: 'row_sha256', sig: 'khipu_signature', timestamp: 'timestamp_utc'},
];
const flush = async () => { for (let i = 0; i < 20; i++) await Promise.resolve(); };
function deferred() {
  let resolve;
  let reject;
  const promise = new Promise((res, rej) => { resolve = res; reject = rej; });
  return {promise, resolve, reject};
}
function receipt(role, cycle = 1) {
  return {
    cycle, [role.timestamp]: '2026-11-10T15:00:00Z', [role.sha]: 'a'.repeat(64),
    ...(role.sig ? {[role.sig]: 'b'.repeat(role.name === 'szl-dream-architect' ? 128 : 64)} : {}),
    ...(role.name !== 'estate-gardener' ? {chain_state: 'CONTINUOUS', findings: [{check: 'inventory', verdict: 'PASS', value: 1}]} : {}),
  };
}
function harness(role, options = {}) {
  const html = readFileSync(path.join(root, 'hf-spaces', role.name, 'static/index.html'), 'utf8');
  const script = html.match(/<script>([\s\S]*?)<\/script>/)[1];
  // Exclude only animation setup, which is independent of the telemetry controller.
  const controller = script.slice(script.indexOf('async function j('));
  const htmlWrites = [];
  const document = {activeElement: null};
  class Element {
    constructor(tag = 'div') { this.tagName = tag; this.children = []; this.style = {}; this.attrs = {}; this.className = ''; this.value = ''; }
    set textContent(value) { this.value = String(value); this.children = []; }
    get textContent() { return this.value + this.children.map(c => c.textContent).join(''); }
    set innerHTML(value) {
      htmlWrites.push(String(value)); this.value = ''; this.children = [];
      // Minimal fixture parser for the baseline's trusted div/span/i/b markup.
      // This is not a browser or an HTML-sanitizer security proof.
      const stack = [this];
      for (const token of String(value).match(/<[^>]+>|[^<]+/g) || []) {
        if (token.startsWith('</')) { if (stack.length > 1) stack.pop(); }
        else if (token.startsWith('<')) {
          const tag = token.match(/^<([a-z0-9]+)/i)?.[1];
          if (!tag) continue;
          const child = new Element(tag);
          child.className = token.match(/class="([^"]*)"/)?.[1] || '';
          stack.at(-1).appendChild(child);
          if (!['img', 'br', 'input'].includes(tag)) stack.push(child);
        } else {
          const text = new Element('#text'); text.textContent = token;
          stack.at(-1).appendChild(text);
        }
      }
    }
    get innerHTML() { return this.value; }
    appendChild(node) { this.children.push(node); return node; }
    replaceChildren(...nodes) {
      if (this.children.includes(document.activeElement)) document.activeElement = null;
      this.value = ''; this.children = [...nodes];
    }
    focus() { document.activeElement = this; }
    setAttribute(key, value) { this.attrs[key] = String(value); }
    getAttribute(key) { return this.attrs[key]; }
    querySelector(selector) {
      const all = this.children.flatMap(child => [child, ...child.descendants()]);
      return all.find(child => selector.startsWith('.') ? child.className.split(' ').includes(selector.slice(1)) : child.tagName === selector) || null;
    }
    descendants() { return this.children.flatMap(child => [child, ...child.descendants()]); }
  }
  const nodes = Object.fromEntries(['cycle','chain','prev','hcount','livepill','gallery','finds'].map(id => [id, new Element()]));
  const calls = [], intervals = [], timers = new Map();
  let nextTimer = 0;
  const h = {
    nodes, calls, intervals, timers, htmlWrites, document,
    history: {service: role.name, receipts: [receipt(role)]},
    status: 200, failure: false,
    pending: options.pendingHistory || null,
  };
  const context = {
    document: Object.assign(document, {getElementById: id => nodes[id], createElement: tag => new Element(tag)}),
    AbortController,
    pulseOrb() {},
    setInterval(fn, delay) { intervals.push({fn, delay}); return intervals.length; },
    setTimeout(fn, delay) { const id = ++nextTimer; timers.set(id, {fn, delay}); return id; },
    clearTimeout(id) { timers.delete(id); },
    fetch(url, request = {}) {
      calls.push({url, request});
      if (url === '/api/live') {
        if (options.pendingLive) return options.pendingLive.promise;
        return Promise.resolve({ok: true, json: async () => options.live || ({cycle: 1, chain_state: 'CONTINUOUS', prev_receipt_sha256: 'a'.repeat(64), history_count: 1})});
      }
      assert.equal(url, '/api/history', 'all requests must stay within the two read-only telemetry routes');
      if (h.pending) {
        request.signal?.addEventListener('abort', () => h.pending.reject(new Error('aborted')), {once: true});
        return h.pending.promise;
      }
      if (h.failure) return Promise.reject(new Error('offline'));
      return Promise.resolve({ok: h.status >= 200 && h.status < 300, json: async () => {
        if (h.jsonFailure) throw new SyntaxError('invalid JSON');
        if (h.pendingBody) {
          request.signal?.addEventListener('abort', () => h.pendingBody.reject(new Error('aborted')), {once: true});
          return h.pendingBody.promise;
        }
        return h.history;
      }});
    },
  };
  vm.runInNewContext(controller, context, {filename: role.name + '/static/index.html', timeout: 1000});
  h.tick = async () => { await Promise.all(intervals.map(({fn}) => fn())); await flush(); };
  h.historyCalls = () => calls.filter(c => c.url === '/api/history');
  h.card = () => nodes.gallery.children.find(n => n.className === 'snap');
  h.assertCleared = () => {
    assert.equal(h.card(), undefined);
    assert.match(nodes.gallery.textContent, /history unavailable/);
    assert.equal(nodes.finds.textContent, '');
    assert.equal(nodes.hcount.textContent, '—');
  };
  return h;
}

for (const role of roles) {
  test(`${role.name}: reported live chain state is inert text with unchanged classification`, async () => {
    const marker = '<img src=x onerror="alert(1)">';
    const h = harness(role, {live: {cycle: 1, chain_state: marker}}); await flush();
    assert.equal(h.nodes.chain.textContent, marker);
    assert.equal(h.nodes.chain.children[0].className, 'warn');
    assert.equal(h.htmlWrites.some(text => text.includes(marker)), false);
    const continuous = harness(role); await flush();
    assert.equal(continuous.nodes.chain.textContent, 'CONTINUOUS');
    assert.equal(continuous.nodes.chain.children[0].className, 'ok');
  });
  test(`${role.name}: initial history preserves existing short proof display`, async () => {
    const h = harness(role); await flush();
    const card = h.card(); assert.ok(card);
    assert.equal(card.querySelector('.sha').textContent, 'sha ' + 'a'.repeat(24) + '…');
    assert.equal(card.querySelector('.sig').textContent, 'sig ' + (role.sig ? 'b'.repeat(32) + '…' : '—'));
  });
  test(`${role.name}: initial server fields never enter the HTML parser`, async () => {
    const h = harness(role);
    const marker = '<img src=x onerror="alert(1)">';
    h.history.receipts[0].findings = [{check: marker, verdict: marker, value: marker}];
    await flush();
    assert.ok(h.card().querySelector('.finds').textContent.includes(marker));
    assert.equal(h.htmlWrites.some(text => text.includes(marker)), false);
  });
  test(`${role.name}: expansion and collapse preserve its exact proof fields`, async () => {
    const h = harness(role); await flush();
    const card = h.card(); assert.ok(card);
    assert.equal(card.getAttribute('role'), 'button');
    assert.equal(card.tabIndex, 0);
    assert.equal(card.querySelector('.sha').textContent, 'sha ' + 'a'.repeat(24) + '…');
    card.onclick();
    assert.equal(card.querySelector('.sha').textContent, 'sha ' + 'a'.repeat(64));
    assert.equal(card.querySelector('.sig').textContent, 'sig ' + (role.sig ? 'b'.repeat(role.name === 'szl-dream-architect' ? 128 : 64) : '—'));
    assert.equal(card.getAttribute('aria-expanded'), 'true');
    card.onclick();
    assert.equal(card.querySelector('.sha').textContent, 'sha ' + 'a'.repeat(24) + '…');
    assert.equal(card.getAttribute('aria-expanded'), 'false');
  });
  test(`${role.name}: keyboard expansion is usable without a pointer`, async () => {
    const h = harness(role); await flush();
    const card = h.card(); let prevented = 0;
    card.onkeydown({key: 'Enter', preventDefault() { prevented++; }});
    assert.equal(card.getAttribute('aria-expanded'), 'true');
    card.onkeydown({key: ' ', preventDefault() { prevented++; }});
    assert.equal(card.getAttribute('aria-expanded'), 'false');
    card.onkeydown({key: 'Tab', preventDefault() { prevented++; }});
    assert.equal(prevented, 2);
  });
  test(`${role.name}: receipt fields and structured findings are inert text`, async () => {
    const h = harness(role); await flush();
    const marker = '<img src=x onerror="alert(1)">';
    const row = receipt(role);
    Object.assign(row, {cycle: marker, chain_state: marker, [role.timestamp]: marker, [role.sha]: marker,
      findings: [{check: marker, verdict: marker, value: marker}, {hypotheses: [{hypothesis: marker}]}]});
    if (role.sig) row[role.sig] = marker;
    h.history.receipts = [row];
    await h.tick();
    const card = h.card(); assert.ok(card);
    assert.ok(card.textContent.includes(marker));
    assert.ok(h.nodes.finds.textContent.includes(marker));
    assert.ok(card.querySelector('.finds').textContent.includes('hypotheses'));
    assert.equal(h.htmlWrites.some(text => text.includes(marker)), false);
    assert.equal(card.querySelector('img'), null);
    card.onclick();
    assert.equal(card.querySelector('.sha').textContent, 'sha ' + marker);
  });
  test(`${role.name}: polling refreshes history and displayed count`, async () => {
    const h = harness(role); await flush();
    h.history.receipts = [receipt(role, 2), receipt(role)];
    await h.tick();
    assert.equal(h.historyCalls().length, 2);
    assert.equal(h.nodes.hcount.textContent, '2');
    assert.equal(h.card().querySelector('.cyc').textContent, 'CYCLE 2');
    assert.ok(h.intervals.every(t => t.delay === 60000));
    for (const {request} of h.historyCalls()) {
      assert.equal(request.cache, 'no-store');
      assert.equal(request.redirect, 'error');
      assert.equal(request.method, undefined); // fetch defaults to GET
      assert.equal(request.body, undefined);
      assert.ok(request.signal instanceof AbortSignal);
    }
  });
  test(`${role.name}: expanded cards keep keyboard focus across unchanged and new history`, async () => {
    const h = harness(role); await flush();
    const card = h.card(); card.onclick(); card.focus();
    await h.tick();
    assert.equal(h.card(), card);
    assert.equal(card.getAttribute('aria-expanded'), 'true');
    assert.equal(h.document.activeElement, card);
    h.history.receipts = [receipt(role, 2), h.history.receipts[0]];
    await h.tick();
    assert.equal(h.nodes.gallery.children[1], card);
    assert.equal(card.getAttribute('aria-expanded'), 'true');
    assert.equal(h.document.activeElement, card);
  });
  test(`${role.name}: invalid JSON clears prior proof and recovers`, async () => {
    const h = harness(role); await flush();
    h.jsonFailure = true; await h.tick(); h.assertCleared();
    h.jsonFailure = false; await h.tick(); assert.ok(h.card());
  });
  test(`${role.name}: timeout during response-body reading clears proof`, async () => {
    const h = harness(role); await flush();
    h.pendingBody = deferred();
    const tick = h.tick(); await flush();
    const timer = [...h.timers.values()].find(t => t.delay === 10000); assert.ok(timer);
    timer.fn(); await tick; h.assertCleared();
    h.pendingBody = null; await h.tick(); assert.ok(h.card());
  });
  test(`${role.name}: empty history clears previous findings without a fabricated cadence`, async () => {
    const h = harness(role); await flush();
    h.history.receipts = []; await h.tick();
    assert.equal(h.card(), undefined);
    assert.equal(h.nodes.finds.textContent, '');
    assert.equal(h.nodes.hcount.textContent, '0');
    assert.match(h.nodes.gallery.textContent, /no receipts returned/);
    assert.doesNotMatch(h.nodes.gallery.textContent, /hourly cycle/);
  });
  test(`${role.name}: network failure removes stale proof and next refresh recovers`, async () => {
    const h = harness(role); await flush();
    h.failure = true; await h.tick(); h.assertCleared();
    h.failure = false; h.history.receipts = [receipt(role, 3)]; await h.tick();
    assert.equal(h.card().querySelector('.cyc').textContent, 'CYCLE 3');
    assert.equal(h.nodes.hcount.textContent, '1');
  });
  test(`${role.name}: HTTP failure clears history instead of trusting its body`, async () => {
    const h = harness(role); await flush();
    h.status = 503; await h.tick(); h.assertCleared();
  });
  test(`${role.name}: wrong service and malformed receipt collections fail closed`, async () => {
    const h = harness(role); await flush();
    for (const history of [null, {}, {service: 'wrong', receipts: []},
      {service: role.name, receipts: {}}, {service: role.name, receipts: [null]},
      {service: role.name, receipts: [receipt(role), {findings: [null]}]},
      {service: role.name, receipts: [{findings: {}}]},
      {service: role.name, receipts: Array.from({length: 25}, () => receipt(role))}]) {
      h.history = history; await h.tick(); h.assertCleared();
    }
  });
  test(`${role.name}: pending history requests cannot overlap`, async () => {
    const pending = deferred(); const h = harness(role, {pendingHistory: pending}); await flush();
    await h.tick(); await h.tick();
    assert.equal(h.historyCalls().length, 1);
    pending.resolve({ok: true, json: async () => h.history}); await flush();
    h.pending = null; await h.tick();
    assert.equal(h.historyCalls().length, 2);
    assert.ok(h.card());
  });
  test(`${role.name}: request timeout clears proof and permits a later retry`, async () => {
    const h = harness(role); await flush();
    h.pending = deferred();
    const tick = h.tick(); await flush();
    const timer = [...h.timers.values()].find(t => t.delay === 10000); assert.ok(timer);
    timer.fn(); await tick; h.assertCleared();
    assert.equal(h.timers.size, 0);
    h.pending = null; await h.tick(); assert.ok(h.card());
  });
  test(`${role.name}: hung live telemetry cannot prevent independent history loading`, async () => {
    const h = harness(role, {pendingLive: deferred()}); await flush();
    assert.ok(h.card());
    assert.equal(h.historyCalls().length, 1);
  });
  test(`${role.name}: missing optional evidence is shown absent, never invented`, async () => {
    const h = harness(role); await flush();
    h.history.receipts = [{cycle: 7}]; await h.tick();
    const card = h.card(); assert.ok(card); card.onclick();
    assert.equal(card.querySelector('.sha').textContent, 'sha —');
    assert.equal(card.querySelector('.sig').textContent, 'sig —');
    assert.equal(card.querySelector('.ts').textContent, '— · —');
    assert.doesNotMatch(card.textContent, /undefined|null/);
  });
}

const gardener = roles.find(role => role.name === 'estate-gardener');
const dreamer = roles.find(role => role.name === 'szl-dream-architect');
function lines(node) {
  return node.querySelector('.finds').children.map(child => child.textContent);
}
test('gardener: current upstream scan/proposal summaries are preserved, including zero', async () => {
  const h = harness(gardener); await flush();
  h.history.receipts = [{...receipt(gardener), datasets_scanned: 17, proposals_made: 0}];
  await h.tick();
  assert.deepEqual(lines(h.card()), ['SCAN17 datasets scanned', 'PROP0 proposals (append-only)']);
  assert.match(h.nodes.finds.textContent, /17 datasets scanned/);
  h.card().onclick();
  assert.deepEqual(lines(h.card()), ['SCAN17 datasets scanned', 'PROP0 proposals (append-only)']);
});
test('gardener: absent proposal count is not converted into zero or null proposals', async () => {
  const h = harness(gardener); await flush();
  h.history.receipts = [{...receipt(gardener), datasets_scanned: null, proposals_made: null}];
  await h.tick();
  assert.deepEqual(lines(h.card()), ['SCAN—', 'PROP—']);
  assert.equal(h.card().querySelector('.sig').textContent, 'sig —');
});
test('gardener: reported summary fields cannot inject HTML', async () => {
  const h = harness(gardener); await flush();
  const marker = '<img src=x onerror="alert(1)">';
  h.history.receipts = [{...receipt(gardener), datasets_scanned: marker, proposals_made: marker}];
  await h.tick();
  assert.ok(h.card().textContent.includes(marker));
  assert.equal(h.htmlWrites.some(text => text.includes(marker)), false);
});
function dreamReceipt() {
  return {...receipt(dreamer), findings: [{
    state: 'MEASURED', questions_read: 2, research_signal: 'UNAVAILABLE',
    note: 'A reviewed idea is not a model update. Dreams are proposals; gates stay human.',
    hypotheses: [1, 2].map(n => ({
      question: {n, title: 'Question ' + n},
      state: 'HYPOTHESIS — not a claim, not a model update, not an autonomous act',
      hypothesis: 'Hypothesis ' + n + ': ' + 'long explanation '.repeat(12),
      nearest_prior_art: 'Prior art ' + n,
      test: 'Experiment ' + n,
      rejection_criterion: 'Reject if gate ' + n + ' fails',
    })),
  }]};
}
test('dreamer: existing question count and short first-hypothesis summary are preserved', async () => {
  const h = harness(dreamer); await flush();
  h.history.receipts = [dreamReceipt()]; await h.tick();
  assert.match(h.card().textContent, /2 frontier questions read/);
  assert.match(h.card().textContent, /HYPHypothesis 1:/);
  assert.doesNotMatch(h.card().textContent, /Hypothesis 2:|Prior art 1|Experiment 1/);
  const item = h.card().querySelector('.finds').children[1];
  assert.equal(item.children[1].textContent.length, 71);
});
test('dreamer: expansion exposes all full hypotheses, sources, tests and rejection criteria', async () => {
  const h = harness(dreamer); await flush();
  const row = dreamReceipt(); h.history.receipts = [row]; await h.tick();
  const card = h.card(); card.onclick();
  for (const hypothesis of row.findings[0].hypotheses) {
    for (const value of [hypothesis.question.title, hypothesis.hypothesis, hypothesis.nearest_prior_art,
      hypothesis.test, hypothesis.rejection_criterion]) assert.ok(card.textContent.includes(value));
  }
  assert.equal(card.querySelector('.sig').textContent, 'sig ' + 'b'.repeat(128));
  card.onclick(); assert.doesNotMatch(card.textContent, /Hypothesis 2:|Prior art 1/);
});
test('dreamer: nested hypothesis fields are text-only in collapsed and expanded views', async () => {
  const h = harness(dreamer); await flush();
  const marker = '<svg onload="alert(1)">';
  const row = dreamReceipt();
  Object.assign(row.findings[0].hypotheses[0], {question: {title: marker}, hypothesis: marker,
    nearest_prior_art: marker, test: marker, rejection_criterion: marker});
  h.history.receipts = [row]; await h.tick();
  assert.ok(h.card().textContent.includes(marker)); h.card().onclick();
  assert.ok(h.card().textContent.includes('REJECT' + marker));
  assert.equal(h.htmlWrites.some(text => text.includes(marker)), false);
});
test('dreamer: missing hypothesis details are absent, never invented', async () => {
  const h = harness(dreamer); await flush();
  h.history.receipts = [{...receipt(dreamer), findings: [{hypotheses: [{hypothesis: 'Only supplied text'}]}]}];
  await h.tick(); const card = h.card(); card.onclick();
  assert.ok(card.textContent.includes('CAVEAT—QUESTION—PRIOR—TEST—REJECT—'));
  assert.doesNotMatch(card.textContent, /undefined|null/);
});

test('dreamer: expansion preserves source limitations and hypothesis caveats', async () => {
  const h = harness(dreamer); await flush();
  const row = dreamReceipt(); h.history.receipts = [row]; await h.tick();
  const card = h.card(); card.onclick();
  assert.ok(card.textContent.includes('RESEARCHUNAVAILABLE'));
  assert.ok(card.textContent.includes('NOTE' + row.findings[0].note));
  for (const hypothesis of row.findings[0].hypotheses) {
    assert.ok(card.textContent.includes('QUESTION#' + hypothesis.question.n + ' ' + hypothesis.question.title));
    assert.ok(card.textContent.includes('CAVEAT' + hypothesis.state));
  }
});
test('dreamer: research limitations and caveats cannot inject HTML', async () => {
  const h = harness(dreamer); await flush();
  const marker = '<img src=x onerror="alert(1)">';
  const row = dreamReceipt();
  Object.assign(row.findings[0], {research_signal: marker, note: marker});
  Object.assign(row.findings[0].hypotheses[0], {state: marker, question: {n: marker, title: marker}});
  h.history.receipts = [row]; await h.tick(); h.card().onclick();
  assert.ok(h.card().textContent.includes('RESEARCH' + marker));
  assert.ok(h.card().textContent.includes('NOTE' + marker));
  assert.ok(h.card().textContent.includes('CAVEAT' + marker));
  assert.equal(h.htmlWrites.some(text => text.includes(marker)), false);
});

test('dreamer: collapsed summary retains unavailable research and hypothesis caveat', async () => {
  const h = harness(dreamer); await flush();
  const row = dreamReceipt(); h.history.receipts = [row]; await h.tick();
  assert.ok(h.card().textContent.includes('RESEARCHUNAVAILABLE'));
  assert.ok(h.card().textContent.includes('CAVEAT' + row.findings[0].hypotheses[0].state));
  assert.ok(h.nodes.finds.textContent.includes('RESEARCHUNAVAILABLE'));
});

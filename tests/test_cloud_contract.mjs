import assert from 'node:assert/strict';
import test from 'node:test';
import { PAYLOAD_TYPES, canonicalize, pae, validateSpec } from '../cloud/sign-job.mjs';

function v2() {
  return {
    jobId: 'job-2026-frontier-sft',
    kind: 'unsloth-frontier-sft-v2',
    createdAt: '2026-07-21T00:00:00Z',
    expiresAt: '2026-07-22T00:00:00Z',
    base: { repoId: 'unsloth/Qwen3', revision: 'a'.repeat(40), licenseId: 'apache-2.0' },
    dataset: {
      repoId: 'SZLHOLDINGS/data', revision: 'b'.repeat(40), file: 'train.jsonl',
      sha256: 'c'.repeat(64), format: 'messages-jsonl', licenseId: 'apache-2.0',
      provenance: 'Each row is linked to a content-addressed and licensed source.',
    },
    recipe: {
      packing: false, assistantOnlyLoss: false, useRsLoRA: true,
      expectedChatTemplateSha256: 'd'.repeat(64),
    },
    gates: {},
    outputs: {
      modelRepoId: 'SZLHOLDINGS/model', receiptsRepoId: 'SZLHOLDINGS/receipts', private: true,
      exports: { adapter: true, merged16bit: true, ggufQuantizations: ['q4_k_m'], requireReloadSmoke: true },
    },
    eval: {
      suite: 'frontier-heldout-v2', convictionCeiling: 0.97, maxDegenerateRate: 0,
      minJsonValidRate: 0.95, minRequiredKeysRate: 0.95, minCeilingRespectRate: 1,
    },
  };
}

test('v2 signer maps to the v2 payload type', () => {
  assert.equal(validateSpec(v2()), PAYLOAD_TYPES['unsloth-frontier-sft-v2']);
});

test('floating revisions and unvalidated exports are refused', () => {
  const floating = v2();
  floating.base.revision = 'main';
  assert.throws(() => validateSpec(floating), /immutable/);
  const noSmoke = v2();
  noSmoke.outputs.exports.requireReloadSmoke = false;
  assert.throws(() => validateSpec(noSmoke), /requireReloadSmoke/);
});

test('v2 requires pinned license and chat-template evidence', () => {
  const noLicense = v2();
  delete noLicense.dataset.licenseId;
  assert.throws(() => validateSpec(noLicense), /dataset\.licenseId/);
  const noTemplate = v2();
  delete noTemplate.recipe.expectedChatTemplateSha256;
  assert.throws(() => validateSpec(noTemplate), /expectedChatTemplateSha256/);
});

test('v2 timestamps require the shared RFC 3339 grammar', () => {
  const invalid = [
    '2026-07-21T00:00:00',
    '2026-07-21Z',
    'Tue, 21 Jul 2026 00:00:00 Z',
    '2026-07-21T24:00:00Z',
    '2026-02-30T00:00:00Z',
    '2026-07-21X00:00:00+00:00',
    '2026-07-21T00:00:00,5+00:00',
    '2026-07-21T00:00:00+00:60',
    '2026-07-21T00:00:00+05',
    '2026-07-21T00:00:00+0500',
    '2026-07-21T00:00:00+05:00:30',
  ];
  for (const value of invalid) {
    const spec = v2();
    spec.createdAt = value;
    assert.throws(() => validateSpec(spec), /createdAt.*RFC 3339/);
  }

  const offset = v2();
  offset.createdAt = '2026-07-21T00:00:00.123456-05:00';
  offset.expiresAt = '2026-07-22T00:00:00+02:00';
  assert.equal(validateSpec(offset), PAYLOAD_TYPES['unsloth-frontier-sft-v2']);

  const microseconds = v2();
  microseconds.createdAt = '2026-07-21T00:00:00.000001Z';
  microseconds.expiresAt = '2026-07-21T00:00:00.000002Z';
  assert.equal(validateSpec(microseconds), PAYLOAD_TYPES['unsloth-frontier-sft-v2']);
  [microseconds.createdAt, microseconds.expiresAt] = [microseconds.expiresAt, microseconds.createdAt];
  assert.throws(() => validateSpec(microseconds), /after createdAt/);
});

test('canonical JSON and PAE are deterministic', () => {
  const body = Buffer.from(canonicalize({ z: 1, a: ['x', true] }));
  assert.equal(body.toString(), '{"a":["x",true],"z":1}');
  assert.equal(pae('type', body).toString(), 'DSSEv1 4 type 22 {"a":["x",true],"z":1}');
});

#!/usr/bin/env node
/**
 * cloud/sign-job.mjs — DSSE-sign a job spec into queue/pending/.
 *
 * The envelope (not the bare spec) is what gets committed; the laptop
 * verifies signature + keyId against its baked-in pin BEFORE parsing the
 * spec. Signing refuses specs that fail basic contract checks — a spec
 * we would not execute must never enter the queue.
 *
 * Usage: SZL_QUANT_KEY=…/engine_key.pem node cloud/sign-job.mjs <spec.json>
 */
import { createHash, sign as edSign, verify as edVerify, createPrivateKey, createPublicKey } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname, basename } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const PAYLOAD_TYPE = 'application/vnd.szl.gpu-bridge.jobspec.v1+json';

/** RFC 8785-style canonical JSON (sorted keys, no whitespace) — matches szl-quant. */
function canonicalize(v) {
  if (v === null || typeof v === 'number' || typeof v === 'boolean') return JSON.stringify(v);
  if (typeof v === 'string') return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(canonicalize).join(',')}]`;
  if (typeof v === 'object') {
    const keys = Object.keys(v).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalize(v[k])}`).join(',')}}`;
  }
  throw new Error(`non-serializable value of type ${typeof v}`);
}

function pae(payloadType, payloadBytes) {
  const pt = Buffer.from(payloadType, 'utf8');
  return Buffer.concat([
    Buffer.from('DSSEv1 ', 'utf8'),
    Buffer.from(String(pt.length), 'utf8'), Buffer.from(' ', 'utf8'), pt,
    Buffer.from(' ', 'utf8'),
    Buffer.from(String(payloadBytes.length), 'utf8'), Buffer.from(' ', 'utf8'), payloadBytes,
  ]);
}

function fail(msg) { console.error(`REFUSED: ${msg}`); process.exit(1); }

const specPath = process.argv[2];
if (!specPath) fail('usage: sign-job.mjs <spec.json>');
const spec = JSON.parse(readFileSync(specPath, 'utf8'));

// contract checks (fail closed — mirror of laptop-side checks)
if (!/^job-[0-9]{4}-[a-z0-9-]+$/.test(spec.jobId ?? '')) fail('bad jobId');
if (spec.kind !== 'unsloth-qlora-sft-v1') fail('unknown kind');
for (const k of ['base', 'dataset', 'recipe', 'gates', 'outputs', 'eval', 'expiresAt', 'createdAt']) {
  if (!(k in spec)) fail(`missing ${k}`);
}
if (!/^[0-9a-f]{64}$/.test(spec.dataset.sha256 ?? '')) fail('dataset.sha256 must be 64-hex');
if (!spec.base.revision || spec.base.revision === 'main') fail('base.revision must be a pinned commit sha');
if (!spec.dataset.revision || spec.dataset.revision === 'main') fail('dataset.revision must be a pinned commit sha');
if (new Date(spec.expiresAt).getTime() <= new Date(spec.createdAt).getTime()) fail('expiresAt must be after createdAt');

const keyPath = process.env.SZL_QUANT_KEY;
if (!keyPath) fail('SZL_QUANT_KEY not set — refusing to enqueue an unsigned spec');
const privateKey = createPrivateKey(readFileSync(keyPath));
const pubJson = JSON.parse(readFileSync(join(ROOT, 'keys/engine_pubkey.json'), 'utf8'));
const publicKey = createPublicKey({ key: Buffer.from(pubJson.publicKeySpkiBase64, 'base64'), format: 'der', type: 'spki' });

// keyId must match the repo pin — a mismatched key must never sign the queue
const spki = publicKey.export({ type: 'spki', format: 'der' });
const keyId = createHash('sha256').update(spki).digest('hex').slice(0, 16);
if (keyId !== pubJson.keyId) fail(`derived keyId ${keyId} ≠ pinned ${pubJson.keyId}`);
// and the private key must actually correspond to the pin
const probe = Buffer.from('szl-gpu-bridge-keycheck');
const sigProbe = edSign(null, probe, privateKey);
if (!edVerify(null, probe, publicKey, sigProbe)) fail('private key does not match pinned pubkey');

const payloadBytes = Buffer.from(canonicalize(spec), 'utf8');
const sig = edSign(null, pae(PAYLOAD_TYPE, payloadBytes), privateKey);

const envelope = {
  payloadType: PAYLOAD_TYPE,
  payload: payloadBytes.toString('base64'),
  signatures: [{ keyid: keyId, sig: sig.toString('base64') }],
  publicKeySpkiBase64: pubJson.publicKeySpkiBase64,
};

const out = join(ROOT, 'queue/pending', `${spec.jobId}.json`);
writeFileSync(out, JSON.stringify(envelope, null, 2) + '\n');
console.log(`signed → ${out}`);
console.log(`keyId ${keyId}  payload sha256 ${createHash('sha256').update(payloadBytes).digest('hex')}`);

#!/usr/bin/env node
/**
 * DSSE-sign a v1 or v2 job spec into queue/pending/.
 *
 * The signer validates security-critical contract fields before signing. The
 * laptop independently repeats verification and validation before dispatch.
 *
 * Usage: SZL_QUANT_KEY=…/engine_key.pem node cloud/sign-job.mjs <spec.json>
 */
import {
  createHash,
  sign as edSign,
  verify as edVerify,
  createPrivateKey,
  createPublicKey,
} from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
export const PAYLOAD_TYPES = Object.freeze({
  'unsloth-qlora-sft-v1': 'application/vnd.szl.gpu-bridge.jobspec.v1+json',
  'unsloth-frontier-sft-v2': 'application/vnd.szl.gpu-bridge.jobspec.v2+json',
});
const REPO_ID = /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/;
const REVISION = /^[0-9a-f]{40,64}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const JOB_ID = /^job-[0-9]{4}-[a-z0-9][a-z0-9-]{2,80}$/;
const QUANTS = new Set(['q4_k_m', 'q5_k_m', 'q6_k', 'q8_0', 'f16']);

export function canonicalize(value) {
  if (value === null || typeof value === 'number' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'string') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalize).join(',')}]`;
  if (typeof value === 'object') {
    const keys = Object.keys(value).sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalize(value[key])}`).join(',')}}`;
  }
  throw new Error(`non-serializable value of type ${typeof value}`);
}

export function pae(payloadType, payloadBytes) {
  const typeBytes = Buffer.from(payloadType, 'utf8');
  return Buffer.concat([
    Buffer.from('DSSEv1 ', 'utf8'),
    Buffer.from(String(typeBytes.length), 'utf8'), Buffer.from(' ', 'utf8'), typeBytes,
    Buffer.from(' ', 'utf8'),
    Buffer.from(String(payloadBytes.length), 'utf8'), Buffer.from(' ', 'utf8'), payloadBytes,
  ]);
}

function requireObject(value, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${name} must be an object`);
  return value;
}

function requireRepoId(value, name) {
  if (typeof value !== 'string' || !REPO_ID.test(value)) throw new Error(`${name} must be owner/name`);
}

function requireRevision(value, name) {
  if (typeof value !== 'string' || !REVISION.test(value)) throw new Error(`${name} must be an immutable lowercase hex revision`);
}

function validateCommon(spec) {
  requireObject(spec, 'spec');
  if (!JOB_ID.test(spec.jobId ?? '')) throw new Error('bad jobId');
  for (const key of ['base', 'dataset', 'recipe', 'gates', 'outputs', 'eval', 'expiresAt', 'createdAt']) {
    if (!(key in spec)) throw new Error(`missing ${key}`);
  }
  const created = new Date(spec.createdAt).getTime();
  const expires = new Date(spec.expiresAt).getTime();
  if (!Number.isFinite(created) || !Number.isFinite(expires) || expires <= created) {
    throw new Error('expiresAt must be a valid date after createdAt');
  }
  requireObject(spec.base, 'base');
  requireObject(spec.dataset, 'dataset');
  requireObject(spec.outputs, 'outputs');
  requireRevision(spec.base.revision, 'base.revision');
  requireRevision(spec.dataset.revision, 'dataset.revision');
  requireRepoId(spec.base.repoId, 'base.repoId');
  requireRepoId(spec.dataset.repoId, 'dataset.repoId');
  requireRepoId(spec.outputs.modelRepoId, 'outputs.modelRepoId');
  requireRepoId(spec.outputs.receiptsRepoId, 'outputs.receiptsRepoId');
  if (!SHA256.test(spec.dataset.sha256 ?? '')) throw new Error('dataset.sha256 must be 64 lowercase hex');
}

function validateV1(spec) {
  validateCommon(spec);
  if (spec.kind !== 'unsloth-qlora-sft-v1') throw new Error('unknown v1 kind');
}

function validateV2(spec) {
  validateCommon(spec);
  if (spec.kind !== 'unsloth-frontier-sft-v2') throw new Error('unknown v2 kind');
  if (typeof spec.base.licenseId !== 'string' || !spec.base.licenseId.trim()) throw new Error('base.licenseId required');
  if (typeof spec.dataset.licenseId !== 'string' || !spec.dataset.licenseId.trim()) throw new Error('dataset.licenseId required');
  if (spec.dataset.format !== 'messages-jsonl') throw new Error('dataset.format must be messages-jsonl');
  if (typeof spec.dataset.provenance !== 'string' || spec.dataset.provenance.trim().length < 20) {
    throw new Error('dataset.provenance must be auditable');
  }
  const recipe = requireObject(spec.recipe, 'recipe');
  for (const key of ['packing', 'assistantOnlyLoss', 'useRsLoRA']) {
    if (typeof recipe[key] !== 'boolean') throw new Error(`recipe.${key} must be boolean`);
  }
  if (!SHA256.test(recipe.expectedChatTemplateSha256 ?? '')) {
    throw new Error('recipe.expectedChatTemplateSha256 must be 64 lowercase hex');
  }
  const exports = requireObject(spec.outputs.exports, 'outputs.exports');
  if (exports.adapter !== true || typeof exports.merged16bit !== 'boolean' || exports.requireReloadSmoke !== true) {
    throw new Error('v2 exports require adapter=true and requireReloadSmoke=true');
  }
  if (!Array.isArray(exports.ggufQuantizations) || new Set(exports.ggufQuantizations).size !== exports.ggufQuantizations.length) {
    throw new Error('ggufQuantizations must be a unique array');
  }
  for (const quant of exports.ggufQuantizations) if (!QUANTS.has(quant)) throw new Error(`unsupported GGUF quantization ${quant}`);
  if (spec.outputs.checkpointBucketId !== undefined) requireRepoId(spec.outputs.checkpointBucketId, 'outputs.checkpointBucketId');
  const evaluation = requireObject(spec.eval, 'eval');
  if (evaluation.suite !== 'frontier-heldout-v2') throw new Error('eval.suite must be frontier-heldout-v2');
  for (const key of [
    'convictionCeiling', 'maxDegenerateRate', 'minJsonValidRate',
    'minRequiredKeysRate', 'minCeilingRespectRate',
  ]) {
    if (typeof evaluation[key] !== 'number' || evaluation[key] < 0 || evaluation[key] > 1) {
      throw new Error(`eval.${key} must be between 0 and 1`);
    }
  }
}

export function validateSpec(spec) {
  const payloadType = PAYLOAD_TYPES[spec?.kind];
  if (!payloadType) throw new Error(`unsupported kind ${spec?.kind ?? '(missing)'}`);
  if (spec.kind.endsWith('-v1')) validateV1(spec);
  else validateV2(spec);
  return payloadType;
}

function fail(message) {
  console.error(`REFUSED: ${message}`);
  process.exit(1);
}

export function main(argv = process.argv.slice(2), env = process.env) {
  const specPath = argv[0];
  if (!specPath) fail('usage: sign-job.mjs <spec.json>');
  let spec;
  try {
    spec = JSON.parse(readFileSync(specPath, 'utf8'));
  } catch (error) {
    fail(`spec JSON unreadable: ${error.message}`);
  }
  let payloadType;
  try {
    payloadType = validateSpec(spec);
  } catch (error) {
    fail(error.message);
  }

  const keyPath = env.SZL_QUANT_KEY;
  if (!keyPath) fail('SZL_QUANT_KEY not set — refusing to enqueue an unsigned spec');
  const privateKey = createPrivateKey(readFileSync(keyPath));
  const pubJson = JSON.parse(readFileSync(join(ROOT, 'keys/engine_pubkey.json'), 'utf8'));
  const publicKey = createPublicKey({
    key: Buffer.from(pubJson.publicKeySpkiBase64, 'base64'),
    format: 'der',
    type: 'spki',
  });

  const spki = publicKey.export({ type: 'spki', format: 'der' });
  const keyId = createHash('sha256').update(spki).digest('hex').slice(0, 16);
  if (keyId !== pubJson.keyId) fail(`derived keyId ${keyId} ≠ pinned ${pubJson.keyId}`);
  const probe = Buffer.from('szl-gpu-bridge-keycheck');
  const probeSignature = edSign(null, probe, privateKey);
  if (!edVerify(null, probe, publicKey, probeSignature)) fail('private key does not match pinned pubkey');

  const payloadBytes = Buffer.from(canonicalize(spec), 'utf8');
  const signature = edSign(null, pae(payloadType, payloadBytes), privateKey);
  const envelope = {
    payloadType,
    payload: payloadBytes.toString('base64'),
    signatures: [{ keyid: keyId, sig: signature.toString('base64') }],
    publicKeySpkiBase64: pubJson.publicKeySpkiBase64,
  };
  const output = join(ROOT, 'queue/pending', `${spec.jobId}.json`);
  writeFileSync(output, `${JSON.stringify(envelope, null, 2)}\n`);
  console.log(`signed → ${output}`);
  console.log(`keyId ${keyId}  payload sha256 ${createHash('sha256').update(payloadBytes).digest('hex')}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();

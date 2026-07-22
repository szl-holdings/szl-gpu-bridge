#!/usr/bin/env node
/** DSSE-sign one reviewed SZL-Nemo v3 job into queue/pending/. */

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
export const NEMO_V3_PAYLOAD_TYPE = 'application/vnd.szl.gpu-bridge.nemo-v3.jobspec.v1+json';
const SHA = /^[0-9a-f]{40,64}$/;
const SHA256 = /^[0-9a-f]{64}$/;
const REPO = /^[A-Za-z0-9][A-Za-z0-9_.-]*\/[A-Za-z0-9][A-Za-z0-9_.-]*$/;
const JOB = /^job-[0-9]{4}-nemo-v3-[a-z0-9][a-z0-9-]{2,64}$/;
const SAFE_PATH = /^[A-Za-z0-9][A-Za-z0-9_./-]*$/;
const HOLDOUT_NAMES = ['original-v2', 'shadow-v2', 'challenge-v3'];

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

function object(value, name) {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error(`${name} must be an object`);
  return value;
}

function idsDigest(ids) {
  return createHash('sha256').update(Buffer.from(`${ids.join('\n')}\n`, 'utf8')).digest('hex');
}

function validatePinnedFile(value, name, records = false) {
  object(value, name);
  if (typeof value.path !== 'string' || !SAFE_PATH.test(value.path) || value.path.startsWith('/') || value.path.split('/').includes('..')) {
    throw new Error(`${name}.path must be a safe relative path`);
  }
  if (!SHA256.test(value.sha256 ?? '')) throw new Error(`${name}.sha256 must be lowercase sha256`);
  if (!Number.isInteger(value.bytes) || value.bytes <= 0) throw new Error(`${name}.bytes must be positive`);
  if (records) {
    if (!Array.isArray(value.recordIds) || !value.recordIds.length || new Set(value.recordIds).size !== value.recordIds.length) {
      throw new Error(`${name}.recordIds must be a non-empty unique array`);
    }
    if (!value.recordIds.every((id) => typeof id === 'string' && id.length >= 3)) throw new Error(`${name}.recordIds invalid`);
    if (!SHA256.test(value.recordIdsSha256 ?? '') || idsDigest(value.recordIds) !== value.recordIdsSha256) {
      throw new Error(`${name}.recordIdsSha256 mismatch`);
    }
  }
}

export function validateNemoV3Spec(spec) {
  object(spec, 'spec');
  if (spec.kind !== 'szl-nemo-governed-v3') throw new Error('bad kind');
  if (!JOB.test(spec.jobId ?? '')) throw new Error('bad jobId');
  const created = new Date(spec.createdAt).getTime();
  const expires = new Date(spec.expiresAt).getTime();
  if (!Number.isFinite(created) || !Number.isFinite(expires) || expires <= created) throw new Error('invalid expiry');

  object(spec.source, 'source');
  if (spec.source.repoId !== 'szl-holdings/a11oy' || spec.source.licenseId !== 'apache-2.0' || !/^[0-9a-f]{40}$/.test(spec.source.revision ?? '')) {
    throw new Error('source must be immutable Apache-2.0 a11oy');
  }
  object(spec.base, 'base');
  if (!REPO.test(spec.base.repoId ?? '') || !SHA.test(spec.base.revision ?? '')) throw new Error('base identity invalid');
  if (typeof spec.base.licenseId !== 'string' || !spec.base.licenseId.trim()) throw new Error('base license required');
  if (typeof spec.base.licenseAcknowledgement !== 'string' || spec.base.licenseAcknowledgement.trim().length < 20) {
    throw new Error('explicit base license acknowledgement required');
  }
  if (typeof spec.base.trustRemoteCode !== 'boolean') throw new Error('base.trustRemoteCode must be boolean');

  object(spec.dataset, 'dataset');
  if (spec.dataset.rightsBasis !== 'PROJECT_AUTHORED_SCENARIOS') throw new Error('dataset rights basis not admitted');
  if (typeof spec.dataset.provenance !== 'string' || spec.dataset.provenance.trim().length < 40) throw new Error('dataset provenance required');
  validatePinnedFile(spec.dataset.train, 'dataset.train');
  validatePinnedFile(spec.dataset.preregistration, 'dataset.preregistration');
  if (!Array.isArray(spec.dataset.holdouts) || spec.dataset.holdouts.length !== 3) throw new Error('exactly three holdouts required');
  const names = [];
  const allIds = [];
  spec.dataset.holdouts.forEach((item, index) => {
    validatePinnedFile(item, `dataset.holdouts[${index}]`, true);
    names.push(item.name);
    allIds.push(...item.recordIds);
  });
  if (JSON.stringify(names) !== JSON.stringify(HOLDOUT_NAMES)) throw new Error('holdout order mismatch');
  if (new Set(allIds).size !== allIds.length) throw new Error('holdout record ids overlap');

  object(spec.recipe, 'recipe');
  if (spec.recipe.batchSize !== 1) throw new Error('batchSize must be 1');
  if (!Number.isInteger(spec.recipe.maxSeqLength) || spec.recipe.maxSeqLength < 256 || spec.recipe.maxSeqLength > 4096) throw new Error('maxSeqLength invalid');
  if (!Number.isInteger(spec.recipe.loraR) || spec.recipe.loraR < 1 || spec.recipe.loraR > 64) throw new Error('loraR invalid');
  if (!Array.isArray(spec.recipe.targetModules) || !spec.recipe.targetModules.length) throw new Error('targetModules required');
  if (!['adamw_8bit', 'paged_adamw_8bit'].includes(spec.recipe.optimizer)) throw new Error('optimizer unsupported');
  if (!['unsloth', 'true'].includes(spec.recipe.gradientCheckpointing)) throw new Error('gradient checkpointing required');

  object(spec.gates, 'gates');
  if (Number(spec.gates.minFreeVramGb) < 5 || Number(spec.gates.minFreeDiskGb) < 20) throw new Error('resource gates weakened');
  if (!Number.isInteger(spec.gates.maxTemperatureC) || spec.gates.maxTemperatureC > 80) throw new Error('temperature gate invalid');
  if (!Number.isInteger(spec.gates.maxUtilizationPct) || spec.gates.maxUtilizationPct > 30) throw new Error('utilization gate invalid');

  object(spec.outputs, 'outputs');
  if (typeof spec.outputs.candidateId !== 'string' || !spec.outputs.candidateId.startsWith('SZL-Nemo-v3-')) throw new Error('candidateId invalid');
  if (!REPO.test(spec.outputs.receiptsRepoId ?? '')) throw new Error('receiptsRepoId invalid');
  if (spec.outputs.private !== true || spec.outputs.publishCandidate !== false) throw new Error('candidate publication must remain disabled');

  object(spec.evaluation, 'evaluation');
  if (spec.evaluation.requiredPassRate !== 1 || spec.evaluation.maxDegenerateRate !== 0 || spec.evaluation.requireExactRecordOrder !== true) {
    throw new Error('all exact holdouts must pass with zero degeneration');
  }
  if (!Number.isInteger(spec.evaluation.maxNewTokens) || spec.evaluation.maxNewTokens < 32 || spec.evaluation.maxNewTokens > 512) {
    throw new Error('maxNewTokens invalid');
  }
  return NEMO_V3_PAYLOAD_TYPE;
}

function fail(message) {
  console.error(`REFUSED: ${message}`);
  process.exit(1);
}

export function main(argv = process.argv.slice(2), env = process.env) {
  const specPath = argv[0];
  if (!specPath) fail('usage: sign-nemo-v3-job.mjs <spec.json>');
  let spec;
  try {
    spec = JSON.parse(readFileSync(specPath, 'utf8'));
    validateNemoV3Spec(spec);
  } catch (error) {
    fail(error.message);
  }
  const keyPath = env.SZL_QUANT_KEY;
  if (!keyPath) fail('SZL_QUANT_KEY not set — refusing unsigned Nemo v3 job');
  const privateKey = createPrivateKey(readFileSync(keyPath));
  const pubJson = JSON.parse(readFileSync(join(ROOT, 'keys/engine_pubkey.json'), 'utf8'));
  const publicKey = createPublicKey({
    key: Buffer.from(pubJson.publicKeySpkiBase64, 'base64'), format: 'der', type: 'spki',
  });
  const spki = publicKey.export({ type: 'spki', format: 'der' });
  const keyId = createHash('sha256').update(spki).digest('hex').slice(0, 16);
  if (keyId !== pubJson.keyId) fail(`derived keyId ${keyId} differs from pin ${pubJson.keyId}`);
  const probe = Buffer.from('szl-gpu-bridge-keycheck');
  if (!edVerify(null, probe, publicKey, edSign(null, probe, privateKey))) fail('private key does not match pinned pubkey');

  const payloadBytes = Buffer.from(canonicalize(spec), 'utf8');
  const signature = edSign(null, pae(NEMO_V3_PAYLOAD_TYPE, payloadBytes), privateKey);
  const envelope = {
    payloadType: NEMO_V3_PAYLOAD_TYPE,
    payload: payloadBytes.toString('base64'),
    signatures: [{ keyid: keyId, sig: signature.toString('base64') }],
    publicKeySpkiBase64: pubJson.publicKeySpkiBase64,
  };
  const output = join(ROOT, 'queue/pending', `${spec.jobId}.json`);
  writeFileSync(output, `${JSON.stringify(envelope, null, 2)}\n`);
  console.log(`signed Nemo v3 job -> ${output}`);
  console.log(`keyId ${keyId} payload sha256 ${createHash('sha256').update(payloadBytes).digest('hex')}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();

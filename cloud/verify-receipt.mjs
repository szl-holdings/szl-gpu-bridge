#!/usr/bin/env node
/** Independently verify v1/v2 laptop receipts and optional receipt chains. */
import { createHash, createPublicKey, verify as edVerify } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export function canonicalize(value) {
  if (value === null || typeof value === 'number' || typeof value === 'boolean') return JSON.stringify(value);
  if (typeof value === 'string') return JSON.stringify(value);
  if (Array.isArray(value)) return `[${value.map(canonicalize).join(',')}]`;
  if (typeof value === 'object') {
    const keys = Object.keys(value).sort();
    return `{${keys.map((key) => `${JSON.stringify(key)}:${canonicalize(value[key])}`).join(',')}}`;
  }
  throw new Error(`non-serializable ${typeof value}`);
}

export function verifyOne(path, expectedKeyId = null) {
  const signed = JSON.parse(readFileSync(path, 'utf8'));
  const spki = Buffer.from(signed.publicKeySpkiBase64, 'base64');
  const keyId = createHash('sha256').update(spki).digest('hex').slice(0, 16);
  if (keyId !== signed.keyId) throw new Error(`${path}: embedded keyId ${signed.keyId} ≠ derived ${keyId}`);
  if (expectedKeyId && keyId !== expectedKeyId) throw new Error(`${path}: keyId ${keyId} ≠ expected pin ${expectedKeyId}`);
  if (signed.scheme !== 'ed25519-over-exact-bytes-v2' || !signed.bodyBase64) {
    throw new Error(`${path}: unsupported exact-byte signing scheme`);
  }
  const publicKey = createPublicKey({ key: spki, format: 'der', type: 'spki' });
  const body = Buffer.from(signed.bodyBase64, 'base64');
  if (!edVerify(null, body, publicKey, Buffer.from(signed.signatureBase64, 'base64'))) {
    throw new Error(`${path}: SIGNATURE INVALID`);
  }
  const receipt = JSON.parse(body.toString('utf8'));
  if (canonicalize(receipt) !== canonicalize(signed.receipt)) {
    throw new Error(`${path}: display receipt diverges from signed bytes`);
  }
  const bodySha = createHash('sha256').update(body).digest('hex');
  console.log(`PASS  ${path}  kind=${receipt.kind}  keyId=${keyId}  bodySha256=${bodySha.slice(0, 16)}…`);
  return { signed, receipt, bodySha };
}

export function verifyChain(training, evaluation) {
  const supported = {
    'szl-bridge-eval-receipt': {
      trainingKind: 'szl-bridge-training-receipt',
      field: 'trainingReceiptSha256',
    },
    'szl-frontier-eval-receipt': {
      trainingKind: 'szl-frontier-training-receipt',
      field: 'trainingReceiptBodySha256',
    },
  };
  const contract = supported[evaluation.receipt.kind];
  if (!contract) throw new Error(`unsupported eval receipt kind ${evaluation.receipt.kind}`);
  if (training.receipt.kind !== contract.trainingKind) {
    throw new Error(`training receipt kind ${training.receipt.kind} does not match ${contract.trainingKind}`);
  }
  if (evaluation.receipt[contract.field] !== training.bodySha) {
    throw new Error(`CHAIN BROKEN: ${contract.field} does not pin the exact training receipt bytes`);
  }
  console.log('CHAIN OK  eval receipt pins this exact training receipt');
}

export function main(argv = process.argv.slice(2)) {
  const args = [...argv];
  const pinIndex = args.indexOf('--expect-keyid');
  const expected = pinIndex >= 0 ? args.splice(pinIndex, 2)[1] : null;
  if (args.length === 0 || args.length > 2) {
    console.error('usage: verify-receipt.mjs <receipt.json> [<eval.json>] [--expect-keyid <16hex>]');
    process.exit(1);
  }
  try {
    const first = verifyOne(args[0], expected);
    if (args[1]) verifyChain(first, verifyOne(args[1], expected));
    console.log('\nAll receipts verified. Claims remain attestations, not proof-of-computation.');
  } catch (error) {
    console.error(`FAIL (fail closed): ${error.message}`);
    process.exit(1);
  }
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();

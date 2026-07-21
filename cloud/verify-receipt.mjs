#!/usr/bin/env node
/**
 * cloud/verify-receipt.mjs — independently verify a laptop receipt.
 *
 * A receipt claim the cloud cannot verify is treated as NO claim.
 * Checks: ed25519 signature over canonical JSON, keyId consistency with
 * the embedded SPKI, optional pin against an announced laptop keyId, and
 * the eval→training chain when both receipts are given.
 *
 * Usage:
 *   node cloud/verify-receipt.mjs <receipt.signed.json> [--expect-keyid <16hex>]
 *   node cloud/verify-receipt.mjs <training.signed.json> <eval.signed.json> [--expect-keyid <16hex>]
 */
import { createHash, createPublicKey, verify as edVerify } from 'node:crypto';
import { readFileSync } from 'node:fs';

function canonicalize(v) {
  if (v === null || typeof v === 'number' || typeof v === 'boolean') return JSON.stringify(v);
  if (typeof v === 'string') return JSON.stringify(v);
  if (Array.isArray(v)) return `[${v.map(canonicalize).join(',')}]`;
  if (typeof v === 'object') {
    const keys = Object.keys(v).sort();
    return `{${keys.map((k) => `${JSON.stringify(k)}:${canonicalize(v[k])}`).join(',')}}`;
  }
  throw new Error(`non-serializable ${typeof v}`);
}

function verifyOne(path, expectKeyId) {
  const signed = JSON.parse(readFileSync(path, 'utf8'));
  const spki = Buffer.from(signed.publicKeySpkiBase64, 'base64');
  const keyId = createHash('sha256').update(spki).digest('hex').slice(0, 16);
  if (keyId !== signed.keyId) throw new Error(`${path}: embedded keyId ${signed.keyId} ≠ derived ${keyId}`);
  if (expectKeyId && keyId !== expectKeyId) throw new Error(`${path}: keyId ${keyId} ≠ expected pin ${expectKeyId}`);
  const pub = createPublicKey({ key: spki, format: 'der', type: 'spki' });

  // Verify over the EXACT bytes the laptop signed (bodyBase64) — never a
  // re-serialization: Python/JS float formatting differs (2.0 vs 2), and a
  // re-canonicalizing verifier would reject honest receipts intermittently.
  if (signed.scheme !== 'ed25519-over-exact-bytes-v2' || !signed.bodyBase64) {
    throw new Error(`${path}: unsupported scheme ${signed.scheme ?? '(none)'} — expected ed25519-over-exact-bytes-v2 with bodyBase64`);
  }
  const body = Buffer.from(signed.bodyBase64, 'base64');
  const ok = edVerify(null, body, pub, Buffer.from(signed.signatureBase64, 'base64'));
  if (!ok) throw new Error(`${path}: SIGNATURE INVALID`);

  // the verified truth is the signed bytes; the display copy must match them
  const receipt = JSON.parse(body.toString('utf8'));
  if (canonicalize(receipt) !== canonicalize(signed.receipt)) {
    throw new Error(`${path}: display copy 'receipt' diverges from signed bodyBase64 — tampered display`);
  }
  const bodySha = createHash('sha256').update(body).digest('hex');
  console.log(`PASS  ${path}  kind=${receipt.kind}  keyId=${keyId}  bodySha256=${bodySha.slice(0, 16)}…`);
  return { signed, receipt, bodySha };
}

const args = process.argv.slice(2);
const pinIdx = args.indexOf('--expect-keyid');
const expect = pinIdx >= 0 ? args.splice(pinIdx, 2)[1] : null;
if (args.length === 0) { console.error('usage: verify-receipt.mjs <receipt.json> [<eval.json>] [--expect-keyid <16hex>]'); process.exit(1); }

try {
  const first = verifyOne(args[0], expect);
  if (args[1]) {
    const second = verifyOne(args[1], expect);
    const evalR = second.receipt;
    if (evalR.kind !== 'szl-bridge-eval-receipt') throw new Error('second file is not an eval receipt');
    if (evalR.trainingReceiptSha256 !== first.bodySha) {
      throw new Error(`CHAIN BROKEN: eval.trainingReceiptSha256 ${evalR.trainingReceiptSha256.slice(0, 16)}… ≠ training receipt body sha ${first.bodySha.slice(0, 16)}…`);
    }
    console.log('CHAIN OK  eval receipt pins this exact training receipt');
  }
  console.log('\nAll receipt(s) verified. Claims inside are REPORTED-from-receipt (attestation, not proof-of-computation).');
} catch (e) {
  console.error(`FAIL (fail closed): ${e.message}`);
  process.exit(1);
}

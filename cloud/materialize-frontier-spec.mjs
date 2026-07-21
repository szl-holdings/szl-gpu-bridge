#!/usr/bin/env node
/**
 * Resolve a human-authored frontier job draft into an immutable unsigned v2 spec.
 *
 * This tool resolves exact Hub commits, reads model/dataset license metadata,
 * pins the tokenizer chat-template hash, and hashes the exact dataset file. The
 * result still must be signed by cloud/sign-job.mjs before the laptop will act.
 *
 * Usage:
 *   HF_TOKEN=... node cloud/materialize-frontier-spec.mjs draft.json spec.json
 */
import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { validateSpec } from './sign-job.mjs';

const HUB = 'https://huggingface.co';
const SHA = /^[0-9a-f]{40,64}$/;

function repoPath(repoId) {
  if (typeof repoId !== 'string' || repoId.split('/').length !== 2) {
    throw new Error(`invalid Hub repo id ${repoId ?? '(missing)'}`);
  }
  return repoId.split('/').map(encodeURIComponent).join('/');
}

function authHeaders(token = process.env.HF_TOKEN || process.env.HUGGING_FACE_HUB_TOKEN) {
  return token ? { Authorization: `Bearer ${token}` } : {};
}

async function fetchChecked(url, options = {}) {
  const response = await fetch(url, {
    redirect: 'follow',
    ...options,
    headers: {
      'User-Agent': 'szl-gpu-bridge-materializer/2',
      ...authHeaders(),
      ...(options.headers || {}),
    },
  });
  if (!response.ok) {
    const body = (await response.text()).slice(0, 500);
    throw new Error(`HTTP ${response.status} for ${url}: ${body}`);
  }
  return response;
}

export function normalizeLicenses(value) {
  const items = Array.isArray(value) ? value : [value];
  return [...new Set(items
    .filter((item) => typeof item === 'string' && item.trim())
    .map((item) => item.trim().toLowerCase()))].sort();
}

export function selectLicense(observed, requested, label) {
  const licenses = normalizeLicenses(observed);
  if (!licenses.length) throw new Error(`${label} card has no license metadata`);
  if (requested) {
    const normalized = String(requested).trim().toLowerCase();
    if (!licenses.includes(normalized)) {
      throw new Error(`${label} requested license ${normalized} not in observed ${licenses.join(', ')}`);
    }
    return normalized;
  }
  if (licenses.length !== 1) {
    throw new Error(`${label} has multiple licenses (${licenses.join(', ')}); select licenseId explicitly`);
  }
  return licenses[0];
}

export function extractChatTemplate(config, fallback = null) {
  const value = config?.chat_template;
  if (typeof value === 'string' && value) return value;
  if (value && typeof value === 'object' && typeof value.default === 'string' && value.default) {
    return value.default;
  }
  if (typeof fallback === 'string' && fallback) return fallback;
  throw new Error('pinned tokenizer has no unambiguous chat template');
}

async function repoInfo(repoId, repoType, revision = 'main') {
  const prefix = repoType === 'dataset' ? 'datasets' : 'models';
  const suffix = revision ? `/revision/${encodeURIComponent(revision)}` : '';
  const response = await fetchChecked(`${HUB}/api/${prefix}/${repoPath(repoId)}${suffix}`);
  const info = await response.json();
  if (!SHA.test(info.sha || '')) throw new Error(`${repoType} ${repoId} did not resolve to an immutable sha`);
  return info;
}

async function fetchOptionalText(url) {
  const response = await fetch(url, {
    redirect: 'follow',
    headers: { 'User-Agent': 'szl-gpu-bridge-materializer/2', ...authHeaders() },
  });
  if (response.status === 404) return null;
  if (!response.ok) throw new Error(`HTTP ${response.status} for ${url}: ${(await response.text()).slice(0, 500)}`);
  return response.text();
}

async function tokenizerTemplate(repoId, revision) {
  const base = `${HUB}/${repoPath(repoId)}/resolve/${revision}`;
  let config = {};
  const configText = await fetchOptionalText(`${base}/tokenizer_config.json`);
  if (configText) config = JSON.parse(configText);
  let fallback = null;
  if (typeof config.chat_template !== 'string' && !(config.chat_template?.default)) {
    fallback = await fetchOptionalText(`${base}/chat_template.jinja`);
  }
  const template = extractChatTemplate(config, fallback);
  return {
    sha256: createHash('sha256').update(Buffer.from(template, 'utf8')).digest('hex'),
    bytes: Buffer.byteLength(template),
  };
}

async function digestRemoteFile(url) {
  const response = await fetchChecked(url);
  const hash = createHash('sha256');
  let bytes = 0;
  for await (const chunk of response.body) {
    const value = Buffer.from(chunk);
    bytes += value.length;
    hash.update(value);
  }
  return { sha256: hash.digest('hex'), bytes };
}

function isoNow() {
  return new Date().toISOString();
}

export async function materialize(draft) {
  const baseInfo = await repoInfo(draft.base?.repoId, 'model', draft.base?.sourceRevision || 'main');
  const datasetInfo = await repoInfo(
    draft.dataset?.repoId,
    'dataset',
    draft.dataset?.sourceRevision || 'main',
  );
  const baseCard = baseInfo.cardData || baseInfo.card_data || {};
  const datasetCard = datasetInfo.cardData || datasetInfo.card_data || {};
  const template = await tokenizerTemplate(draft.base.repoId, baseInfo.sha);
  const datasetUrl = `${HUB}/datasets/${repoPath(draft.dataset.repoId)}/resolve/${datasetInfo.sha}/${draft.dataset.file.split('/').map(encodeURIComponent).join('/')}`;
  const datasetDigest = await digestRemoteFile(datasetUrl);
  const createdAt = draft.createdAt || isoNow();
  const expiresAt = draft.expiresAt || new Date(
    new Date(createdAt).getTime() + Number(draft.ttlHours || 12) * 60 * 60 * 1000,
  ).toISOString();
  const spec = {
    jobId: draft.jobId,
    kind: 'unsloth-frontier-sft-v2',
    createdAt,
    expiresAt,
    base: {
      repoId: draft.base.repoId,
      revision: baseInfo.sha,
      licenseId: selectLicense(baseCard.license, draft.base.licenseId, 'base model'),
      trustRemoteCode: Boolean(draft.base.trustRemoteCode),
    },
    dataset: {
      repoId: draft.dataset.repoId,
      revision: datasetInfo.sha,
      file: draft.dataset.file,
      sha256: datasetDigest.sha256,
      provenance: draft.dataset.provenance,
      format: 'messages-jsonl',
      licenseId: selectLicense(datasetCard.license, draft.dataset.licenseId, 'dataset'),
    },
    recipe: {
      ...draft.recipe,
      expectedChatTemplateSha256: template.sha256,
    },
    gates: draft.gates,
    outputs: draft.outputs,
    eval: draft.eval,
    ...(draft.notes ? { notes: draft.notes } : {}),
  };
  validateSpec(spec);
  return {
    spec,
    evidence: {
      resolvedAt: isoNow(),
      base: { sha: baseInfo.sha, license: spec.base.licenseId, chatTemplate: template },
      dataset: { sha: datasetInfo.sha, license: spec.dataset.licenseId, file: datasetDigest },
    },
  };
}

async function main(argv = process.argv.slice(2)) {
  const [draftPath, outputPath] = argv;
  if (!draftPath || !outputPath) {
    throw new Error('usage: materialize-frontier-spec.mjs <draft.json> <spec.json>');
  }
  const draft = JSON.parse(readFileSync(draftPath, 'utf8'));
  const result = await materialize(draft);
  writeFileSync(outputPath, `${JSON.stringify(result.spec, null, 2)}\n`);
  writeFileSync(`${outputPath}.evidence.json`, `${JSON.stringify(result.evidence, null, 2)}\n`);
  console.log(`materialized immutable v2 spec -> ${outputPath}`);
  console.log(`base ${result.spec.base.revision} dataset ${result.spec.dataset.revision}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  main().catch((error) => {
    console.error(`REFUSED: ${error.message}`);
    process.exit(1);
  });
}

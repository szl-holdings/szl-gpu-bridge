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
const ENGINE_KEY_ID = /^[0-9a-f]{16}$/;
const SAFE_PATH = /^[A-Za-z0-9][A-Za-z0-9_./-]*$/;
const HOLDOUT_NAMES = ['original-v2', 'shadow-v2', 'challenge-v3'];
const OWNER_DISPATCH_FIELDS = [
  'workflowIdentity', 'workflowBlob', 'workflowVersion', 'trainingImage',
  'candidateUpload', 'modelCardUpload', 'datasetUpload', 'receiptsRepoId',
];
const OWNER_WORKFLOW_IDENTITY = 'szl-holdings/a11oy/.github/workflows/nemo-v3-isolated-owner-dispatch.yml@refs/heads/main';
const OWNER_WORKFLOW_VERSION = 'nemo-v3-owner-dispatch.v2';
const OWNER_TRAINING_IMAGE = 'unsloth/unsloth@sha256:9cc97606fc386b4b13455285eb7bd2668f51530988a9c2578707fe6cdfc46123';
const OWNER_RECEIPTS_REPO = 'SZLHOLDINGS/szl-training-receipts';
const LEGACY_ENGINE_KEY_ID = '5c6cf59741ade920';
const PROVISIONAL_ENGINE_KEY_ID = '815714c8d4ae3e4d';
const COORDINATED_ENGINE_KEY_ID = 'b8041281c81c4caa';
const COORDINATED_ENGINE_SPKI_SHA256 = 'b8041281c81c4caaea18112df5e8c99ea8472f0711fc796fc3072c27398af2cf';
const SETTLED_A11OY_SOURCE_REVISION = '5f98d90a42e021cf29948457a2404a159f236487';
const SETTLED_OWNER_WORKFLOW_BLOB = '7e08ffc8aa87b78d0fa1618d7d3c3e68cb81ca33';
const SETTLED_A11OY_RELOCK_RUN_URL = 'https://github.com/szl-holdings/a11oy/actions/runs/30561614589';
const CORRECTED_BRIDGE_REVISION = '2237bb3f36663343ace29d98cda6c32e165450a0';
const NEXT_REVIEWED_JOB_ID = 'job-2026-nemo-v3-governed-attempt-4';
const FINAL_A11OY_SOURCE_REVISION = 'e3d4a46724b222c8a5b2b6f04877bc115a6c82cb';
const FINAL_OWNER_WORKFLOW_BLOB = '2522d3b54eeb7adc37ffc47e7c685a5ce7edf68f';
const FINAL_A11OY_RELOCK_RUN_URL = 'https://github.com/szl-holdings/a11oy/actions/runs/30588489971';
const FINAL_CORRECTED_BRIDGE_REVISION = 'a2015accc0be8060c4084455e829a9373e5c99e2';
const ATTEMPT_5_REVIEWED_JOB_ID = 'job-2026-nemo-v3-governed-attempt-5';
const EXECUTION_A11OY_SOURCE_REVISION = '78b35d244b89c7663063372ff459894bab2977b6';
const EXECUTION_OWNER_WORKFLOW_BLOB = 'd29d937b2d398e9c207777a9a819aadd050ac231';
const EXECUTION_A11OY_RELOCK_RUN_URL = 'https://github.com/szl-holdings/a11oy/actions/runs/30592401025';
const EXECUTION_CORRECTED_BRIDGE_REVISION = '69a097d2eb0619506d673464353f1aea7174cf05';
const ATTEMPT_6_REVIEWED_JOB_ID = 'job-2026-nemo-v3-governed-attempt-6';
const SUCCESSOR_A11OY_SOURCE_REVISION = '2b190b3806a5d2b3faa58f34c2db41c5dc4668fa';
const SUCCESSOR_OWNER_WORKFLOW_BLOB = 'd29d937b2d398e9c207777a9a819aadd050ac231';
const SUCCESSOR_A11OY_RELOCK_RUN_URL = 'https://github.com/szl-holdings/a11oy/actions/runs/30601635066';
const SUCCESSOR_CORRECTED_BRIDGE_REVISION = '2f33607d8fcbec76fe98290258ec3dfa728fb509';
const FUTURE_REVIEWED_JOB_ID = 'job-2026-nemo-v3-governed-attempt-7';
const NEXT_RUNTIME_REVIEWED_JOB_ID = 'job-2026-nemo-v3-governed-attempt-8';
const NEXT_RUNTIME_CORRECTED_BRIDGE_REVISION = 'dc36af2b264bbdb4cc101593c54c5b2c24c1d9cf';
const QUARANTINED_JOB_IDS = new Set([
  'job-2026-nemo-v3-governed-attempt-2',
  'job-2026-nemo-v3-governed-successor-3',
  'job-2026-nemo-v3-governed-attempt-4',
  'job-2026-nemo-v3-governed-attempt-5',
  'job-2026-nemo-v3-governed-attempt-6',
  'job-2026-nemo-v3-governed-attempt-7',
]);
const FINAL_OWNER_WORKFLOW_VERSION = 'nemo-v3-owner-dispatch.v4';
const COORDINATED_JOB_BINDINGS = {
  [NEXT_REVIEWED_JOB_ID]: {
    sourceRevision: SETTLED_A11OY_SOURCE_REVISION,
    workflowBlob: SETTLED_OWNER_WORKFLOW_BLOB,
    workflowVersion: OWNER_WORKFLOW_VERSION,
    relockRunUrl: SETTLED_A11OY_RELOCK_RUN_URL,
    correctedBridgeRevision: CORRECTED_BRIDGE_REVISION,
    successorGeneration: 4,
  },
  [ATTEMPT_5_REVIEWED_JOB_ID]: {
    sourceRevision: FINAL_A11OY_SOURCE_REVISION,
    workflowBlob: FINAL_OWNER_WORKFLOW_BLOB,
    workflowVersion: FINAL_OWNER_WORKFLOW_VERSION,
    relockRunUrl: FINAL_A11OY_RELOCK_RUN_URL,
    correctedBridgeRevision: FINAL_CORRECTED_BRIDGE_REVISION,
    successorGeneration: 5,
  },
  [ATTEMPT_6_REVIEWED_JOB_ID]: {
    sourceRevision: EXECUTION_A11OY_SOURCE_REVISION,
    workflowBlob: EXECUTION_OWNER_WORKFLOW_BLOB,
    workflowVersion: FINAL_OWNER_WORKFLOW_VERSION,
    relockRunUrl: EXECUTION_A11OY_RELOCK_RUN_URL,
    correctedBridgeRevision: EXECUTION_CORRECTED_BRIDGE_REVISION,
    successorGeneration: 6,
  },
  [FUTURE_REVIEWED_JOB_ID]: {
    sourceRevision: SUCCESSOR_A11OY_SOURCE_REVISION,
    workflowBlob: SUCCESSOR_OWNER_WORKFLOW_BLOB,
    workflowVersion: FINAL_OWNER_WORKFLOW_VERSION,
    relockRunUrl: SUCCESSOR_A11OY_RELOCK_RUN_URL,
    correctedBridgeRevision: SUCCESSOR_CORRECTED_BRIDGE_REVISION,
    successorGeneration: 7,
  },
  [NEXT_RUNTIME_REVIEWED_JOB_ID]: {
    sourceRevision: SUCCESSOR_A11OY_SOURCE_REVISION,
    workflowBlob: SUCCESSOR_OWNER_WORKFLOW_BLOB,
    workflowVersion: FINAL_OWNER_WORKFLOW_VERSION,
    relockRunUrl: SUCCESSOR_A11OY_RELOCK_RUN_URL,
    correctedBridgeRevision: NEXT_RUNTIME_CORRECTED_BRIDGE_REVISION,
    successorGeneration: 8,
  },
};

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

function validateLineage(spec) {
  if (spec.lineage === undefined) return;
  const lineage = object(spec.lineage, 'lineage');
  const legacyFields = [
    'predecessorJobId', 'predecessorClaimSha256', 'predecessorEnvelopeSha256',
    'predecessorBridgeRevision', 'predecessorImageId', 'predecessorClaimedAt',
    'incidentUrl', 'failurePhase', 'successorGeneration', 'automaticRetry',
    'trainingStarted', 'modelRepositoryCodeImported', 'holdoutsAccessed',
    'candidateProduced', 'receiptIntentProduced', 'terminalLedgerWritten',
    'scienceInputsReused',
  ];
  const transportFields = [
    'predecessorJobId', 'predecessorEnvelopeSha256', 'predecessorPayloadSha256',
    'predecessorEnvelopeRevision', 'predecessorExecutionBridgeRevision',
    'transportEvidenceUrl', 'failurePhase', 'successorGeneration',
    'automaticRetry', 'eventCreated', 'workflowRunCreated', 'claimCreated',
    'trainingStarted', 'modelRepositoryCodeImported', 'holdoutsAccessed',
    'candidateProduced', 'receiptIntentProduced', 'terminalLedgerWritten',
    'scienceInputsReused',
  ];
  const keys = Object.keys(lineage).sort();
  const legacy = JSON.stringify(keys) === JSON.stringify([...legacyFields].sort());
  const transport = JSON.stringify(keys) === JSON.stringify([...transportFields].sort());
  if (!legacy && !transport) {
    throw new Error('lineage fields must be exact');
  }
  if (!JOB.test(lineage.predecessorJobId ?? '') || lineage.predecessorJobId === spec.jobId) {
    throw new Error('lineage predecessor jobId invalid');
  }
  if (!SHA256.test(lineage.predecessorEnvelopeSha256 ?? '')) {
    throw new Error('lineage predecessor envelope digest invalid');
  }
  let expectedEventCreated = false;
  if (transport) {
    if (!SHA256.test(lineage.predecessorPayloadSha256 ?? '')
        || !/^[0-9a-f]{40}$/.test(lineage.predecessorEnvelopeRevision ?? '')
        || !/^[0-9a-f]{40}$/.test(lineage.predecessorExecutionBridgeRevision ?? '')) {
      throw new Error('lineage predecessor transport identity invalid');
    }
    let expectedEvidence;
    let expectedFailurePhase;
    if (lineage.predecessorJobId === NEXT_REVIEWED_JOB_ID) {
      expectedEvidence = 'https://github.com/szl-holdings/szl-gpu-bridge/issues/32';
      expectedFailurePhase = 'PRE_EVENT_TRANSPORT_VALIDATION';
      expectedEventCreated = false;
    } else if (lineage.predecessorJobId === ATTEMPT_5_REVIEWED_JOB_ID) {
      expectedEvidence = 'https://github.com/szl-holdings/a11oy/actions/runs/30591897165';
      expectedFailurePhase = 'PRE_ADMISSION_HOST_EXECUTION_POLICY';
      expectedEventCreated = true;
    } else if (lineage.predecessorJobId === ATTEMPT_6_REVIEWED_JOB_ID) {
      expectedEvidence = 'https://github.com/szl-holdings/szl-gpu-bridge/issues/41';
      expectedFailurePhase = 'PRE_DISPATCH_VALIDATOR_REJECTION';
      expectedEventCreated = false;
    } else if (lineage.predecessorJobId === FUTURE_REVIEWED_JOB_ID) {
      expectedEvidence = 'https://github.com/szl-holdings/a11oy/actions/runs/30605081533';
      expectedFailurePhase = 'PRE_CLAIM_RUNTIME_CONTRACT_VALIDATION';
      expectedEventCreated = true;
    } else {
      throw new Error('lineage predecessor transport recovery is not admitted');
    }
    if (lineage.transportEvidenceUrl !== expectedEvidence
        || lineage.failurePhase !== expectedFailurePhase) {
      throw new Error('lineage predecessor transport evidence invalid');
    }
  } else {
    if (!SHA256.test(lineage.predecessorClaimSha256 ?? '')
        || !/^[0-9a-f]{40}$/.test(lineage.predecessorBridgeRevision ?? '')
        || !/^sha256:[0-9a-f]{64}$/.test(lineage.predecessorImageId ?? '')) {
      throw new Error('lineage predecessor execution identity invalid');
    }
    if (!Number.isFinite(new Date(lineage.predecessorClaimedAt).getTime())
        || !/^https:\/\/github\.com\/szl-holdings\/szl-gpu-bridge\/issues\/[0-9]+#issuecomment-[0-9]+$/.test(lineage.incidentUrl ?? '')) {
      throw new Error('lineage predecessor evidence invalid');
    }
    if (lineage.failurePhase !== 'PRE_TRAINING_RUNTIME_SOURCE_PARSE') {
      throw new Error('lineage recovery phase invalid');
    }
  }
  if (!Number.isInteger(lineage.successorGeneration) || lineage.successorGeneration < 2) {
    throw new Error('lineage recovery phase invalid');
  }
  const boundaries = {
    automaticRetry: false,
    trainingStarted: false,
    modelRepositoryCodeImported: false,
    holdoutsAccessed: false,
    candidateProduced: false,
    receiptIntentProduced: false,
    terminalLedgerWritten: false,
    scienceInputsReused: true,
  };
  if (transport) {
    Object.assign(boundaries, {
      eventCreated: expectedEventCreated,
      workflowRunCreated: expectedEventCreated,
      claimCreated: false,
    });
  }
  for (const [field, expected] of Object.entries(boundaries)) {
    if (lineage[field] !== expected) throw new Error(`lineage ${field} boundary invalid`);
  }
}

function validateOwnerDispatch(spec, coordinatedBinding) {
  if (spec.ownerDispatch === undefined) return;
  const dispatch = object(spec.ownerDispatch, 'ownerDispatch');
  const keys = Object.keys(dispatch).sort();
  if (JSON.stringify(keys) !== JSON.stringify([...OWNER_DISPATCH_FIELDS].sort())) {
    throw new Error('ownerDispatch fields must be exact');
  }
  const workflowVersion = coordinatedBinding?.workflowVersion ?? OWNER_WORKFLOW_VERSION;
  if (dispatch.workflowIdentity !== OWNER_WORKFLOW_IDENTITY
      || dispatch.workflowVersion !== workflowVersion) {
    throw new Error('ownerDispatch workflow identity is not admitted');
  }
  if (!/^[0-9a-f]{40}$/.test(dispatch.workflowBlob ?? '')) {
    throw new Error('ownerDispatch workflowBlob must be an exact git blob');
  }
  if (dispatch.trainingImage !== OWNER_TRAINING_IMAGE) {
    throw new Error('ownerDispatch training image is not admitted');
  }
  for (const field of ['candidateUpload', 'modelCardUpload', 'datasetUpload']) {
    if (dispatch[field] !== false) throw new Error(`ownerDispatch ${field} must remain false`);
  }
  if (dispatch.receiptsRepoId !== OWNER_RECEIPTS_REPO) {
    throw new Error('ownerDispatch receipts repository is not admitted');
  }
}

function validateAuthorization(spec, coordinatedBinding) {
  if (spec.authorization === undefined) return false;
  const authorization = object(spec.authorization, 'authorization');
  const legacyFields = [
    'engineKeyId', 'previousEngineKeyId', 'recoveryIssueUrl',
    'rotationMode', 'oldKeyStatus', 'decisionAt',
  ];
  const coordinated = authorization.rotationMode === 'COORDINATED_FINAL_TRUST_ROOT_NEW_GENERATION';
  const fields = coordinated
    ? [
      ...legacyFields,
      'enginePublicKeySpkiSha256', 'provisionalEngineKeyId',
      'provisionalKeyStatus', 'coordinationMode',
      'settledA11oyRelockRunUrl', 'cryptographicContinuityClaimed',
      'correctedBridgeRevision',
    ]
    : legacyFields;
  if (JSON.stringify(Object.keys(authorization).sort()) !== JSON.stringify([...fields].sort())) {
    throw new Error('authorization fields must be exact');
  }
  if (!ENGINE_KEY_ID.test(authorization.engineKeyId ?? '')
      || !ENGINE_KEY_ID.test(authorization.previousEngineKeyId ?? '')
      || authorization.engineKeyId === authorization.previousEngineKeyId) {
    throw new Error('authorization engine key identities are invalid');
  }
  if (authorization.previousEngineKeyId !== LEGACY_ENGINE_KEY_ID
      || !['LOST_PRIVATE_KEY_NEW_GENERATION', 'COORDINATED_FINAL_TRUST_ROOT_NEW_GENERATION'].includes(authorization.rotationMode)
      || authorization.oldKeyStatus !== 'VERIFY_ONLY') {
    throw new Error('authorization recovery boundary is invalid');
  }
  if (authorization.recoveryIssueUrl !== 'https://github.com/szl-holdings/szl-gpu-bridge/issues/25'
      || !Number.isFinite(new Date(authorization.decisionAt).getTime())) {
    throw new Error('authorization recovery evidence is invalid');
  }
  if (coordinated) {
    if (!coordinatedBinding
        || authorization.engineKeyId !== COORDINATED_ENGINE_KEY_ID
        || authorization.enginePublicKeySpkiSha256 !== COORDINATED_ENGINE_SPKI_SHA256
        || authorization.provisionalEngineKeyId !== PROVISIONAL_ENGINE_KEY_ID
        || authorization.provisionalKeyStatus !== 'VERIFY_ONLY'
        || authorization.coordinationMode !== 'FINAL_ACTIVE_TRUST_ROOT'
        || authorization.settledA11oyRelockRunUrl !== coordinatedBinding.relockRunUrl
        || authorization.cryptographicContinuityClaimed !== false
        || authorization.correctedBridgeRevision !== coordinatedBinding.correctedBridgeRevision) {
      throw new Error('coordinated authorization boundary is invalid');
    }
  }
  return coordinated;
}

function loadEnginePin(expectedKeyId) {
  const keyring = JSON.parse(readFileSync(join(ROOT, 'keys/engine_keyring.json'), 'utf8'));
  if (keyring.kind !== 'szl-quant-engine-keyring' || keyring.v !== 1
      || !keyring.keys || typeof keyring.keys !== 'object' || Array.isArray(keyring.keys)) {
    throw new Error('engine keyring contract is invalid');
  }
  const entry = keyring.keys[expectedKeyId];
  if (!entry || entry.status !== 'ACTIVE'
      || !/^engine_pubkey(?:_[0-9a-f]{16})?\.json$/.test(entry.file ?? '')) {
    throw new Error(`engine key ${expectedKeyId} is not active for new authorization`);
  }
  const pin = JSON.parse(readFileSync(join(ROOT, 'keys', entry.file), 'utf8'));
  if (pin.keyId !== expectedKeyId) {
    throw new Error(`engine keyring entry ${expectedKeyId} differs from its pin`);
  }
  return pin;
}

export function validateNemoV3Spec(spec) {
  object(spec, 'spec');
  if (spec.kind !== 'szl-nemo-governed-v3') throw new Error('bad kind');
  if (!JOB.test(spec.jobId ?? '')) throw new Error('bad jobId');
  const created = new Date(spec.createdAt).getTime();
  const expires = new Date(spec.expiresAt).getTime();
  if (!Number.isFinite(created) || !Number.isFinite(expires) || expires <= created) throw new Error('invalid expiry');
  const coordinatedBinding = COORDINATED_JOB_BINDINGS[spec.jobId];
  validateLineage(spec);
  validateOwnerDispatch(spec, coordinatedBinding);
  const coordinatedAuthorization = validateAuthorization(spec, coordinatedBinding);

  object(spec.source, 'source');
  if (spec.source.repoId !== 'szl-holdings/a11oy' || spec.source.licenseId !== 'apache-2.0' || !/^[0-9a-f]{40}$/.test(spec.source.revision ?? '')) {
    throw new Error('source must be immutable Apache-2.0 a11oy');
  }
  if (coordinatedAuthorization
      && (spec.source.revision !== coordinatedBinding.sourceRevision
          || spec.ownerDispatch?.workflowBlob !== coordinatedBinding.workflowBlob
          || spec.lineage?.successorGeneration !== coordinatedBinding.successorGeneration)) {
    throw new Error('coordinated recovery binding is invalid');
  }
  if (spec.jobId === ATTEMPT_5_REVIEWED_JOB_ID) {
    const exactTransportLineage = {
      predecessorJobId: NEXT_REVIEWED_JOB_ID,
      predecessorEnvelopeSha256: 'e240a176849b1f6c0d453ac55277cd7732b3a302ea9679db78d3c612501f27f2',
      predecessorPayloadSha256: '14441cf982b177c1b613e56e63eae8be3e589ae35444826b40731c32312268e5',
      predecessorEnvelopeRevision: '7045fe223703ba8fb2d710a59989f971080e7702',
      predecessorExecutionBridgeRevision: CORRECTED_BRIDGE_REVISION,
      transportEvidenceUrl: 'https://github.com/szl-holdings/szl-gpu-bridge/issues/32',
      failurePhase: 'PRE_EVENT_TRANSPORT_VALIDATION',
      successorGeneration: 5,
      automaticRetry: false,
      eventCreated: false,
      workflowRunCreated: false,
      claimCreated: false,
      trainingStarted: false,
      modelRepositoryCodeImported: false,
      holdoutsAccessed: false,
      candidateProduced: false,
      receiptIntentProduced: false,
      terminalLedgerWritten: false,
      scienceInputsReused: true,
    };
    if (canonicalize(spec.lineage) !== canonicalize(exactTransportLineage)) {
      throw new Error('attempt-5 transport recovery lineage is not exact');
    }
  }
  if (spec.jobId === ATTEMPT_6_REVIEWED_JOB_ID) {
    const exactHostPolicyLineage = {
      predecessorJobId: ATTEMPT_5_REVIEWED_JOB_ID,
      predecessorEnvelopeSha256: '30549fc522238193b4985dbf96a690518bad2ae8c399dc3ee78fb9dd7f551009',
      predecessorPayloadSha256: '374901dec6923e0c28688407e581d374827d76f7567970d8ec481b6bf140c67b',
      predecessorEnvelopeRevision: 'd127d7bcd734235fba83e786de923787ab90c51b',
      predecessorExecutionBridgeRevision: FINAL_CORRECTED_BRIDGE_REVISION,
      transportEvidenceUrl: 'https://github.com/szl-holdings/a11oy/actions/runs/30591897165',
      failurePhase: 'PRE_ADMISSION_HOST_EXECUTION_POLICY',
      successorGeneration: 6,
      automaticRetry: false,
      eventCreated: true,
      workflowRunCreated: true,
      claimCreated: false,
      trainingStarted: false,
      modelRepositoryCodeImported: false,
      holdoutsAccessed: false,
      candidateProduced: false,
      receiptIntentProduced: false,
      terminalLedgerWritten: false,
      scienceInputsReused: true,
    };
    if (canonicalize(spec.lineage) !== canonicalize(exactHostPolicyLineage)) {
      throw new Error('attempt-6 host-policy recovery lineage is not exact');
    }
  }
  if (spec.jobId === FUTURE_REVIEWED_JOB_ID) {
    const exactValidatorRejectionLineage = {
      predecessorJobId: ATTEMPT_6_REVIEWED_JOB_ID,
      predecessorEnvelopeSha256: 'c68e1ecf380d7023c27439e9988ca182ebd9b2446dc769269d4de1c48d507d70',
      predecessorPayloadSha256: 'd0fa9bd15f8e576411b643858d650470b6f1d5ddd56003cd53eda28d83dd914d',
      predecessorEnvelopeRevision: '72f9bf650b081fec0a016825f2cb7f962c52242d',
      predecessorExecutionBridgeRevision: EXECUTION_CORRECTED_BRIDGE_REVISION,
      transportEvidenceUrl: 'https://github.com/szl-holdings/szl-gpu-bridge/issues/41',
      failurePhase: 'PRE_DISPATCH_VALIDATOR_REJECTION',
      successorGeneration: 7,
      automaticRetry: false,
      eventCreated: false,
      workflowRunCreated: false,
      claimCreated: false,
      trainingStarted: false,
      modelRepositoryCodeImported: false,
      holdoutsAccessed: false,
      candidateProduced: false,
      receiptIntentProduced: false,
      terminalLedgerWritten: false,
      scienceInputsReused: true,
    };
    if (canonicalize(spec.lineage) !== canonicalize(exactValidatorRejectionLineage)) {
      throw new Error('attempt-7 validator-rejection recovery lineage is not exact');
    }
  }
  if (spec.jobId === NEXT_RUNTIME_REVIEWED_JOB_ID) {
    const exactRuntimeBindingLineage = {
      predecessorJobId: FUTURE_REVIEWED_JOB_ID,
      predecessorEnvelopeSha256: '8c1e333f797a8de634217b19cd140994a1d4f3920afebdf6f658dcc984188a96',
      predecessorPayloadSha256: '0fa239d3e14f0644d26b76c0e605ea8068b305cd4d96ea41385cad38fbdfbde7',
      predecessorEnvelopeRevision: '21553a898db76dddba3227e91518835185b55a6f',
      predecessorExecutionBridgeRevision: SUCCESSOR_CORRECTED_BRIDGE_REVISION,
      transportEvidenceUrl: 'https://github.com/szl-holdings/a11oy/actions/runs/30605081533',
      failurePhase: 'PRE_CLAIM_RUNTIME_CONTRACT_VALIDATION',
      successorGeneration: 8,
      automaticRetry: false,
      eventCreated: true,
      workflowRunCreated: true,
      claimCreated: false,
      trainingStarted: false,
      modelRepositoryCodeImported: false,
      holdoutsAccessed: false,
      candidateProduced: false,
      receiptIntentProduced: false,
      terminalLedgerWritten: false,
      scienceInputsReused: true,
    };
    if (canonicalize(spec.lineage) !== canonicalize(exactRuntimeBindingLineage)) {
      throw new Error('attempt-8 runtime-binding recovery lineage is not exact');
    }
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
  if (QUARANTINED_JOB_IDS.has(spec.jobId)) {
    fail(`job ${spec.jobId} is quarantined and marked NEVER_DISPATCH`);
  }
  if (spec.jobId !== NEXT_RUNTIME_REVIEWED_JOB_ID) {
    fail(`signer is locked to ${NEXT_RUNTIME_REVIEWED_JOB_ID}`);
  }
  const keyPath = env.SZL_QUANT_KEY;
  if (!keyPath) fail('SZL_QUANT_KEY not set — refusing unsigned Nemo v3 job');
  const privateKey = createPrivateKey(readFileSync(keyPath));
  const expectedKeyId = spec.authorization?.engineKeyId ?? LEGACY_ENGINE_KEY_ID;
  const pubJson = loadEnginePin(expectedKeyId);
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
  writeFileSync(output, `${JSON.stringify(envelope, null, 2)}\n`, { flag: 'wx' });
  console.log(`signed Nemo v3 job -> ${output}`);
  console.log(`keyId ${keyId} payload sha256 ${createHash('sha256').update(payloadBytes).digest('hex')}`);
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) main();

import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import {
  NEMO_V3_PAYLOAD_TYPE,
  canonicalize,
  pae,
  validateNemoV3Spec,
} from '../cloud/sign-nemo-v3-job.mjs';

function idsDigest(ids) {
  return createHash('sha256').update(`${ids.join('\n')}\n`).digest('hex');
}

function withIds(path, name, ids) {
  return {
    path,
    name,
    recordIds: ids,
    sha256: 'a'.repeat(64),
    bytes: 100,
    recordIdsSha256: idsDigest(ids),
  };
}

function spec() {
  return {
    jobId: 'job-2026-nemo-v3-governed',
    kind: 'szl-nemo-governed-v3',
    createdAt: '2026-07-22T00:00:00Z',
    expiresAt: '2026-07-23T00:00:00Z',
    source: { repoId: 'szl-holdings/a11oy', revision: '1'.repeat(40), licenseId: 'apache-2.0' },
    base: {
      repoId: 'nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16', revision: '2'.repeat(40),
      licenseId: 'nvidia-open-model-license', trustRemoteCode: true,
      licenseAcknowledgement: 'I accept the pinned NVIDIA upstream license and preserve attribution.',
    },
    dataset: {
      provenance: 'Every row is independently project-authored and kept separate from all frozen evaluation suites.',
      rightsBasis: 'PROJECT_AUTHORED_SCENARIOS',
      train: { path: 'model_release/szl-nemo-v3/train.jsonl', sha256: 'a'.repeat(64), bytes: 100 },
      holdouts: [
        withIds('model_release/szl-nemo-v3/holdout-original-v2.jsonl', 'original-v2', ['eval:a']),
        withIds('model_release/szl-nemo-v3/holdout-shadow-v2.jsonl', 'shadow-v2', ['shadow:a']),
        withIds('model_release/szl-nemo-v3/holdout-challenge-v3.jsonl', 'challenge-v3', ['challenge:a']),
      ],
      preregistration: { path: 'model_release/szl-nemo-v3/preregistration.json', sha256: 'a'.repeat(64), bytes: 100 },
    },
    recipe: {
      maxSeqLength: 2048, loraR: 16, loraAlpha: 32, loraDropout: 0,
      targetModules: ['q_proj', 'k_proj', 'v_proj', 'o_proj'], batchSize: 1,
      gradAccum: 8, epochs: 2, learningRate: 0.0001, optimizer: 'adamw_8bit',
      gradientCheckpointing: 'unsloth', seed: 3407, warmupRatio: 0.05,
      weightDecay: 0.01, lrSchedulerType: 'linear',
    },
    gates: {
      minFreeVramGb: 6.5, minFreeDiskGb: 50, maxWallclockMinutes: 240,
      maxDatasetRows: 500, maxTemperatureC: 78, maxUtilizationPct: 15,
    },
    outputs: {
      candidateId: 'SZL-Nemo-v3-Nemotron-4B-Adapter',
      receiptsRepoId: 'SZLHOLDINGS/szl-training-receipts', private: true, publishCandidate: false,
    },
    evaluation: {
      requiredPassRate: 1, maxDegenerateRate: 0, maxNewTokens: 192, requireExactRecordOrder: true,
    },
  };
}

function ownerDispatch() {
  return {
    workflowIdentity: 'szl-holdings/a11oy/.github/workflows/nemo-v3-isolated-owner-dispatch.yml@refs/heads/main',
    workflowBlob: '7e08ffc8aa87b78d0fa1618d7d3c3e68cb81ca33',
    workflowVersion: 'nemo-v3-owner-dispatch.v2',
    trainingImage: `unsloth/unsloth@sha256:${'9cc97606fc386b4b13455285eb7bd2668f51530988a9c2578707fe6cdfc46123'}`,
    candidateUpload: false,
    modelCardUpload: false,
    datasetUpload: false,
    receiptsRepoId: 'SZLHOLDINGS/szl-training-receipts',
  };
}

test('Nemo v3 signer selects the dedicated payload type', () => {
  assert.equal(validateNemoV3Spec(spec()), NEMO_V3_PAYLOAD_TYPE);
});

test('Nemo v3 signer refuses publication and holdout drift', () => {
  const publish = spec();
  publish.outputs.publishCandidate = true;
  assert.throws(() => validateNemoV3Spec(publish), /publication/);
  const reordered = spec();
  reordered.dataset.holdouts.reverse();
  assert.throws(() => validateNemoV3Spec(reordered), /order/);
  const threshold = spec();
  threshold.evaluation.requiredPassRate = 0.9;
  assert.throws(() => validateNemoV3Spec(threshold), /all exact holdouts/);
});

test('Nemo v3 successor requires fail-closed predecessor lineage', () => {
  const successor = spec();
  successor.jobId = 'job-2026-nemo-v3-governed-successor-2';
  successor.lineage = {
    predecessorJobId: 'job-2026-nemo-v3-governed-attempt-1',
    predecessorClaimSha256: 'a'.repeat(64),
    predecessorEnvelopeSha256: 'b'.repeat(64),
    predecessorBridgeRevision: 'c'.repeat(40),
    predecessorImageId: `sha256:${'d'.repeat(64)}`,
    predecessorClaimedAt: '2026-07-29T16:41:34.8842570+00:00',
    incidentUrl: 'https://github.com/szl-holdings/szl-gpu-bridge/issues/4#issuecomment-5120817312',
    failurePhase: 'PRE_TRAINING_RUNTIME_SOURCE_PARSE',
    successorGeneration: 2,
    automaticRetry: false,
    trainingStarted: false,
    modelRepositoryCodeImported: false,
    holdoutsAccessed: false,
    candidateProduced: false,
    receiptIntentProduced: false,
    terminalLedgerWritten: false,
    scienceInputsReused: true,
  };
  assert.equal(validateNemoV3Spec(successor), NEMO_V3_PAYLOAD_TYPE);
  successor.lineage.automaticRetry = true;
  assert.throws(() => validateNemoV3Spec(successor), /automaticRetry/);
});

test('Nemo v3 owner dispatch binds exact workflow and receipt-only effects', () => {
  const attempt = spec();
  attempt.jobId = 'job-2026-nemo-v3-governed-attempt-2';
  attempt.ownerDispatch = ownerDispatch();
  assert.equal(validateNemoV3Spec(attempt), NEMO_V3_PAYLOAD_TYPE);

  for (const field of ['candidateUpload', 'modelCardUpload', 'datasetUpload']) {
    const widened = structuredClone(attempt);
    widened.ownerDispatch[field] = true;
    assert.throws(() => validateNemoV3Spec(widened), new RegExp(field));
  }
  const mutableImage = structuredClone(attempt);
  mutableImage.ownerDispatch.trainingImage = 'unsloth/unsloth:latest';
  assert.throws(() => validateNemoV3Spec(mutableImage), /training image/);
  const extra = structuredClone(attempt);
  extra.ownerDispatch.unreviewed = false;
  assert.throws(() => validateNemoV3Spec(extra), /fields must be exact/);
});

test('Nemo v3 coordinated attempt 4 binds the corrected trust lineage', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260730-attempt-4-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);
  assert.equal(
    createHash('sha256').update(Buffer.from(canonicalize(reviewed))).digest('hex'),
    '14441cf982b177c1b613e56e63eae8be3e589ae35444826b40731c32312268e5',
  );

  const continuityClaim = structuredClone(reviewed);
  continuityClaim.authorization.cryptographicContinuityClaimed = true;
  assert.throws(
    () => validateNemoV3Spec(continuityClaim),
    /coordinated authorization/,
  );

  const staleSource = structuredClone(reviewed);
  staleSource.source.revision = 'a'.repeat(40);
  assert.throws(
    () => validateNemoV3Spec(staleSource),
    /coordinated recovery binding/,
  );

  const wrongGeneration = structuredClone(reviewed);
  wrongGeneration.lineage.successorGeneration = 3;
  assert.throws(
    () => validateNemoV3Spec(wrongGeneration),
    /coordinated recovery binding/,
  );
});

test('Nemo v3 attempt 5 binds nested-v3 transport and zero predecessor effects', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260730-attempt-5-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);

  const eventCreated = structuredClone(reviewed);
  eventCreated.lineage.eventCreated = true;
  assert.throws(
    () => validateNemoV3Spec(eventCreated),
    /eventCreated/,
  );

  const staleRuntime = structuredClone(reviewed);
  staleRuntime.authorization.correctedBridgeRevision = 'a'.repeat(40);
  assert.throws(
    () => validateNemoV3Spec(staleRuntime),
    /coordinated authorization/,
  );

  const flatTransportLineage = structuredClone(reviewed);
  flatTransportLineage.lineage.predecessorClaimSha256 = 'b'.repeat(64);
  assert.throws(
    () => validateNemoV3Spec(flatTransportLineage),
    /lineage fields must be exact/,
  );
});

test('Nemo v3 attempt 6 binds the one-run pre-admission host-policy failure', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260730-attempt-6-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);

  const missingRun = structuredClone(reviewed);
  missingRun.lineage.workflowRunCreated = false;
  assert.throws(
    () => validateNemoV3Spec(missingRun),
    /workflowRunCreated/,
  );

  const wrongEvidence = structuredClone(reviewed);
  wrongEvidence.lineage.transportEvidenceUrl = 'https://github.com/szl-holdings/szl-gpu-bridge/issues/32';
  assert.throws(
    () => validateNemoV3Spec(wrongEvidence),
    /transport evidence/,
  );

  const staleRuntime = structuredClone(reviewed);
  staleRuntime.authorization.correctedBridgeRevision = 'a'.repeat(40);
  assert.throws(
    () => validateNemoV3Spec(staleRuntime),
    /coordinated authorization/,
  );
});

test('Nemo v3 attempt 7 binds the zero-event validator rejection', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260731-attempt-7-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);

  const eventCreated = structuredClone(reviewed);
  eventCreated.lineage.eventCreated = true;
  assert.throws(
    () => validateNemoV3Spec(eventCreated),
    /eventCreated/,
  );

  const skippedPredecessor = structuredClone(reviewed);
  skippedPredecessor.lineage.predecessorJobId = 'job-2026-nemo-v3-governed-attempt-5';
  assert.throws(
    () => validateNemoV3Spec(skippedPredecessor),
    /transport evidence/,
  );

  const staleRuntime = structuredClone(reviewed);
  staleRuntime.authorization.correctedBridgeRevision = 'a'.repeat(40);
  assert.throws(
    () => validateNemoV3Spec(staleRuntime),
    /coordinated authorization/,
  );
});

test('Nemo v3 canonical JSON and PAE are deterministic', () => {
  const body = Buffer.from(canonicalize({ z: 1, a: ['x', true] }));
  assert.equal(body.toString(), '{"a":["x",true],"z":1}');
  assert.equal(pae('type', body).toString(), 'DSSEv1 4 type 22 {"a":["x",true],"z":1}');
});

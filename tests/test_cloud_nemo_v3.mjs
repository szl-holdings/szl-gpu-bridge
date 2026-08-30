import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import {
  NEMO_V3_PAYLOAD_TYPE,
  canonicalize,
  pae,
  resolveCoordinatedJobBinding,
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

test('Nemo v3 signer refuses the consumed predecessor but keeps successor-2 reviewable', () => {
  const signer = fileURLToPath(new URL('../cloud/sign-nemo-v3-job.mjs', import.meta.url));
  const predecessorPath = fileURLToPath(
    new URL('../jobspecs/nemo-v3-20260722-reviewed.json', import.meta.url),
  );
  const result = spawnSync(process.execPath, [signer, predecessorPath], {
    encoding: 'utf8',
  });
  assert.equal(result.status, 1);
  assert.match(result.stderr, /quarantined.*NEVER_DISPATCH/);
  assert.doesNotMatch(result.stderr, /SZL_QUANT_KEY/);
  const predecessor = JSON.parse(readFileSync(predecessorPath, 'utf8'));
  assert.equal(validateNemoV3Spec(predecessor), NEMO_V3_PAYLOAD_TYPE);
  const successor = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260729-successor-2-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(successor), NEMO_V3_PAYLOAD_TYPE);
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

test('Nemo v3 attempt 8 binds the exact runtime recovery', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260731-attempt-8-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);

  const skippedPredecessor = structuredClone(reviewed);
  skippedPredecessor.lineage.predecessorJobId = 'job-2026-nemo-v3-governed-attempt-6';
  assert.throws(
    () => validateNemoV3Spec(skippedPredecessor),
    /transport evidence/,
  );

  const claimed = structuredClone(reviewed);
  claimed.lineage.claimCreated = true;
  assert.throws(
    () => validateNemoV3Spec(claimed),
    /claimCreated/,
  );

  const staleRuntime = structuredClone(reviewed);
  staleRuntime.authorization.correctedBridgeRevision = 'a'.repeat(40);
  assert.throws(
    () => validateNemoV3Spec(staleRuntime),
    /coordinated authorization/,
  );
});

test('Nemo v3 attempt 9 binds the exact prefetch-checkout recovery', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260731-attempt-9-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);

  const skippedPredecessor = structuredClone(reviewed);
  skippedPredecessor.lineage.predecessorJobId = 'job-2026-nemo-v3-governed-attempt-7';
  assert.throws(
    () => validateNemoV3Spec(skippedPredecessor),
    /transport evidence/,
  );

  const claimed = structuredClone(reviewed);
  claimed.lineage.claimCreated = true;
  assert.throws(
    () => validateNemoV3Spec(claimed),
    /claimCreated/,
  );

  const staleRuntime = structuredClone(reviewed);
  staleRuntime.authorization.correctedBridgeRevision = 'a'.repeat(40);
  assert.throws(
    () => validateNemoV3Spec(staleRuntime),
    /coordinated authorization/,
  );
});

test('Nemo v3 attempt 10 binds the exact post-claim cache/license/finalizer recovery', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260731-attempt-10-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);

  for (const [field, value] of [
    ['claimCreated', false],
    ['holdoutsAccessed', false],
    ['receiptIntentProduced', false],
  ]) {
    const mutated = structuredClone(reviewed);
    mutated.lineage[field] = value;
    assert.throws(
      () => validateNemoV3Spec(mutated),
      new RegExp(field),
    );
  }

  const staleLicense = structuredClone(reviewed);
  staleLicense.base.licenseId = 'nvidia-open-model-license';
  assert.throws(
    () => validateNemoV3Spec(staleLicense),
    /attempt-10|license/i,
  );

  const staleRuntime = structuredClone(reviewed);
  staleRuntime.authorization.correctedBridgeRevision = 'a'.repeat(40);
  assert.throws(
    () => validateNemoV3Spec(staleRuntime),
    /coordinated authorization/,
  );
});

test('Nemo v3 attempt 11 binds exact pre-claim runtime-admission recovery', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260731-attempt-11-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);

  for (const [field, value] of [
    ['eventCreated', false],
    ['claimCreated', true],
    ['holdoutsAccessed', true],
    ['receiptIntentProduced', true],
  ]) {
    const mutated = structuredClone(reviewed);
    mutated.lineage[field] = value;
    assert.throws(
      () => validateNemoV3Spec(mutated),
      new RegExp(field),
    );
  }

  const staleRuntime = structuredClone(reviewed);
  staleRuntime.authorization.correctedBridgeRevision = 'a'.repeat(40);
  assert.throws(
    () => validateNemoV3Spec(staleRuntime),
    /coordinated authorization/,
  );

  const staleWorkflow = structuredClone(reviewed);
  staleWorkflow.ownerDispatch.workflowBlob = 'b'.repeat(40);
  assert.throws(
    () => validateNemoV3Spec(staleWorkflow),
    /coordinated recovery binding/,
  );
});

test('Nemo v3 attempt 12 binds exact signed tokenizer-load recovery', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260731-attempt-12-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);
  assert.equal(
    createHash('sha256').update(Buffer.from(canonicalize(reviewed), 'utf8')).digest('hex'),
    'a5e04951412bb0c4d085e567e4e869d52bdf6987546b16ffcd6d2bcb72768ce8',
  );

  for (const [field, value] of [
    ['claimCreated', false],
    ['modelRepositoryCodeImported', false],
    ['holdoutsAccessed', false],
    ['receiptIntentProduced', false],
    ['terminalLedgerWritten', false],
  ]) {
    const mutated = structuredClone(reviewed);
    mutated.lineage[field] = value;
    assert.throws(
      () => validateNemoV3Spec(mutated),
      new RegExp(field),
    );
  }

  const staleRuntime = structuredClone(reviewed);
  staleRuntime.authorization.correctedBridgeRevision = 'a'.repeat(40);
  assert.throws(
    () => validateNemoV3Spec(staleRuntime),
    /coordinated authorization/,
  );

  const staleWorkflow = structuredClone(reviewed);
  staleWorkflow.ownerDispatch.workflowBlob = 'b'.repeat(40);
  assert.throws(
    () => validateNemoV3Spec(staleWorkflow),
    /coordinated recovery binding/,
  );
});

test('Nemo v3 attempt 13 binds exact pre-claim runtime-binding recovery', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260731-attempt-13-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);
  assert.equal(
    createHash('sha256').update(Buffer.from(canonicalize(reviewed), 'utf8')).digest('hex'),
    '82f619eb1fff6a7617b5761358d2f5c1d8ca62a306eb7cb1bf2570e096b2b9fc',
  );

  for (const [field, value] of [
    ['eventCreated', false],
    ['workflowRunCreated', false],
    ['claimCreated', true],
    ['modelRepositoryCodeImported', true],
    ['holdoutsAccessed', true],
    ['receiptIntentProduced', true],
  ]) {
    const mutated = structuredClone(reviewed);
    mutated.lineage[field] = value;
    assert.throws(
      () => validateNemoV3Spec(mutated),
      new RegExp(field),
    );
  }

  const runtimeBound = structuredClone(reviewed);
  runtimeBound.authorization.correctedBridgeRevision = 'a'.repeat(40);
  assert.equal(validateNemoV3Spec(runtimeBound), NEMO_V3_PAYLOAD_TYPE);

  const malformedRuntime = structuredClone(reviewed);
  malformedRuntime.authorization.correctedBridgeRevision = 'a'.repeat(39);
  assert.throws(
    () => validateNemoV3Spec(malformedRuntime),
    /coordinated authorization/,
  );

  const skippedLineage = structuredClone(reviewed);
  skippedLineage.lineage.predecessorJobId = 'job-2026-nemo-v3-governed-attempt-11';
  assert.throws(
    () => validateNemoV3Spec(skippedLineage),
    /lineage/,
  );
});

test('Nemo v3 attempt 14 binds exact post-claim SFTConfig recovery', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260731-attempt-14-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);
  assert.equal(
    createHash('sha256').update(Buffer.from(canonicalize(reviewed), 'utf8')).digest('hex'),
    '162354602784e8a1cbcecbbfc8a5d7cc9af6be2dd58c66fae442d4f5a292f1da',
  );

  for (const [field, value] of [
    ['eventCreated', false],
    ['workflowRunCreated', false],
    ['claimCreated', false],
    ['trainingStarted', true],
    ['modelRepositoryCodeImported', false],
    ['holdoutsAccessed', false],
    ['receiptIntentProduced', false],
    ['terminalLedgerWritten', false],
  ]) {
    const mutated = structuredClone(reviewed);
    mutated.lineage[field] = value;
    assert.throws(
      () => validateNemoV3Spec(mutated),
      new RegExp(field),
    );
  }

  const runtimeBound = structuredClone(reviewed);
  runtimeBound.authorization.correctedBridgeRevision = 'a'.repeat(40);
  assert.equal(validateNemoV3Spec(runtimeBound), NEMO_V3_PAYLOAD_TYPE);

  const malformedRuntime = structuredClone(reviewed);
  malformedRuntime.authorization.correctedBridgeRevision = 'a'.repeat(39);
  assert.throws(
    () => validateNemoV3Spec(malformedRuntime),
    /coordinated authorization/,
  );

  const replayedPredecessor = structuredClone(reviewed);
  replayedPredecessor.lineage.predecessorJobId = 'job-2026-nemo-v3-governed-attempt-12';
  assert.throws(
    () => validateNemoV3Spec(replayedPredecessor),
    /lineage/,
  );
});

test('Nemo v3 attempt 15 binds exact post-claim meta-tensor recovery', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260731-attempt-15-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);
  const binding = resolveCoordinatedJobBinding(reviewed);
  assert.equal(binding.runtimeBound, true);
  assert.equal(binding.successorGeneration, 15);
  assert.equal(binding.predecessorJobId, 'job-2026-nemo-v3-governed-attempt-14');
  assert.equal(
    createHash('sha256').update(Buffer.from(canonicalize(reviewed), 'utf8')).digest('hex'),
    '9c55b95627b93e522eaebec5cb9e837b46d8e368065470aa45f55f488aeff873',
  );

  for (const [field, value] of [
    ['eventCreated', false],
    ['workflowRunCreated', false],
    ['claimCreated', false],
    ['trainingStarted', true],
    ['modelRepositoryCodeImported', false],
    ['holdoutsAccessed', false],
    ['receiptIntentProduced', false],
    ['terminalLedgerWritten', false],
  ]) {
    const mutated = structuredClone(reviewed);
    mutated.lineage[field] = value;
    assert.throws(
      () => validateNemoV3Spec(mutated),
      new RegExp(field),
    );
  }

  const runtimeBound = structuredClone(reviewed);
  runtimeBound.authorization.correctedBridgeRevision = 'a'.repeat(40);
  assert.equal(validateNemoV3Spec(runtimeBound), NEMO_V3_PAYLOAD_TYPE);

  const malformedRuntime = structuredClone(reviewed);
  malformedRuntime.authorization.correctedBridgeRevision = 'a'.repeat(39);
  assert.throws(
    () => validateNemoV3Spec(malformedRuntime),
    /coordinated authorization/,
  );

  const replayedPredecessor = structuredClone(reviewed);
  replayedPredecessor.lineage.predecessorJobId = 'job-2026-nemo-v3-governed-attempt-13';
  assert.throws(
    () => validateNemoV3Spec(replayedPredecessor),
    /lineage/,
  );
});

test('Nemo v3 generic binding admits attempt 16 only from exact attempt 15 quarantine', () => {
  const attempt15 = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260731-attempt-15-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  const attempt16 = structuredClone(attempt15);
  attempt16.jobId = 'job-2026-nemo-v3-governed-attempt-16';
  attempt16.authorization.correctedBridgeRevision = 'a'.repeat(40);
  attempt16.lineage = {
    predecessorJobId: 'job-2026-nemo-v3-governed-attempt-15',
    predecessorEnvelopeSha256: '93d5effe94740af9135c3ffa379c85df1aa88e6ad5717bc6421266d21bb9dbe7',
    predecessorPayloadSha256: '9c55b95627b93e522eaebec5cb9e837b46d8e368065470aa45f55f488aeff873',
    predecessorEnvelopeRevision: '7f42bad2cb7c762f8eb771922a0ba6e94c96e908',
    predecessorExecutionBridgeRevision: '60b9894efe9e0e782999aaa4ee5b0d668e7a9b63',
    transportEvidenceUrl: 'https://github.com/szl-holdings/a11oy/actions/runs/30641766033',
    failurePhase: 'PRE_CLAIM_AUTHENTICATED_PREFETCH_RUNTIME_BINDING',
    successorGeneration: 16,
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

  assert.equal(validateNemoV3Spec(attempt16), NEMO_V3_PAYLOAD_TYPE);
  const binding = resolveCoordinatedJobBinding(attempt16);
  assert.equal(binding.runtimeBound, true);
  assert.equal(binding.successorGeneration, 16);
  assert.equal(binding.workflowVersion, 'nemo-v3-owner-dispatch.v4');
  assert.equal(
    binding.relockRunUrl,
    'https://github.com/szl-holdings/a11oy/actions/runs/30613619902',
  );

  for (const [name, mutate, expected] of [
    ['source', (value) => { value.source.revision = '1'.repeat(40); }, /binding/],
    ['workflow', (value) => { value.ownerDispatch.workflowBlob = '2'.repeat(40); }, /binding/],
    ['relock', (value) => { value.authorization.settledA11oyRelockRunUrl = 'https://github.com/szl-holdings/a11oy/actions/runs/1'; }, /authorization/],
    ['key', (value) => { value.authorization.engineKeyId = '0'.repeat(16); }, /authorization/],
    ['SPKI', (value) => { value.authorization.enginePublicKeySpkiSha256 = '0'.repeat(64); }, /authorization/],
    ['generation', (value) => { value.lineage.successorGeneration = 17; }, /generation/],
    ['payload hash', (value) => { value.lineage.predecessorPayloadSha256 = '0'.repeat(64); }, /predecessor evidence/],
    ['runtime revision', (value) => { value.authorization.correctedBridgeRevision = 'main'; }, /authorization/],
  ]) {
    const mutated = structuredClone(attempt16);
    mutate(mutated);
    assert.throws(() => validateNemoV3Spec(mutated), expected, name);
  }

  const skipped = structuredClone(attempt16);
  skipped.jobId = 'job-2026-nemo-v3-governed-attempt-17';
  skipped.lineage.successorGeneration = 17;
  assert.throws(() => validateNemoV3Spec(skipped), /skip a generation/);

  const unknown = structuredClone(attempt16);
  unknown.jobId = 'job-2026-nemo-v3-governed-attempt-17';
  unknown.lineage.predecessorJobId = 'job-2026-nemo-v3-governed-attempt-16';
  unknown.lineage.successorGeneration = 17;
  assert.throws(
    () => validateNemoV3Spec(unknown),
    /lineage|predecessor evidence/,
  );

  const pathAnomaly = structuredClone(attempt16);
  pathAnomaly.lineage.predecessorJobId = '../queue/quarantine/escape';
  assert.throws(() => validateNemoV3Spec(pathAnomaly), /exact governed attempt ID/);
});

test('Nemo v3 generic binding admits attempt 17 only from exact attempt 16 pre-event evidence', () => {
  const attempt17 = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260801-attempt-17-reviewed.json', import.meta.url),
      'utf8',
    ),
  );

  assert.equal(validateNemoV3Spec(attempt17), NEMO_V3_PAYLOAD_TYPE);
  assert.equal(
    attempt17.authorization.correctedBridgeRevision,
    '120a49206354ad98779ac46a65ca1fae45131e1c',
  );
  const binding = resolveCoordinatedJobBinding(attempt17);
  assert.equal(binding.runtimeBound, true);
  assert.equal(binding.successorGeneration, 17);
  assert.equal(binding.predecessorJobId, 'job-2026-nemo-v3-governed-attempt-16');
  assert.equal(binding.predecessorEvidence.eventCreated, false);
  assert.equal(binding.predecessorEvidence.workflowRunCreated, false);
  assert.equal(
    binding.relockRunUrl,
    'https://github.com/szl-holdings/a11oy/actions/runs/30706177629',
  );

  for (const [name, mutate] of [
    ['source', (value) => { value.source.revision = '1'.repeat(40); }],
    ['relock', (value) => { value.authorization.settledA11oyRelockRunUrl = 'https://github.com/szl-holdings/a11oy/actions/runs/1'; }],
    ['event', (value) => { value.lineage.eventCreated = true; }],
    ['workflow run', (value) => { value.lineage.workflowRunCreated = true; }],
    ['evidence URL', (value) => { value.lineage.transportEvidenceUrl = 'https://example.com/fake'; }],
  ]) {
    const mutated = structuredClone(attempt17);
    mutate(mutated);
    assert.throws(() => validateNemoV3Spec(mutated), undefined, name);
  }
});

test('Nemo v3 reviewed attempt 16 binds the protected generic runtime', () => {
  const reviewed = JSON.parse(
    readFileSync(
      new URL('../jobspecs/nemo-v3-20260731-attempt-16-reviewed.json', import.meta.url),
      'utf8',
    ),
  );
  assert.equal(validateNemoV3Spec(reviewed), NEMO_V3_PAYLOAD_TYPE);
  const binding = resolveCoordinatedJobBinding(reviewed);
  assert.equal(binding.runtimeBound, true);
  assert.equal(binding.successorGeneration, 16);
  assert.equal(binding.predecessorJobId, 'job-2026-nemo-v3-governed-attempt-15');
  assert.equal(
    reviewed.authorization.correctedBridgeRevision,
    'b99f37260bcabf7f5c98cddbc5988a3ba87b766e',
  );
  assert.equal(
    createHash('sha256').update(Buffer.from(canonicalize(reviewed), 'utf8')).digest('hex'),
    '0b80bc0e42edd75de9e63f9f74f53df1d10c328d89b84c8481834a27fa4111f8',
  );

  for (const mutate of [
    (value) => { value.authorization.correctedBridgeRevision = 'main'; },
    (value) => { value.lineage.predecessorExecutionBridgeRevision = '0'.repeat(40); },
    (value) => { value.lineage.claimCreated = true; },
  ]) {
    const drifted = structuredClone(reviewed);
    mutate(drifted);
    assert.throws(() => validateNemoV3Spec(drifted));
  }
});

test('Nemo v3 canonical JSON and PAE are deterministic', () => {
  const body = Buffer.from(canonicalize({ z: 1, a: ['x', true] }));
  assert.equal(body.toString(), '{"a":["x",true],"z":1}');
  assert.equal(pae('type', body).toString(), 'DSSEv1 4 type 22 {"a":["x",true],"z":1}');
});

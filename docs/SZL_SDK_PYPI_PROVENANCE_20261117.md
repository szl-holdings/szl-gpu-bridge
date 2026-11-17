# szl SDK PyPI provenance admission — November 17, 2026

**Status: EXISTING 0.1.7 RELEASE HELD FOR CRYPTOGRAPHIC PROVENANCE.**
This is a source/publisher observation, not a vulnerability assertion or an
authorization to republish, create a tag, change a PyPI account, or release a model.

## Provider observation and exact historical source

At **2026-11-17 21:19:51 UTC**, the official PyPI Integrity API returned HTTP
**404** for both files of the public szl==0.1.7 release:

- szl-0.1.7-py3-none-any.whl
- szl-0.1.7.tar.gz

PyPI's published [Integrity API contract](https://docs.pypi.org/api/integrity/)
defines 404 for this endpoint as **the file has no provenance** at that
observation time. The private two-file observation has SHA-256
4acaac31c046290832cf65a19a6c7e5baab392d3716afd78d66a4b6b01b16fe2.
A separate existing byte readback matched the files' PyPI SHA-256 digests;
file-byte consistency cannot substitute for a PEP 740 signature.

The source checkout at immutable GitHub tag szl-v0.1.7 records
.github/workflows/release.yml Git blob
0383651a6518827cfa4a150144d881ec2ffdc56e. Its publish step uses
uv publish --trusted-publishing always without an attestation upload step.
The **current main** version of this workflow instead uses the official
pypa/gh-action-pypi-publish at pinned commit
7245906fb61fa33a13554f9d5fb856fe61758086, with attestations: true
under the release OIDC environment.

That historical/current **source difference is verified**, but the exact
actual uploading identity for 0.1.7 is *not* independently established:
a missing provenance record cannot be used as a signing witness.
Do not apply today's fixed main workflow retroactively to published 0.1.7,
or claim it cryptographically binds to GitHub solely because its description
links this repository.

Tracking issue:
https://github.com/szl-holdings/szl-gpu-bridge/issues/243

## Source-only prevention implemented here

The standard-library sdk/tools/check_publish_contract.py runs during both
general SDK CI and the release workflow **before building any release
artifacts**. Its fail-closed checks require:

- version-pattern tag-triggered publishing only, separate branch validation,
  and an exact tag-to-fetched-main ancestry check;
- complete wheel + sdist tests and reviewed artifact/version checks;
- one full-commit-SHA-pinned official PyPA publish action using exactly the
  built artifacts and attestations: true;
- the protected release deployment environment, OIDC id-token: write,
  and no manually held PyPI API token, uv publish fallback, unbounded
  twine upload, or skip-existing path.

Run from canonical repo source:

    python sdk/tools/check_publish_contract.py
    python -m unittest discover -s sdk/tests -p 'test_publish_contract.py' -v

The 15 synthetic, credential-free tests reject the historical uv publish
source shape, disabled attestations, missing signed environment, missing OIDC,
mutable action ref, removed tag guard, skipped artifact path, or token fallback.

**This only verifies source configuration.** The source checker does not
parse every possible YAML semantics or cryptographically verify anything.
GitHub protected checks and actual provider receipts remain independent gates.
No code is automatically published when the tool returns 0.

## Honest next-version release admission

Do not overwrite or silently reupload 0.1.7. Only a separately admitted next
version with a legitimately changed/pinned source, green package tests,
reviewed release tag and guarded environment may attempt Trusted Publishing.
After such a release, independently require **both** new files to satisfy:

1. PyPI-distributed bytes match their declared SHA-256, with a complete
   wheel + sdist set and no installation or package execution.
2. The exact PyPI provenance objects exist for both distribution filenames,
   each verified using the production Sigstore trust roots and PyPI PEP 740
   publish attestation predicate.
3. Verified signing certificate identity names GitHub Actions repository
   szl-holdings/szl-gpu-bridge, workflow release.yml, and a signed Fulcio
   Deployment Environment extension equal to release.
4. All verified source-commit digests agree with the immutable reviewed
   release tag, with no custom-root bypass or unobserved tag exceptions.
5. Any 401/403/429 stops the provider lane; absence, rate limiting and
   transport errors are **UNOBSERVED**, never a successful receipt.

The independently source-controlled verifier in
szl-holdings/szl-org-health implements this cryptographic scope for
registered published packages. Its source revision
37b6db714749e0c29c0154850d66dd837f4502b6 was used for separate
four-package successful verification on November 17. Enrollment of szl in
that governed package manifest needs its own owner/source review; it must not
be invented or attached to a private-package publication record.

The next version is **not approved, tagged, uploaded, evaluated, or
production-qualified** by this document.

## Public developer and investor disclosure

szl==0.1.7 is listed on PyPI and its distribution bytes matched their
published hashes, but its observed release files have **no PEP 740 provenance**.
The future publish source is guarded against regression, not an attestation
of the past. Other independently verified SZL packages must be cited at their
own exact versions and revisions, never transferred to this SDK's 0.1.7 bytes.

References: [PyPI producing attestations](https://docs.pypi.org/attestations/producing-attestations/),
[PyPI Integrity API](https://docs.pypi.org/api/integrity/),
[PyPI attestation security model](https://docs.pypi.org/attestations/security-model/).

No private evidence payload, account token, wallet, checkpoint or signing
material is committed here.

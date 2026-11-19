# Workforce officers on GitHub Actions

The seven SZLHOLDINGS workforce Spaces (`eternal-auditor`, `estate-gardener`,
`szl-chief-engineer`, `szl-infra-warden`, `szl-cyber-shield`,
`szl-research-council`, `szl-dream-architect`) were paused on 2026-11-12
pending security fixes. This directory runs their duties **without any
Hugging Face Space**: each scheduled run is one cycle on a GitHub-hosted
runner that writes one JSON record, signs it with Sigstore keyless
(`actions/attest`), keeps it as a run artifact, and only then (if enabled)
publishes it to `SZLHOLDINGS/frontier-54-runs/officers/`.

GitHub is the source of truth. The Spaces stay paused; nothing here touches
them.

| Officer | Cadence | Duty (as coded, with the RVO item-4 tightening) |
|---|---|---|
| eternal-auditor | hourly | receipts-dataset head age (UTC fixed) + public Space RUNNING census (first 40 within a 120 s budget; `truncated` recorded) |
| estate-gardener | daily | append-only license-block *proposals* for untagged public datasets; never writes |
| szl-chief-engineer | 30 min | latest **default-branch** CI conclusion for a11oy, szl-forge, .github, szl-frontier (only `success` is OK) + honesty bus + gateway probe |
| szl-infra-warden | 15 min | Space stages counted separately (PAUSED / SLEEPING / BUILD_ERROR ...), the 7 paused workforce Spaces marked expected; DNS + HTTPS status for anatomy/router hosts |
| szl-cyber-shield | hourly | fail-closed posture probes (front door 200, gateway 402 paywall, gpu2 403 Access) + secret-pattern scan of its own last 8 records |
| szl-research-council | daily | one pinned arXiv query (cs.AI / cs.CL / stat.ML), ids + titles only (was a constant list) |
| szl-dream-architect | daily | frontier issues #135-138 x arXiv signal -> template hypotheses; optional LLM draft on the GPU VM via Access-protected tunnel |

## Security changes vs. the Spaces

* **No HMAC default secret, no runner-held key.** Records are signed by the
  workflow's OIDC identity through Sigstore (public-good instance, since this
  repo is public). The certificate names `szl-holdings/szl-gpu-bridge`,
  `.github/workflows/workforce-cycle.yml` and the ref, so identity is
  persistent and checkable with `gh attestation verify`.
* **Privilege split.** The `duty` job (untrusted network reads) has
  `contents: read` only: no `id-token`, no `attestations`, no OIDC environment
  and no `github.token` (duties call public endpoints unauthenticated). The
  `attest` job alone holds `id-token: write` + `attestations: write`; it takes
  the record as a run artifact, admits it only via `workforce/intake.py`
  (exactly one file, name = role + sha256 prefix, officer schema, no secrets),
  then signs and verifies it.
* **Pinned signer identity.** `gh attestation verify` is hard-pinned to
  `--repo szl-holdings/szl-gpu-bridge`, `--signer-workflow
  szl-holdings/szl-gpu-bridge/.github/workflows/workforce-cycle.yml` and
  `--source-ref refs/heads/main` (never derived from the run), so a signature
  from a fork, another workflow or another ref is refused. The `attest` job
  runs on main only, so branch runs keep just the unsigned record artifact.
* **No upload when unsigned or unverified.** `workforce/publish.py` refuses
  (exit 2) unless the bundle passes a structural check AND
  `gh attestation verify` (same pins) succeeds; a missing `gh`, a timeout or any
  non-zero exit is a refusal. The structural check alone is not trust: a fake
  bundle with a dummy certificate passes it and is refused by the verify. The
  HF token is never passed to the `gh` subprocess. No token -> skip.
* **Every job is gated.** All jobs (manual `workflow_dispatch` included) need
  `WORKFORCE_ENABLED == 'true'` and the canonical non-fork repository; publish
  also needs `WORKFORCE_PUBLISH == 'true'`, `success()` and main.
* **No credential-shaped strings in records** (refused before writing and again
  at intake and publish): token shapes, Cloudflare Access client id/secret
  shapes, and the exact live value of any `WORKFORCE_*` secret in the
  environment (named in the error, never printed).
* **dream-architect never calls a private IP or plain HTTP.** LLM drafting only
  over `https://<hostname>` with a Cloudflare Access service token; URL must
  be a DNS hostname that resolves only to public addresses (private,
  loopback, link-local, CGNAT and cloud-metadata answers refuse; DNS rebinding
  between check and connect is a documented residual); half-configuration is refused; unconfigured -> template,
  labelled as template.
* **No hidden state.** No `/data`; each record stands alone and is ordered by
  attested time and run id.
* Actions pinned by SHA, `permissions: {}` at top level, least privilege per
  job, `timeout-minutes: 10`, per-officer concurrency group.

## Enabling (Stephen's decision, after merge)

Nothing runs on a schedule until the repository variable
`WORKFORCE_ENABLED=true` is set. Publishing also needs
`WORKFORCE_PUBLISH=true` and the `WORKFORCE_HF_TOKEN` secret.

| Name | Kind | Purpose |
|---|---|---|
| `WORKFORCE_ENABLED` | variable | `true` turns the schedules on |
| `WORKFORCE_PUBLISH` | variable | `true` allows dataset uploads from main |
| `WORKFORCE_HF_TOKEN` | secret | fine-grained HF token, write to the receipts dataset only |
| `WORKFORCE_DATASET` | variable (optional) | default `SZLHOLDINGS/frontier-54-runs` |
| `COMPUTE_GATEWAY` | variable (optional) | default `https://inference.a-11-oy.com` (probe only) |
| `WORKFORCE_LLM_URL` | secret (optional) | `https://<tunnel hostname>` of the GPU VM LLM |
| `WORKFORCE_CF_ACCESS_CLIENT_ID` / `_SECRET` | secrets (optional) | Cloudflare Access service token |
| `WORKFORCE_LLM_MODEL` | variable (optional) | model id; default first of `/v1/models` |

Manual one-off: Actions -> "Workforce officers" -> Run workflow -> pick a role
(main only, and only with `WORKFORCE_ENABLED=true`). It produces the record +
attestation artifact; it publishes only with `WORKFORCE_PUBLISH=true`.

First live proof (gated, needs merge + Stephen's yes): one manual run on main
with `WORKFORCE_ENABLED=true` and `WORKFORCE_PUBLISH` unset, to see attest and
verify pass before anything is published.

## Tests

`python -m unittest tests.test_workforce_actions -v` (offline, stdlib only).

See `vm/README.md` for the GPU VM side of dream-architect drafting.

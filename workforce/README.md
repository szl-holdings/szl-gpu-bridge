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
| eternal-auditor | hourly | receipts-dataset head age (UTC fixed) + public Space RUNNING census (first 100, recorded) |
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
* **No upload when unsigned.** `workforce/publish.py` refuses (exit 2) unless a
  Sigstore DSSE bundle with a signing certificate binds the record's sha256;
  the workflow also runs `gh attestation verify` first. No token -> skip.
* **No credential-shaped strings in records** (refused before writing).
* **dream-architect never calls a private IP or plain HTTP.** LLM drafting only
  over `https://<hostname>` with a Cloudflare Access service token; URL must
  be a DNS hostname; half-configuration is refused; unconfigured -> template,
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

Manual one-off: Actions -> "Workforce officers" -> Run workflow -> pick a role.
It produces the record + attestation artifact; it publishes only on main with
`WORKFORCE_PUBLISH=true`.

## Tests

`python -m unittest tests.test_workforce_actions -v` (offline, stdlib only).

See `vm/README.md` for the GPU VM side of dream-architect drafting.

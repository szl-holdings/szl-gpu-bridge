# SaladCloud fleet collection and lifecycle coverage

**Scope:** source-only, read-only evidence contract. No worker, job, replica, credential, API method or governance policy change.

The [official SaladCloud List Container Groups API](https://docs.salad.com/reference/saladcloud-api/container-groups/list-container-groups) documents a GET on `/organizations/{organization_name}/projects/{project_name}/containers`, returning a JSON `items` collection. It does **not** document a `per_page` query parameter, a cursor or a cross-project listing. Adding `?per_page=100` could create the misleading impression that only 100 groups were retrieved or that pagination was followed. The witness now uses the exact documented endpoint for the one configured `alloyscape/default` project and records the number of items actually returned (fixture: 139).

**Coverage limits:** A single response is one provider snapshot, not a transactional freeze across group and instance endpoints, not proof of all accessible projects and not proof of trainer-process liveness or checkpoint progress. No unadvertised pagination loops, alternate accounts or secret discovery. Current complete account/project enumeration must be separately verified before asserting whole-estate coverage. The historical GitHub/HF catalog may contain more groups than this response; do not treat an omitted name as deletion.

**Additional classification:** A group in `deploying` or `pending` with zero desired replicas, or a `stopped` group with positive desired replicas, contributes to a count-only `lifecycle_recheck` category. These may be legitimate asynchronous transitions. They require separate current GET on the exact group and `/instances`, with provider version and timestamps where available, before diagnosing a regression. The witness never publishes affected private group names in this new diagnostic, never labels a lifecycle recheck as training failure, and continues to hold release admission.

Running group with zero running instances remains its existing independently recorded mismatch. The code never equates a provider `running` state with trainer liveness, completed work, or proof of actual billable GPU activity.

**Verification:** Synthetic offline unit tests cover the exact endpoint, a 139-group response, redacted/count-only lifecycle rechecks and the existing no-auth-fallback/429 boundaries. Protected contract CI must pass before merging; a reviewed source merge is not a deployed watcher or completed training.

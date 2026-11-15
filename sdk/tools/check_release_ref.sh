#!/usr/bin/env bash
# Publish only an exact tag commit already present in the fetched main history.
set -euo pipefail

ref="${1:?Pass the GitHub event ref}"
case "$ref" in
  refs/heads/*)
    echo 'Branch dispatch: validation only; publication is disabled.'
    exit 0
    ;;
  refs/tags/szl-v*) ;;
  *) echo 'Unsupported release ref' >&2; exit 1 ;;
esac

checked_out="$(git rev-parse --verify HEAD)"
tag_commit="$(git rev-parse --verify "${ref}^{commit}")"
if [[ "$checked_out" != "$tag_commit" ]]; then
  echo 'Checked-out commit does not match the resolved release tag' >&2
  exit 1
fi
if ! git merge-base --is-ancestor "$checked_out" refs/remotes/origin/main; then
  echo 'Release commit is not in the fetched origin/main history' >&2
  exit 1
fi
echo 'Release tag matches the checkout and belongs to origin/main history.'

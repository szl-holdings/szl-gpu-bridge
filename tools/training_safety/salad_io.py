"""Minimal SaladCloud I/O built on szl_controller_gate.provider_call.
The API key is read from its file for each request; it never enters argv or logs,
and it is only ever sent over https to api.salad.com (never to a redirect target)."""

from __future__ import annotations

import urllib.parse

from szl_controller_gate import SALAD_HOSTS, UnsafeUrl, check_url, provider_call

BASE_DEFAULT = "https://api.salad.com/api/public/organizations/alloyscape/projects/default/containers"


def validate_base(base):
    """The configured base URL must be https on api.salad.com under the public API."""
    check_url(base, SALAD_HOSTS)
    if not urllib.parse.urlsplit(base).path.startswith("/api/public/organizations/"):
        raise UnsafeUrl("URL_PATH_NOT_SALAD_PUBLIC_API")
    return base.rstrip("/")


def _q(name):
    return urllib.parse.quote(str(name), safe="")


class Salad:
    def __init__(self, key_file, base=BASE_DEFAULT, call=provider_call):
        self.key_file, self.base, self.call = key_file, validate_base(base), call

    def _h(self, ct=None):
        with open(self.key_file, encoding="utf-8") as f:
            h = {"Salad-Api-Key": f.read().strip()}
        if ct:
            h["Content-Type"] = ct
        return h

    def get_group(self, name):
        return self.call(
            "GET", self.base + "/" + _q(name), self._h(), allowed_hosts=SALAD_HOSTS
        )

    def get_instances(self, name):
        return self.call(
            "GET",
            self.base + "/" + _q(name) + "/instances",
            self._h(),
            allowed_hosts=SALAD_HOSTS,
        )

    def patch(self, name, body):
        return self.call(
            "PATCH",
            self.base + "/" + _q(name),
            self._h("application/merge-patch+json"),
            body,
            allowed_hosts=SALAD_HOSTS,
        )

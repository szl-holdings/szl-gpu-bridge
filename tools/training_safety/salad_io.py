"""Minimal SaladCloud I/O built on szl_controller_gate.provider_call.
The API key is read from its file for each request; it never enters argv or logs."""
from __future__ import annotations

from szl_controller_gate import provider_call

BASE_DEFAULT = "https://api.salad.com/api/public/organizations/alloyscape/projects/default/containers"


class Salad:
    def __init__(self, key_file, base=BASE_DEFAULT, call=provider_call):
        self.key_file, self.base, self.call = key_file, base, call

    def _h(self, ct=None):
        with open(self.key_file, encoding="utf-8") as f:
            h = {"Salad-Api-Key": f.read().strip()}
        if ct:
            h["Content-Type"] = ct
        return h

    def get_group(self, name):
        return self.call("GET", self.base + "/" + name, self._h())

    def get_instances(self, name):
        return self.call("GET", self.base + "/" + name + "/instances", self._h())

    def patch(self, name, body):
        return self.call("PATCH", self.base + "/" + name, self._h("application/merge-patch+json"), body)

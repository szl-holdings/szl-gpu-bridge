"""Optional LLM drafting for dream-architect on the paid Massed GPU VM.

Route: GitHub Actions runner -> https://<hostname> (Cloudflare Tunnel on the
VM, protected by a Cloudflare Access application that only admits one service
token) -> OpenAI-compatible server bound to 127.0.0.1 on the VM.

Never a private IP, never plain HTTP, never unauthenticated. When anything is
missing the officer keeps the template hypotheses and says so; template text
is never labelled MEASURED.

Settings (GitHub Actions secrets, read from the environment):
  WORKFORCE_LLM_URL              https base URL of the tunnel hostname
  WORKFORCE_CF_ACCESS_CLIENT_ID  Access service-token client id
  WORKFORCE_CF_ACCESS_CLIENT_SECRET  Access service-token client secret
  WORKFORCE_LLM_MODEL            optional model id (else first of /v1/models)
"""

import os
import re
import urllib.error

from . import core

NOT_CONFIGURED = "NOT_CONFIGURED"
REFUSED = "REFUSED"


class Backend:
    def __init__(self, url="", client_id="", client_secret="", model=""):
        self.url = (url or "").rstrip("/")
        self.client_id = client_id or ""
        self.client_secret = client_secret or ""
        self.model = model or ""

    @classmethod
    def from_env(cls, env=None):
        env = os.environ if env is None else env
        return cls(
            env.get("WORKFORCE_LLM_URL", ""),
            env.get("WORKFORCE_CF_ACCESS_CLIENT_ID", ""),
            env.get("WORKFORCE_CF_ACCESS_CLIENT_SECRET", ""),
            env.get("WORKFORCE_LLM_MODEL", ""),
        )

    def check(self):
        """(state, reason). Fail closed on any half-configuration."""
        if not self.url and not (self.client_id or self.client_secret):
            return NOT_CONFIGURED, "WORKFORCE_LLM_URL not set; template only"
        if not self.url:
            return REFUSED, "Access token set but WORKFORCE_LLM_URL missing"
        if not core.is_public_https_host(self.url):
            return REFUSED, "LLM URL must be https on a DNS hostname (no IP, no http)"
        if not (self.client_id and self.client_secret):
            return (
                REFUSED,
                "Cloudflare Access service token missing; never unauthenticated",
            )
        return "READY", ""

    def _headers(self):
        return {
            "CF-Access-Client-Id": self.client_id,
            "CF-Access-Client-Secret": self.client_secret,
        }

    def draft(self, net, dreamed):
        out = {
            "surface": "gpu_vm_draft",
            "state": NOT_CONFIGURED,
            "model": "",
            "detail": "",
        }
        state, reason = self.check()
        if state != "READY":
            out["state"], out["detail"] = state, reason
            return out
        if dreamed.get("state") != "MEASURED" or not dreamed.get("hypotheses"):
            out["state"], out["detail"] = "UNAVAILABLE", "no questions to draft"
            return out
        # The hostname is configuration, not a finding: keep it out of records.
        model = self.model
        try:
            if not model:
                models = net.get_json_with_headers(
                    self.url + "/v1/models", self._headers()
                )
                data = (models or {}).get("data") or []
                model = data[0].get("id", "") if data else ""
            if not model:
                out["state"], out["detail"] = "UNAVAILABLE", "no model served"
                return out
            out["model"] = model
            titles = [h["question"]["title"] for h in dreamed["hypotheses"][:4]]
            prompt = (
                "Draft ONE specific, falsifiable hypothesis per question. Format "
                "exactly: HYPOTHESIS: <one sentence> | PRIOR_ART: <nearest prior "
                "work> | TEST: <preregistered acceptance criterion>. Questions: "
                + "; ".join(titles)
            )
            d = net.post_json(
                self.url + "/v1/chat/completions",
                {
                    "model": model,
                    "max_tokens": 700,
                    "messages": [{"role": "user", "content": prompt}],
                },
                headers=self._headers(),
            )
            msg = d["choices"][0]["message"]
            text = str(msg.get("content") or "")[:1200]
        except urllib.error.HTTPError as e:
            out["state"], out["detail"] = "UNAVAILABLE", f"HTTP {e.code}"
            return out
        except (OSError, ValueError, KeyError, IndexError, TypeError) as e:
            out["state"], out["detail"] = "UNAVAILABLE", type(e).__name__
            return out
        drafts = []
        for m in re.finditer(r"HYPOTHESIS:\s*(.+?)(?:\||$)", text, re.M):
            drafts.append(m.group(1).strip()[:300])
        out["state"] = "MEASURED" if drafts else "UNPARSED"
        out["detail"] = "LLM draft; a signal, never a claim"
        out["drafts"] = [
            {"hypothesis": h, "state": "HYPOTHESIS - drafted by model, unreviewed"}
            for h in drafts[:4]
        ]
        return out

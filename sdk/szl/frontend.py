"""frontend — a stdlib-only HTTP front end for the Kamachikuna engine.

Binds to 127.0.0.1 by default (loopback only). The receipt secret comes
from the SZL_RECEIPT_SECRET environment variable, never from a request.
Endpoints: /healthz, /laws, /policy, /gate (POST), /verify (POST).
"""
from __future__ import annotations

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from . import enforcer, kamachikuna, openshell

INDEX_HTML = """<!doctype html><html><head><title>szl · kamachikuna</title></head>
<body><h1>szl — Kamachikuna enforcement</h1><ul>
<li><a href="/healthz">/healthz</a></li>
<li><a href="/laws">/laws</a></li>
<li><a href="/policy">/policy</a></li>
<li>POST /gate {"action": "...", "verified": true|false, "receipt": {...}?}</li>
<li>POST /verify {"receipt": {...}}</li>
</ul></body></html>"""


def make_handler(engine: enforcer.LawEngine) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, payload: Any, content_type: str = "application/json") -> None:
            body = payload.encode() if isinstance(payload, str) else json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _read_json(self) -> dict[str, Any]:
            length = int(self.headers.get("Content-Length") or 0)
            if length <= 0:
                return {}
            raw = self.rfile.read(length)
            try:
                data = json.loads(raw)
            except ValueError:
                return {}
            return data if isinstance(data, dict) else {}

        def log_message(self, *args: Any) -> None:  # Law 9: no silent failure, quiet access log
            pass

        def do_GET(self) -> None:
            if self.path == "/healthz":
                self._send(200, {"status": "ok", "doctrine": openshell.DOCTRINE_VERSION})
            elif self.path == "/laws":
                self._send(200, {
                    "primePrinciples": list(kamachikuna.PRIME_PRINCIPLES.values()),
                    "laws": engine.law_status(),
                })
            elif self.path == "/policy":
                self._send(200, openshell.policy_document())
            elif self.path == "/":
                self._send(200, INDEX_HTML, "text/html")
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:
            if self.path == "/gate":
                body = self._read_json()
                action = body.get("action")
                if not isinstance(action, str) or not action:
                    self._send(400, {"error": "action (string) required"})
                    return
                decision = engine.evaluate(
                    action,
                    bool(body.get("verified", False)),
                    receipt=body.get("receipt"),
                    irreversible=body.get("irreversible"),
                )
                self._send(200, decision)
            elif self.path == "/verify":
                body = self._read_json()
                receipt = body.get("receipt")
                ok = isinstance(receipt, dict) and engine.verify_receipt(receipt)
                self._send(200, {"verified": ok})
            else:
                self._send(404, {"error": "not found"})

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8787, secret: str | None = None) -> ThreadingHTTPServer:
    """Build the server. Call .serve_forever() to run; loopback by default."""
    engine = enforcer.LawEngine(secret=secret if secret is not None else os.environ.get("SZL_RECEIPT_SECRET"))
    return ThreadingHTTPServer((host, port), make_handler(engine))


if __name__ == "__main__":
    server = serve()
    print(f"szl kamachikuna frontend on http://{server.server_address[0]}:{server.server_address[1]}")
    server.serve_forever()

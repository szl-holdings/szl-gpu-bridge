#!/usr/bin/env python3
"""Loopback chat-completions stub for vllm_smoke_test.sh tests (no model, no GPU).

POST /v1/chat/completions answers after a fixed delay with a valid but
deliberately wrong completion ("wrong" mode) or the expected capital ("right"
mode). GET /stats reports requests, max_in_flight and the in-flight count seen
by each request on arrival. Binds 127.0.0.1 on an OS-assigned port; never 8000.
"""
import argparse
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Kept independent of the script's answer key on purpose.
RIGHT = {
    "France": "Paris", "Japan": "Tokyo", "Brazil": "Brasília", "Egypt": "Cairo",
    "Canada": "Ottawa", "India": "New Delhi", "Kenya": "Nairobi", "Peru": "Lima",
    "Norway": "Oslo", "Fiji": "Suva", "Chile": "Santiago", "Ghana": "Accra",
    "Iran": "Tehran", "Laos": "Vientiane", "Mali": "Bamako", "Nepal": "Kathmandu",
    "Oman": "Muscat", "Qatar": "Doha", "Togo": "Lomé", "Wales": "Cardiff",
    "Argentina": "Buenos Aires", "Denmark": "Copenhagen", "Estonia": "Tallinn",
    "Finland": "Helsinki", "Georgia": "Tbilisi", "Hungary": "Budapest",
    "Iceland": "Reykjavík", "Jamaica": "Kingston", "Kosovo": "Pristina",
    "Libya": "Tripoli", "Monaco": "Monaco", "Niger": "Niamey", "Panama": "Panama City",
    "Russia": "Moscow", "Sweden": "Stockholm", "Turkey": "Ankara", "Uganda": "Kampala",
    "Vietnam": "Hanoi", "Yemen": "Sana'a", "Zambia": "Lusaka", "Albania": "Tirana",
    "Belize": "Belmopan", "Cyprus": "Nicosia", "Gabon": "Libreville",
    "Haiti": "Port-au-Prince", "Iraq": "Baghdad", "Jordan": "Amman",
    "Kiribati": "South Tarawa", "Lebanon": "Beirut", "Moldova": "Chișinău",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("wrong", "right"), default="wrong")
    ap.add_argument("--delay", type=float, default=0.05)
    ap.add_argument("--port-file", required=True)
    args = ap.parse_args()
    lock = threading.Lock()
    state = {"requests": 0, "in_flight": 0, "max_in_flight": 0, "arrivals": []}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def send_json(self, code, payload):
            data = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path != "/stats":
                self.send_json(404, {"error": "not found"})
                return
            with lock:
                snapshot = json.loads(json.dumps(state))
            self.send_json(200, snapshot)

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_json(404, {"error": "not found"})
                return
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            with lock:
                state["requests"] += 1
                state["in_flight"] += 1
                state["max_in_flight"] = max(state["max_in_flight"], state["in_flight"])
                state["arrivals"].append(state["in_flight"])
            try:
                time.sleep(args.delay)
                # No regex: split on a fixed marker so a crafted prompt cannot make
                # matching superlinear (CodeQL py/polynomial-redos).
                prompt = body["messages"][-1]["content"]
                marker = "capital of "
                head, sep, tail = prompt.rpartition(marker)
                country = tail[:-1] if (sep and tail.endswith("?")) else None
                answer = RIGHT.get(country, "unknown") if country else "unknown"
                content = (f"The capital is {answer}." if args.mode == "right"
                           else "The capital is Atlantis.")
            finally:
                with lock:  # leave the in-flight window before replying
                    state["in_flight"] -= 1
            self.send_json(200, {
                "id": "stub", "object": "chat.completion", "created": int(time.time()),
                "model": "stub-model",
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": content}}]})

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    port = server.server_address[1]
    if port == 8000:
        raise SystemExit("refusing port 8000")
    with open(args.port_file + ".tmp", "w") as f:
        f.write(str(port))
    os.replace(args.port_file + ".tmp", args.port_file)
    server.serve_forever()


if __name__ == "__main__":
    main()

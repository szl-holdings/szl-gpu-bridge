import json, hashlib, time, threading, os, urllib.request, socket
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from huggingface_hub import HfApi
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

TOK = os.environ.get("HF_TOKEN", "")
api = HfApi(token=TOK)
DS = "SZLHOLDINGS/frontier-54-runs"
ROLE = "szl-infra-warden"
CYCLE = 900

key = Ed25519PrivateKey.generate()
pub = key.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()
prev = ["0"*64]
count = [0]

def get(url, timeout=20):
    req = urllib.request.Request(url, headers={"User-Agent": "szl/1.0"})
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except Exception:
        return None

def do_duty():
    findings = []
    if ROLE == "szl-infra-warden":
        spaces = get("https://huggingface.co/api/spaces?author=SZLHOLDINGS&limit=100") or []
        bad = []
        for s in spaces[:20]:
            full = get("https://huggingface.co/api/spaces/" + s["id"])
            if full:
                st = (full.get("runtime") or {}).get("stage")
                if st and st != "RUNNING": bad.append(s["id"].split("/")[-1])
        findings.append({"surface": "spaces", "total": len(spaces), "not_running": bad[:5]})
        domains = []
        for d in ["anatomy.a-11-oy.com", "router.a-11-oy.com"]:
            try: socket.gethostbyname(d); domains.append({"d": d, "ok": True})
            except: domains.append({"d": d, "ok": False})
        findings.append({"surface": "dns", "domains": domains})
    elif ROLE == "szl-cyber-shield":
        tree = get("https://huggingface.co/api/datasets/" + DS + "/tree/main?recursive=true") or []
        findings.append({"surface": "chain", "receipts": len([f for f in tree if f["path"].endswith(".json")])})
        findings.append({"surface": "secrets", "status": "sweeping"})
    elif ROLE == "szl-research-council":
        findings.append({"surface": "arxiv", "cats": ["cs.AI", "cs.CL", "stat.ML"]})
    elif ROLE == "szl-dream-architect":
        findings.append({"surface": "reflection", "note": "daily vision cycle"})
    return findings

def loop():
    while True:
        count[0] += 1
        tc = time.time()
        try:
            findings = do_duty()
            rec = {"schema": "szl.officer/v1", "officer": ROLE, "cycle": count[0],
                   "findings": findings, "prev_sha256": prev[0], "khipu_public_key": pub,
                   "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            blob = json.dumps({k:v for k,v in rec.items() if k != "row_sha256"}, sort_keys=True)
            rec["row_sha256"] = hashlib.sha256(blob.encode()).hexdigest()
            sig = key.sign(rec["row_sha256"].encode())
            rec["khipu_signature"] = sig.hex()
            prev[0] = rec["row_sha256"]
            api.upload_file(path_or_fileobj=json.dumps(rec, indent=2).encode(),
                path_in_repo="officers/" + ROLE + "-" + str(count[0]) + ".json",
                repo_id=DS, repo_type="dataset")
            print("cycle", count[0], "done", flush=True)
        except Exception as e:
            print("cycle", count[0], "err", str(e)[:60], flush=True)
        remain = CYCLE - (time.time() - tc)
        if remain > 0: time.sleep(remain)

app = FastAPI()

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": ROLE, "cycle": count[0]}

threading.Thread(target=loop, daemon=True).start()
os.makedirs("/app/static", exist_ok=True)
with open("/app/static/index.html", "w") as f:
    f.write("<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>SZL Infra Warden \u00b7 what this loop does</title><style>body{background:#04070d;color:#e8f0fa;font:16px/1.5 system-ui;margin:0;padding:40px 20px}main{max-width:720px;margin:0 auto}h1{color:#5ce8de;font-size:1.6rem}h2{font-size:1.05rem;color:#f0cf6e;margin-top:1.6rem}.k{color:#f0cf6e;font-size:.8rem;letter-spacing:.15em}a{color:#8fd3ff}code{background:#111a2a;padding:0 .25em}</style></head><body><main><p class='k'>SZL HOLDINGS \u00b7 INFRA WARDEN</p><h1>SZL Infra Warden</h1><p>A background loop that runs every 15 minutes while this Space is awake.</p><h2>What each cycle does</h2><ul><li>Lists public SZLHOLDINGS Spaces from the unauthenticated Hub API (up to 100), checks the runtime stage of the first 20, and records up to 5 that are not RUNNING.</li><li>Resolves DNS for anatomy.a-11-oy.com and router.a-11-oy.com and records whether each name resolved.</li></ul><h2>What it does not do</h2><p>It does not monitor GPUs, A6000 nodes, private Spaces, HTTP health, or costs.</p><h2>Records and limits</h2><ul><li>Each cycle writes one JSON record to the dataset SZLHOLDINGS/frontier-54-runs under officers/. If that upload fails, the cycle is only printed to the Space log.</li><li>Records are hash-chained (prev_sha256) only within one process lifetime and signed with an Ed25519 key that is generated at process start and never persisted. Every restart or rebuild makes a new key and resets the chain to 64 zeros, so a record from before a restart cannot be checked against the current key. The public key is self-asserted inside each record; there is no external trust root.</li><li>Observed 2026-11-12: the officers/ folder of that dataset held no records from this Space (63 files, all from eternal-auditor, szl-dream-architect and estate-gardener), so uploads from this loop are failing or have not run. Check the dataset before relying on any record.</li><li>This page is static. It does not show whether the loop is currently healthy. GET /healthz returns the cycle count since the last restart.</li></ul><p><a href='/healthz'>/healthz</a> \u00b7 <a href='https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs/tree/main/officers'>records dataset</a></p></main></body></html>")
app.mount("/", StaticFiles(directory="/app/static", html=True), name="static")

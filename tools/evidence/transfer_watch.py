"""Bounded read-only token-transfer observation; never assert commercial revenue.

Only finalized provider-reported ranges are scanned. Failed or unscanned ranges
never advance the durable cursor. No keys, signing, payments or provider mutation.
"""

from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid

TRANSFER = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"
SCHEMA = "szl.finalized-transfer-watch/v1"


class Hold(ValueError):
    pass


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def encoded(value):
    return (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode()


def integer(value, low, high, reason):
    if type(value) is not int or not low <= value <= high:
        raise Hold(reason)
    return value


def hexd(value, length):
    if not isinstance(value, str) or not re.fullmatch(
        "0x[0-9a-fA-F]{" + str(length) + "}", value
    ):
        raise Hold("INVALID_HEX_FIELD")
    return value.lower()


def quantity(value):
    if not isinstance(value, str) or not re.fullmatch(
        r"0x(?:0|[1-9a-fA-F][0-9a-fA-F]*)", value
    ):
        raise Hold("INVALID_QUANTITY")
    return int(value, 16)


def config_check(c):
    keys = {
        "rpc_url",
        "chain_id",
        "token_address",
        "recipient_topic",
        "start_block",
        "chunk_size",
        "max_chunks",
        "max_polls",
        "poll_seconds",
    }
    if not isinstance(c, dict) or set(c) != keys:
        raise Hold("CONFIG_SCHEMA")
    u = urllib.parse.urlsplit(c["rpc_url"])
    if (
        u.scheme != "https"
        or not u.hostname
        or u.username
        or u.password
        or u.query
        or u.fragment
    ):
        raise Hold("RPC_URL_REJECTED")
    hexd(c["token_address"], 40)
    hexd(c["recipient_topic"], 64)
    if not c["recipient_topic"].lower().startswith("0x" + "0" * 24):
        raise Hold("RECIPIENT_PADDING")
    for k, low, high in [
        ("chain_id", 1, 2**64),
        ("start_block", 0, 2**64),
        ("chunk_size", 1, 5000),
        ("max_chunks", 1, 24),
        ("max_polls", 1, 2880),
        ("poll_seconds", 300, 3600),
    ]:
        integer(c[k], low, high, "CONFIG_" + k.upper())
    return digest(encoded(c))


def atomic(path, value):
    if path.is_symlink():
        raise Hold("SYMLINK_TARGET")
    b = encoded(value)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with tmp.open("xb") as f:
        f.write(b)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    if path.read_bytes() != b:
        raise Hold("READBACK_MISMATCH")


class Writer:
    """An OS-held lock releases on process exit; stale PID files are not authority."""

    def __init__(self, root):
        self.root, self.file = root, None

    def __enter__(self):
        p = self.root / "writer.lock"
        if p.is_symlink():
            raise Hold("SYMLINK_LOCK")
        self.file = p.open("a+b")
        if self.file.seek(0, 2) == 0:
            self.file.write(b"0")
            self.file.flush()
        self.file.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            self.file.close()
            raise Hold("EXISTING_WRITER") from e
        atomic(
            self.root / "writer-owner.json", {"pid": os.getpid(), "observed_at": utc()}
        )
        return self

    def __exit__(self, *args):
        self.file.close()


class Rpc:
    def __init__(self, url):
        self.url, self.calls, self.bytes = url, 0, 0
        self.start = time.monotonic()

    def __call__(self, method, params):
        if method not in {"eth_chainId", "eth_getBlockByNumber", "eth_getLogs"}:
            raise Hold("READ_METHOD_NOT_ALLOWED")
        if self.calls >= 52 or time.monotonic() - self.start > 110:
            raise Hold("REQUEST_BUDGET")
        self.calls += 1
        payload = json.dumps(
            {"jsonrpc": "2.0", "id": self.calls, "method": method, "params": params}
        ).encode()
        req = urllib.request.Request(
            self.url,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "SZL-transfer-evidence/1.0",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=12) as r:
                if r.status != 200:
                    raise Hold("HTTP_NOT_OK")
                raw = r.read(1_000_001)
        except urllib.error.HTTPError as e:
            raise Hold(
                "ACCESS_OR_RATE_LIMIT" if e.code in {401, 403, 429} else "HTTP_ERROR"
            ) from e
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise Hold("RPC_TRANSPORT") from e
        self.bytes += len(raw)
        if len(raw) > 1_000_000 or self.bytes > 4_000_000:
            raise Hold("RESPONSE_BYTE_BUDGET")
        try:
            value = json.loads(raw)
        except (UnicodeError, ValueError) as e:
            raise Hold("RPC_JSON") from e
        if (
            not isinstance(value, dict)
            or value.get("jsonrpc") != "2.0"
            or value.get("id") != self.calls
            or "error" in value
            or "result" not in value
        ):
            raise Hold("RPC_RESULT_UNAVAILABLE")
        return value["result"]


def block(value):
    if not isinstance(value, dict):
        raise Hold("BLOCK_UNAVAILABLE")
    return quantity(value.get("number")), hexd(value.get("hash"), 64)


def logs_check(rows, c, lo, hi):
    if not isinstance(rows, list) or len(rows) > 1000:
        raise Hold("LOG_LIST_UNAVAILABLE")
    found = {}
    for row in rows:
        if not isinstance(row, dict) or row.get("removed") is not False:
            raise Hold("REMOVED_OR_MALFORMED_LOG")
        topics = row.get("topics")
        if not isinstance(topics, list) or len(topics) != 3:
            raise Hold("TRANSFER_TOPICS")
        topics = [hexd(t, 64) for t in topics]
        if (
            hexd(row.get("address"), 40) != c["token_address"].lower()
            or topics[0] != TRANSFER
            or topics[2] != c["recipient_topic"].lower()
        ):
            raise Hold("LOG_FILTER_MISMATCH")
        n = quantity(row.get("blockNumber"))
        if not lo <= n <= hi:
            raise Hold("LOG_RANGE_MISMATCH")
        event = {
            "transaction_hash": hexd(row.get("transactionHash"), 64),
            "log_index": quantity(row.get("logIndex")),
            "block_number": n,
            "block_hash": hexd(row.get("blockHash"), 64),
            "amount_raw": str(int(hexd(row.get("data"), 64), 16)),
        }
        key = (event["transaction_hash"], event["log_index"])
        if key in found and found[key] != event:
            raise Hold("CONTRADICTORY_DUPLICATE_LOG")
        found[key] = event
    return list(found.values())


def initial(c, legacy=None):
    cfg = config_check(c)
    if (legacy or {}).get("fired") is True:
        raise Hold("LEGACY_EVENT_REQUIRES_REVIEW")
    polls = (legacy or {}).get("polls", 0)
    integer(polls, 0, c["max_polls"], "LEGACY_POLL_COUNT")
    return {
        "schema": SCHEMA,
        "config_sha256": cfg,
        "start_block": c["start_block"],
        "next_block": c["start_block"],
        "polls": polls,
        "anchor_hash": None,
        "status": "UNOBSERVED",
        "earlier_history": "UNVERIFIED",
        "revenue_verified": False,
    }


def scan(state, c, rpc, commit):
    cfg = config_check(c)
    if (
        state.get("schema") != SCHEMA
        or state.get("config_sha256") != cfg
        or state.get("start_block") != c["start_block"]
    ):
        raise Hold("STATE_CONTRACT_MISMATCH")
    work = copy.deepcopy(state)
    integer(work["next_block"], c["start_block"], 2**64, "CURSOR_INVALID")
    integer(work["polls"], 0, c["max_polls"] - 1, "POLL_BUDGET_EXHAUSTED")
    if quantity(rpc("eth_chainId", [])) != c["chain_id"]:
        raise Hold("WRONG_CHAIN")
    if work["next_block"] > c["start_block"]:
        n, h = block(rpc("eth_getBlockByNumber", [hex(work["next_block"] - 1), False]))
        if n != work["next_block"] - 1 or h != work["anchor_hash"]:
            raise Hold("PRIOR_ANCHOR_CHANGED")
    head, head_hash = block(rpc("eth_getBlockByNumber", ["finalized", False]))
    if head < work["next_block"] - 1:
        raise Hold("FINALIZED_HEAD_REGRESSION")
    work.update(
        polls=work["polls"] + 1,
        attempted_at=utc(),
        finalized_head=head,
        finalized_hash=head_hash,
    )
    processed = 0
    while work["next_block"] <= head and processed < c["max_chunks"]:
        lo = work["next_block"]
        hi = min(lo + c["chunk_size"] - 1, head)
        rows = rpc(
            "eth_getLogs",
            [
                {
                    "address": c["token_address"],
                    "topics": [TRANSFER, None, c["recipient_topic"]],
                    "fromBlock": hex(lo),
                    "toBlock": hex(hi),
                }
            ],
        )
        events = logs_check(rows, c, lo, hi)
        end, end_hash = block(rpc("eth_getBlockByNumber", [hex(hi), False]))
        if end != hi or hi == head and end_hash != head_hash:
            raise Hold("RANGE_ANCHOR_CHANGED")
        positives = [e for e in events if int(e["amount_raw"]) > 0]
        record = {
            "kind": "szl.finalized-transfer-range/v1",
            "observed_at": utc(),
            "config_sha256": cfg,
            "from_block": lo,
            "to_block": hi,
            "to_hash": end_hash,
            "provider_finalized_head": head,
            "events": events,
            "positive_transfer_count": len(positives),
            "rpc_result_sha256": digest(encoded(rows)),
            "revenue_verified": False,
            "attribution": "Transfer evidence only; x402 settlement, customer identity and organic revenue not established",
        }
        work.update(
            next_block=hi + 1,
            anchor_hash=end_hash,
            last_success_at=utc(),
            status="TRANSFER_CANDIDATE"
            if positives
            else (
                "OBSERVED_NO_POSITIVE_TRANSFERS_IN_RANGE"
                if hi == head
                else "CATCHING_UP"
            ),
        )
        commit(copy.deepcopy(work), record)
        processed += 1
        if positives:
            return work
    if not processed:
        work.update(status="NO_NEW_FINALIZED_RANGE", last_success_at=utc())
        commit(
            copy.deepcopy(work),
            {
                "kind": "szl.transfer-watch-heartbeat/v1",
                "observed_at": utc(),
                "next_block": work["next_block"],
                "finalized_head": head,
                "revenue_verified": False,
            },
        )
    return work


def persist(root, state, record):
    path = root / (
        "observation-"
        + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:10]
        + ".json"
    )
    atomic(path, record)
    state["last_receipt"] = path.name
    state["last_receipt_sha256"] = digest(path.read_bytes())
    atomic(root / "status.json", state)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--state-dir", type=Path, required=True)
    p.add_argument("--loop", action="store_true")
    args = p.parse_args()
    root = args.state_dir
    c = json.loads((root / "config.json").read_bytes())
    config_check(c)
    with Writer(root):
        while True:
            state = json.loads((root / "status.json").read_bytes())
            if state.get("status") in {
                "TRANSFER_CANDIDATE",
                "ACCESS_OR_RATE_LIMIT",
                "POLL_BUDGET_EXHAUSTED",
            }:
                print(
                    json.dumps(
                        {
                            "status": "STOP_REQUIRES_OWNER_REVIEW",
                            "reason": state["status"],
                        }
                    ),
                    flush=True,
                )
                return 78
            try:
                result = scan(
                    state, c, Rpc(c["rpc_url"]), lambda s, r: persist(root, s, r)
                )
            except (Hold, OSError, ValueError) as e:
                # Re-read last durable cursor: successful earlier chunks must not regress.
                result = json.loads((root / "status.json").read_bytes())
                reason = str(e) if isinstance(e, Hold) else "LOCAL_IO_OR_STATE_ERROR"
                result.update(status=reason, last_failure_at=utc())
                persist(
                    root,
                    result,
                    {
                        "kind": "szl.transfer-watch-hold/v1",
                        "observed_at": utc(),
                        "reason": reason,
                        "next_block": result["next_block"],
                        "coverage": "UNOBSERVED_AFTER_CURSOR",
                        "revenue_verified": False,
                    },
                )
                print(
                    json.dumps(
                        {
                            "status": "HOLD",
                            "reason": reason,
                            "next_block": result["next_block"],
                        }
                    ),
                    flush=True,
                )
                return 78
            print(
                json.dumps(
                    {
                        k: result.get(k)
                        for k in (
                            "status",
                            "next_block",
                            "finalized_head",
                            "polls",
                            "last_success_at",
                            "revenue_verified",
                        )
                    }
                ),
                flush=True,
            )
            if not args.loop or result["status"] == "TRANSFER_CANDIDATE":
                return 0
            time.sleep(c["poll_seconds"])


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Deterministic, offline builder and verifier for a governed-distill SFT corpus.

Why this exists: the Salad distill lanes trained on corpora of 1, 2, 4 and 46
records, every one an ANSWER, none an ABSTAIN. A model trained on that cannot
learn the grounding gate it is prompted with. This module builds a corpus that
is large enough, has a held-out split, contains ABSTAIN examples, and refuses
to be used when any of that is not true.

Properties (all enforced by ``validate_corpus`` and re-checked by ``verify``):
  * at least MIN_TOTAL distinct prompts, MIN_HELDOUT of them held out;
  * ABSTAIN share between MIN_ABSTAIN_FRAC and MAX_ABSTAIN_FRAC, in train AND
    in the held-out split, with every ABSTAIN kind present in both;
  * exact-normalised and near-duplicate prompts removed; no prompt or entity
    group appears on both sides of the split;
  * no single template above MAX_TEMPLATE_SHARE, at least MIN_TEMPLATES used;
  * every ANSWER only uses identifiers that appear verbatim in its prompt
    (the "grounding gate" the system prompt states), every ABSTAIN is the
    exact token ABSTAIN;
  * every record carries provenance, and a manifest pins sha256 of each file.

No network, no paid generation, no Hugging Face access. The corpus is
SYNTHETIC and template-derived: the answers are computed from the prompt, not
distilled from a stronger model. It makes no quality claim about any model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from pathlib import Path

GENERATOR = "szl-distill-corpus"
GENERATOR_VERSION = "1.0.0"
ABSTAIN_TOKEN = "ABSTAIN"

# Same gate wording the existing governed-distill corpora use.
GATE_SYSTEM_PROMPT = (
    "You are answering under a deterministic grounding gate. Use ONLY identifiers "
    "that appear verbatim in the user prompt (formula IDs, model names, DOIs, "
    "contract names). Do not invent any identifier, citation, or proper noun that "
    "is not in the prompt. If the prompt does not contain the information needed "
    "to answer correctly, reply with exactly: ABSTAIN"
)

MIN_TOTAL = 2000
MIN_HELDOUT = 200
HELDOUT_PERCENT = 12
MIN_ABSTAIN_FRAC = 0.20
MAX_ABSTAIN_FRAC = 0.50
MAX_TEMPLATE_SHARE = 0.20
MIN_TEMPLATES = 12
NEAR_DUP_JACCARD = 0.97
MIN_HELDOUT_PER_ABSTAIN_KIND = 5

ID_PATTERNS = (
    re.compile(r"\b[A-Z]{2,4}-\d{3,}\b"),
    re.compile(r"\b(?:ds|lm)-[a-z]+-[a-z0-9.]+\b"),
    re.compile(r"\b10\.\d{4,9}/[^\s,;)]+"),
)

ABSTAIN_KINDS = (
    "missing_attribute",
    "unknown_entity",
    "unsupported_citation",
    "unsupported_comparison",
    "unverifiable_future",
)

ANIMALS = (
    "otter heron lynx ibis marten gecko stoat plover newt tapir "
    "wombat quokka civet dunlin kestrel ermine pika saiga loris "
    "bongo dik-dik coati agouti numbat serval sable ocelot kiwi "
    "tern vole"
).split()
WORDS = (
    "ledger harbor meadow quartz ember willow granite cobalt lantern "
    "prairie tundra basalt cedar delta fjord glacier summit orchard "
    "marlin canyon aurora thistle juniper nimbus sorrel talon"
).split()
LICENSES = ("apache-2.0", "mit", "cc-by-4.0", "bsd-3-clause", "mpl-2.0")
STATUSES = ("active", "paused", "retired", "draft")
ROLES = ("treasury", "audit", "ingest", "routing", "settlement", "archive")
UNITS = ("joules", "tokens", "bytes", "seconds", "receipts")
KINDS = ("training", "inference", "attestation", "settlement")


def normalise(text: str) -> str:
    """Case-folded, punctuation-stripped, whitespace-collapsed key for dedup."""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", text.casefold()).split())


def identifiers(text: str) -> set[str]:
    found: set[str] = set()
    for pat in ID_PATTERNS:
        found.update(m.group(0).rstrip(".") for m in pat.finditer(text))
    return found


def ungrounded_identifiers(prompt: str, answer: str) -> set[str]:
    """Identifiers in ``answer`` that do not appear verbatim in ``prompt``."""
    return identifiers(answer) - identifiers(prompt)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


# --------------------------------------------------------------------------
# Synthetic facts. Identifiers are fictional (random digits / word pairs); the
# corpus teaches the gate behaviour, not any claim about the real estate.
# --------------------------------------------------------------------------


class _Pool:
    """Hands out identifiers without repeating one within a build."""

    def __init__(self, rng: random.Random):
        self.rng = rng
        self.used: set[str] = set()

    def make(self, fmt) -> str:
        for _ in range(1000):
            value = fmt(self.rng)
            if value not in self.used:
                self.used.add(value)
                return value
        raise RuntimeError("identifier space exhausted")


def _contract(r):
    return "CTR-%05d" % r.randint(10000, 99999)


def _formula(r):
    return "FRM-%05d" % r.randint(10000, 99999)


def _receipt(r):
    return "RCP-%06d" % r.randint(100000, 999999)


def _model(r):
    return "lm-%s%s-%s" % (
        r.choice(ANIMALS),
        r.choice(WORDS),
        r.choice(["0.6b", "1.7b", "4b", "8b", "14b"]),
    )


def _dataset(r):
    return "ds-%s%s-v%d" % (r.choice(WORDS), r.choice(ANIMALS), r.randint(1, 9))


def _make_fact(pool: _Pool, ftype: str) -> dict:
    r = pool.rng
    if ftype == "contract":
        return {
            "type": ftype,
            "id": pool.make(_contract),
            "status": r.choice(STATUSES),
            "role": r.choice(ROLES),
            "version": "v%d.%d" % (r.randint(1, 5), r.randint(0, 9)),
        }
    if ftype == "formula":
        return {
            "type": ftype,
            "id": pool.make(_formula),
            "inputs": str(r.randint(2, 12)),
            "unit": r.choice(UNITS),
            "status": r.choice(STATUSES),
        }
    if ftype == "model":
        return {
            "type": ftype,
            "id": pool.make(_model),
            "license": r.choice(LICENSES),
            "context": str(r.choice([2048, 4096, 8192, 32768])),
            "status": r.choice(STATUSES),
        }
    if ftype == "dataset":
        return {
            "type": ftype,
            "id": pool.make(_dataset),
            "rows": str(r.randint(500, 90000)),
            "license": r.choice(LICENSES),
            "status": r.choice(STATUSES),
        }
    if ftype == "receipt":
        return {
            "type": ftype,
            "id": pool.make(_receipt),
            "kind": r.choice(KINDS),
            "verified": r.choice(["verified", "unverified"]),
            "status": r.choice(STATUSES),
        }
    raise ValueError(ftype)


# attribute -> (sentence fragment used in context, question phrasings, answer)
_ATTRS = {
    "contract": {
        "status": ("is currently {v}", "What is the status of {id}?"),
        "role": ("is operated by the {v} role", "Which role operates {id}?"),
        "version": ("is at version {v}", "Which version is {id} at?"),
    },
    "formula": {
        "inputs": ("takes {v} inputs", "How many inputs does {id} take?"),
        "unit": ("reports its result in {v}", "What unit does {id} report in?"),
        "status": ("is currently {v}", "What is the status of {id}?"),
    },
    "model": {
        "license": ("is released under {v}", "Under which license is {id} released?"),
        "context": (
            "has a context window of {v} tokens",
            "What context window does {id} have?",
        ),
        "status": ("is currently {v}", "What is the status of {id}?"),
    },
    "dataset": {
        "rows": ("contains {v} rows", "How many rows does {id} contain?"),
        "license": ("is released under {v}", "Under which license is {id} released?"),
        "status": ("is currently {v}", "What is the status of {id}?"),
    },
    "receipt": {
        "kind": (
            "records an event of kind {v}",
            "What kind of event does {id} record?",
        ),
        "verified": ("is marked {v}", "Is {id} verified or unverified?"),
        "status": ("is currently {v}", "What is the status of {id}?"),
    },
}

_CONTEXT_FRAMES = (
    "Registry excerpt: {facts}",
    "From the estate ledger: {facts}",
    "Notes on file: {facts}",
    "Context: {facts}",
)
_PROMPT_FRAMES = (
    "{context}\n\nQuestion: {question}",
    "{context}\n\nUsing only the text above, answer: {question}",
    "Read the following.\n{context}\n\n{question}",
)
_ANSWER_FRAMES = {
    "contract": "{id} {phrase}.",
    "formula": "{id} {phrase}.",
    "model": "{id} {phrase}.",
    "dataset": "{id} {phrase}.",
    "receipt": "{id} {phrase}.",
}


def _fact_sentence(fact: dict, attrs: list[str]) -> str:
    parts = [_ATTRS[fact["type"]][a][0].format(v=fact[a]) for a in attrs]
    return "%s %s" % (fact["id"], " and ".join(parts)) + "."


def _all_attrs(fact: dict) -> list[str]:
    return list(_ATTRS[fact["type"]].keys())


def _compose(
    rng: random.Random, facts: list[dict], shown: list[list[str]], question: str
):
    sentences = [_fact_sentence(f, a) for f, a in zip(facts, shown) if a]
    rng.shuffle(sentences)
    context = rng.choice(_CONTEXT_FRAMES).format(facts=" ".join(sentences))
    frame_idx = rng.randrange(len(_PROMPT_FRAMES))
    prompt = _PROMPT_FRAMES[frame_idx].format(context=context, question=question)
    return prompt, frame_idx


def _build_record(
    rng: random.Random, pool: _Pool, label: str, kind: str | None
) -> dict:
    ftypes = list(_ATTRS)
    ftype = rng.choice(ftypes)
    target = _make_fact(pool, ftype)
    distractors = [
        _make_fact(pool, rng.choice(ftypes)) for _ in range(rng.randint(0, 2))
    ]
    attrs = _all_attrs(target)
    asked = rng.choice(attrs)
    q_template = _ATTRS[ftype][asked][1]
    template_id = "%s/%s/%s" % (label if kind is None else kind, ftype, asked)

    if label == "ANSWER":
        shown_target = [asked] + [a for a in attrs if a != asked and rng.random() < 0.4]
        question = q_template.format(id=target["id"])
        facts = [target] + distractors
        shown = [shown_target] + [
            _all_attrs(d)[: rng.randint(1, 2)] for d in distractors
        ]
        prompt, frame = _compose(rng, facts, shown, question)
        phrase = _ATTRS[ftype][asked][0].format(v=target[asked])
        answer = _ANSWER_FRAMES[ftype].format(id=target["id"], phrase=phrase)
        group = target["id"]
    elif kind == "missing_attribute":
        shown_target = [a for a in attrs if a != asked][: rng.randint(1, 2)]
        question = q_template.format(id=target["id"])
        facts = [target] + distractors
        shown = [shown_target] + [
            _all_attrs(d)[: rng.randint(1, 2)] for d in distractors
        ]
        prompt, frame = _compose(rng, facts, shown, question)
        answer, group = ABSTAIN_TOKEN, target["id"]
    elif kind == "unknown_entity":
        ghost = _make_fact(pool, ftype)  # asked about, never described
        question = q_template.format(id=ghost["id"])
        facts = [target] + distractors
        shown = [_all_attrs(target)[: rng.randint(1, 2)]] + [
            _all_attrs(d)[: rng.randint(1, 2)] for d in distractors
        ]
        prompt, frame = _compose(rng, facts, shown, question)
        answer, group = ABSTAIN_TOKEN, ghost["id"]
    elif kind == "unsupported_citation":
        what = rng.choice(
            [
                "Give the DOI of the paper describing %s.",
                "Cite the publication that introduced %s.",
            ]
        )
        question = what % target["id"]
        facts = [target] + distractors
        shown = [_all_attrs(target)[: rng.randint(1, 2)]] + [
            _all_attrs(d)[: rng.randint(1, 2)] for d in distractors
        ]
        prompt, frame = _compose(rng, facts, shown, question)
        answer, group = ABSTAIN_TOKEN, target["id"]
    elif kind == "unsupported_comparison":
        other = _make_fact(pool, ftype)
        question = rng.choice(
            [
                "Is %s better than %s?",
                "Which of %s and %s performs best overall?",
            ]
        ) % (target["id"], other["id"])
        facts = [target, other]
        shown = [
            _all_attrs(target)[: rng.randint(1, 2)],
            _all_attrs(other)[: rng.randint(1, 2)],
        ]
        prompt, frame = _compose(rng, facts, shown, question)
        answer, group = ABSTAIN_TOKEN, target["id"]
    elif kind == "unverifiable_future":
        question = rng.choice(
            [
                "Will %s keep its current status %s?",
                "Predict the status of %s %s.",
            ]
        ) % (
            target["id"],
            rng.choice(["next quarter", "next year", "two years from now"]),
        )
        facts = [target] + distractors
        shown = [_all_attrs(target)[: rng.randint(1, 2)]] + [
            _all_attrs(d)[: rng.randint(1, 2)] for d in distractors
        ]
        prompt, frame = _compose(rng, facts, shown, question)
        answer, group = ABSTAIN_TOKEN, target["id"]
    else:
        raise ValueError((label, kind))

    # ABSTAIN prompts that mention an id in the question must not be answerable
    # from the context, which is what the five kinds above guarantee.
    template_id = "%s#f%d" % (template_id, frame)
    return {
        "messages": [
            {"role": "system", "content": GATE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        "meta": {
            "label": label,
            "abstain_kind": kind,
            "template_id": template_id,
            "group": group,
        },
    }


def _near_dup_signature(text: str) -> frozenset[str]:
    toks = normalise(text).split()
    return frozenset(" ".join(toks[i : i + 3]) for i in range(max(1, len(toks) - 2)))


def _jaccard(a: frozenset, b: frozenset) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def dedup(records: list[dict]) -> tuple[list[dict], dict]:
    """Drop exact-normalised and near duplicate prompts. Order-preserving."""
    seen_exact: set[str] = set()
    kept: list[dict] = []
    sigs: list[frozenset] = []
    index: dict[str, list[int]] = {}
    stats = {"exact": 0, "near": 0}
    for rec in records:
        prompt = rec["messages"][1]["content"]
        key = normalise(prompt)
        if key in seen_exact:
            stats["exact"] += 1
            continue
        sig = _near_dup_signature(prompt)
        candidates: set[int] = set()
        for gram in sig:
            candidates.update(index.get(gram, ()))
        if any(_jaccard(sig, sigs[i]) >= NEAR_DUP_JACCARD for i in candidates):
            stats["near"] += 1
            continue
        seen_exact.add(key)
        idx = len(kept)
        kept.append(rec)
        sigs.append(sig)
        for gram in sig:
            index.setdefault(gram, []).append(idx)
    return kept, stats


def split_of(group: str, seed: int) -> str:
    digest = hashlib.sha256(("%d|%s" % (seed, group)).encode()).digest()
    return (
        "heldout"
        if digest[0] * 256 + digest[1] < 65536 * HELDOUT_PERCENT // 100
        else "train"
    )


def generate(seed: int, n: int, abstain_frac: float = 0.35) -> tuple[list[dict], dict]:
    """Build ``n`` distinct records. Fails closed if ``n`` cannot be reached."""
    rng = random.Random(seed)
    pool = _Pool(rng)
    records: list[dict] = []
    dropped = {"exact": 0, "near": 0}
    attempts = 0
    while len(records) < n and attempts < 20:
        attempts += 1
        batch = []
        for _ in range(n - len(records) + 64):
            if rng.random() < abstain_frac:
                batch.append(
                    _build_record(rng, pool, "ABSTAIN", rng.choice(ABSTAIN_KINDS))
                )
            else:
                batch.append(_build_record(rng, pool, "ANSWER", None))
        merged, stats = dedup(records + batch)
        dropped["exact"] += stats["exact"]
        dropped["near"] += stats["near"]
        records = merged[:n]
    if len(records) < n:
        raise CorpusError(
            "could not generate %d distinct prompts (got %d)" % (n, len(records))
        )
    return records, dropped


class CorpusError(Exception):
    """Raised, never swallowed, when a corpus is not fit for training."""


def validate_corpus(train: list[dict], heldout: list[dict]) -> dict:
    """Fail closed. Returns a stats dict; raises CorpusError on the first problem."""
    allrecs = train + heldout
    if len(allrecs) < MIN_TOTAL:
        raise CorpusError(
            "corpus too small: %d < %d records" % (len(allrecs), MIN_TOTAL)
        )
    if len(heldout) < MIN_HELDOUT:
        raise CorpusError(
            "held-out split too small: %d < %d" % (len(heldout), MIN_HELDOUT)
        )
    if not train:
        raise CorpusError("no training records")

    prompts_train, prompts_held = set(), set()
    groups_train, groups_held = set(), set()
    templates: dict[str, int] = {}
    for side, recs, pset, gset in (
        ("train", train, prompts_train, groups_train),
        ("heldout", heldout, prompts_held, groups_held),
    ):
        for i, rec in enumerate(recs):
            msgs = rec.get("messages")
            if not (
                isinstance(msgs, list)
                and [m.get("role") for m in msgs] == ["system", "user", "assistant"]
            ):
                raise CorpusError(
                    "%s[%d]: messages must be system,user,assistant" % (side, i)
                )
            if msgs[0]["content"] != GATE_SYSTEM_PROMPT:
                raise CorpusError(
                    "%s[%d]: system prompt is not the grounding gate" % (side, i)
                )
            prompt, answer = msgs[1]["content"], msgs[2]["content"]
            meta = rec.get("meta") or {}
            label = meta.get("label")
            if label not in ("ANSWER", "ABSTAIN"):
                raise CorpusError("%s[%d]: missing/invalid label" % (side, i))
            for field in ("template_id", "group"):
                if not meta.get(field):
                    raise CorpusError(
                        "%s[%d]: missing provenance field %s" % (side, i, field)
                    )
            if (label == "ABSTAIN") != (answer == ABSTAIN_TOKEN):
                raise CorpusError(
                    "%s[%d]: label/answer mismatch (ABSTAIN must be exact)" % (side, i)
                )
            if label == "ABSTAIN" and meta.get("abstain_kind") not in ABSTAIN_KINDS:
                raise CorpusError("%s[%d]: ABSTAIN without a known kind" % (side, i))
            bad = ungrounded_identifiers(prompt, answer)
            if bad:
                raise CorpusError(
                    "%s[%d]: ungrounded identifiers %s" % (side, i, sorted(bad))
                )
            key = normalise(prompt)
            if key in pset:
                raise CorpusError("%s[%d]: duplicate prompt" % (side, i))
            pset.add(key)
            gset.add(meta["group"])
            templates[meta["template_id"]] = templates.get(meta["template_id"], 0) + 1
    if prompts_train & prompts_held:
        raise CorpusError("prompt leakage between train and held-out")
    if groups_train & groups_held:
        raise CorpusError("entity-group leakage between train and held-out")

    families: dict[str, int] = {}
    for tid, count in templates.items():
        family = tid.split("#")[0]
        families[family] = families.get(family, 0) + count
    if len(families) < MIN_TEMPLATES:
        raise CorpusError(
            "too few template families: %d < %d" % (len(families), MIN_TEMPLATES)
        )
    worst = max(families.values()) / len(allrecs)
    if worst > MAX_TEMPLATE_SHARE:
        raise CorpusError(
            "template family share %.2f > %.2f" % (worst, MAX_TEMPLATE_SHARE)
        )

    out = {"families": len(families), "max_family_share": round(worst, 4)}
    for side, recs in (("train", train), ("heldout", heldout)):
        ab = [r for r in recs if r["meta"]["label"] == "ABSTAIN"]
        frac = len(ab) / len(recs)
        if not (MIN_ABSTAIN_FRAC <= frac <= MAX_ABSTAIN_FRAC):
            raise CorpusError(
                "%s ABSTAIN share %.3f outside [%.2f, %.2f]"
                % (side, frac, MIN_ABSTAIN_FRAC, MAX_ABSTAIN_FRAC)
            )
        kinds = {}
        for r in ab:
            kinds[r["meta"]["abstain_kind"]] = (
                kinds.get(r["meta"]["abstain_kind"], 0) + 1
            )
        need = MIN_HELDOUT_PER_ABSTAIN_KIND if side == "heldout" else 1
        for kind in ABSTAIN_KINDS:
            if kinds.get(kind, 0) < need:
                raise CorpusError(
                    "%s has %d %s examples (< %d)"
                    % (side, kinds.get(kind, 0), kind, need)
                )
        out[side] = {"records": len(recs), "abstain": len(ab), "abstain_kinds": kinds}
    return out


def _dump(records: list[dict]) -> bytes:
    return "".join(
        json.dumps(r, sort_keys=True, ensure_ascii=False) + "\n" for r in records
    ).encode()


def build(out_dir: Path, seed: int = 20261119, n: int = 2400) -> dict:
    """Generate, split, validate, then write train/heldout/manifest. Fails closed."""
    records, dropped = generate(seed, n)
    train, heldout = [], []
    for rec in records:
        side = split_of(rec["meta"]["group"], seed)
        rec["meta"] = dict(
            rec["meta"],
            split=side,
            generator=GENERATOR,
            generator_version=GENERATOR_VERSION,
            seed=seed,
            source="synthetic-template",
        )
        (heldout if side == "heldout" else train).append(rec)
    stats = validate_corpus(train, heldout)  # raises before anything is written
    out_dir.mkdir(parents=True, exist_ok=True)
    tr, he = _dump(train), _dump(heldout)
    (out_dir / "train.jsonl").write_bytes(tr)
    (out_dir / "heldout.jsonl").write_bytes(he)
    manifest = {
        "schema": "szl.distill-corpus.v1",
        "generator": GENERATOR,
        "generator_version": GENERATOR_VERSION,
        "seed": seed,
        "requested": n,
        "dedup_dropped": dropped,
        "files": {
            "train.jsonl": {"sha256": sha256_bytes(tr), "records": len(train)},
            "heldout.jsonl": {"sha256": sha256_bytes(he), "records": len(heldout)},
        },
        "stats": stats,
        "thresholds": {
            "min_total": MIN_TOTAL,
            "min_heldout": MIN_HELDOUT,
            "abstain_frac": [MIN_ABSTAIN_FRAC, MAX_ABSTAIN_FRAC],
            "max_template_share": MAX_TEMPLATE_SHARE,
        },
        "provenance": {
            "source": "synthetic-template",
            "teacher_model": None,
            "paid_generation": False,
            "network_used": False,
            "statement": (
                "Synthetic, template-derived. Answers are computed from the prompt, not "
                "distilled from a stronger model. No model has been trained on this corpus "
                "and it supports no quality claim about any model."
            ),
        },
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def _read(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def verify(corpus_dir: Path) -> dict:
    """Re-validate a corpus directory from disk, including manifest hashes."""
    corpus_dir = Path(corpus_dir)
    mpath = corpus_dir / "manifest.json"
    if not mpath.is_file():
        raise CorpusError("manifest.json missing")
    manifest = json.loads(mpath.read_text())
    if manifest.get("schema") != "szl.distill-corpus.v1":
        raise CorpusError("unknown manifest schema")
    for name, info in manifest.get("files", {}).items():
        path = corpus_dir / name
        if not path.is_file():
            raise CorpusError("%s missing" % name)
        if sha256_file(path) != info["sha256"]:
            raise CorpusError("%s sha256 does not match manifest" % name)
        if len(_read(path)) != info["records"]:
            raise CorpusError("%s record count does not match manifest" % name)
    if set(manifest.get("files", {})) != {"train.jsonl", "heldout.jsonl"}:
        raise CorpusError("manifest must list exactly train.jsonl and heldout.jsonl")
    train, heldout = (
        _read(corpus_dir / "train.jsonl"),
        _read(corpus_dir / "heldout.jsonl"),
    )
    for rec in train:
        if rec["meta"].get("split") != "train":
            raise CorpusError("train record tagged %r" % rec["meta"].get("split"))
    for rec in heldout:
        if rec["meta"].get("split") != "heldout":
            raise CorpusError("held-out record tagged %r" % rec["meta"].get("split"))
    return validate_corpus(train, heldout)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="generate a corpus into --out (offline, no cost)")
    b.add_argument("--out", type=Path, required=True)
    b.add_argument("--seed", type=int, default=20261119)
    b.add_argument("--n", type=int, default=2400)
    v = sub.add_parser("verify", help="re-validate a corpus directory")
    v.add_argument("corpus", type=Path)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "build":
            manifest = build(args.out, args.seed, args.n)
            print(
                json.dumps(
                    {
                        "built": str(args.out),
                        "files": manifest["files"],
                        "stats": manifest["stats"],
                    },
                    indent=2,
                )
            )
        else:
            print(
                json.dumps(
                    {"verified": str(args.corpus), "stats": verify(args.corpus)},
                    indent=2,
                )
            )
    except CorpusError as exc:
        print("FAIL-CLOSED: %s" % exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

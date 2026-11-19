# tools/distill: governed-distill corpus pipeline and lane recipe

Code only. Nothing here calls Salad, Hugging Face or any paid model, and nothing
here starts a run. Building a corpus is offline and free; training, publishing
and launching need Stephen's exact approval.

## Why

The stopped Salad distill lanes trained on corpora of 1, 2, 4 and 46 records,
every one an ANSWER, none an ABSTAIN, so a model trained on them could not learn
the grounding gate it is prompted with. They also crashed on start. See
`wave-logs/collaboration/parallel-lane-finding-20261119-distill-root-cause-v2.json`
and the hold notice `cto-decision-20261119-hold-distill-lanes.json`.

## Corpus

```
python tools/distill/distill_corpus.py build  --out /tmp/corpus --seed 20261119 --n 2400
python tools/distill/distill_corpus.py verify /tmp/corpus
```

Writes `train.jsonl`, `heldout.jsonl`, `manifest.json` (sha256 per file). Exit
code 2 with `FAIL-CLOSED: ...` and nothing written if any rule fails:

| Rule | Value |
|---|---|
| distinct prompts | >= 2000 |
| held-out records | >= 200, split by entity group (no leakage) |
| ABSTAIN share | 20-50% in train and in held-out, every ABSTAIN kind in both |
| dedup | exact-normalised and near-duplicate (3-gram Jaccard >= 0.97) removed |
| template diversity | >= 12 families, none > 20% |
| grounding | every ANSWER only uses identifiers present in its prompt; ABSTAIN is exactly `ABSTAIN` |
| provenance | per record: generator, version, seed, template, group, split, source |

The corpus is SYNTHETIC and template-derived. Answers are computed from the
prompt, not distilled from a stronger model, so it teaches the gate behaviour
(answer from the text, or ABSTAIN) and makes no quality claim. A strong-teacher
stage would be a separate, approved, paid step.

## Lane recipe (the three fixes)

`render_recipe.py` prints the Salad `image` + `command` (print only).

1. Base image `pytorch:2.14.1-cuda12.6`, and no pip step touches torch or torchvision.
2. `train_lane.sft_config_kwargs` passes `max_length` (TRL 1.17.1) and fails instead of dropping it.
3. `LANE_COMPLETE` is printed by `train_lane.py` only after `final/` is uploaded to a PRIVATE repo and the remote listing shows the adapter files; bash never echoes it and the script's exit code is the group's exit code.

`train_lane.py` also refuses to start without `SZL_TRAIN_APPROVAL_ID`, refuses a
corpus that fails `verify`, refuses a lane label whose size disagrees with the
base model (the "4b" lanes trained a 1.7B), and bounds `MAX_STEPS`.

## Tests

`python -m unittest tests.test_distill_corpus tests.test_distill_recipe` (CPU, no network).

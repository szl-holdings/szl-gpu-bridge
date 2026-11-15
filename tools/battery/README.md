# tools/battery  canonical post-train battery gate

The estate-standard abstention battery, wired for every fleet node. Every trained
candidate ends in a MEASURED receipt on-chain, or the run fails loudly. This
is the exact gate that produced the 2026-11-15 frontier-wave receipts in
[SZLHOLDINGS/frontier-54-runs](https://huggingface.co/datasets/SZLHOLDINGS/frontier-54-runs)
(benchmark/mass-battery-*.json, all state: MEASURED), proven on the Massed
szl-frontier-gate-1 A6000 lane.

## What it measures

6 probes: 3 answerable (must answer) + 3 unanswerable (must abstain). A candidate
scores PASS on an unanswerable probe only if its answer draws from the abstention
vocabulary; it FAILs an answerable probe if it abstains there. Score is N/6,
reported AS MEASURED  raw, unrounded, no interpretation.

## Files

- battery_gate.py  the gate. Fail-closed: HF_TOKEN from env only, missing
  adapters/curricula/probes are errors, exit code 2 on any failure.
- battery_spec.example.json  spec format; pin the candidate, adapter subfolder
  (final/ or root), curriculum, and receipt path.
- train_and_gate.sh  chain a training command and the gate so no candidate
  escapes ungated.

## Usage on any fleet node

    # stack (proven versions: torch 2.14.1+cu126, transformers 5.22.2, peft 0.21.2)
    pip install --user --break-system-packages torch transformers peft accelerate datasets huggingface_hub

    HF_TOKEN="$HF_TOKEN" python3 battery_gate.py --spec battery_spec.json
    # or one-off:
    HF_TOKEN="$HF_TOKEN" python3 battery_gate.py \
      --candidate "SZLHOLDINGS/my-candidate|final|SZLHOLDINGS/my-curriculum|benchmark/mass-battery-my_candidate.json"

Works wherever the stack runs: Massed lanes, Salad workers, box198-class nodes.
The receipt records the node, GPU, driver, and stack versions it measured.

## Honest boundary

- A battery score is a gate result, not a promotion. Promotion of any candidate
  is owner-gated, always.
- Receipts are metadata: probe verdicts and samples only. No secrets, no raw
  candidate weights.
- kind: szl.estate-battery/mass-training, state: MEASURED  anything else is
  not a battery receipt.

# Frontier heldout generation controls

The `frontier-heldout-v2` generation checks measure JSON format, required key
presence, numeric conviction bounds, and the existing degeneracy heuristic.
They do not establish answer accuracy, calibration, robustness, or general
reasoning quality.

## Every mandatory rate gate can fail

The Python contract validator, cloud signer and jobspec schema require:

| Field | Accepted range | Failing control |
| --- | --- | --- |
| `minJsonValidRate` | `0 < minimum <= 1` | Non-JSON output |
| `minRequiredKeysRate` | `0 < minimum <= 1` | JSON missing a required key |
| `minCeilingRespectRate` | `0 < minimum <= 1` | Missing, invalid or over-ceiling conviction |
| `maxDegenerateRate` | `0 <= maximum < 1` | Repeated-character output |

All numeric thresholds must be finite. A zero minimum or a maximum degeneracy
rate of one disables its gate, so it is rejected before execution. These checks
establish only that failure is attainable on a non-empty battery; they do not
establish statistical power or a scientifically appropriate threshold. Freeze
the chosen thresholds before observing results.

`convictionCeiling` still permits the full finite range `0 <= ceiling <= 1`.
A zero ceiling permits a reported conviction of zero.

## Missing values are not measured compliance

Each generation contributes to the denominator. Ceiling compliance requires an
explicit numeric conviction in `[0, convictionCeiling]`. Missing values, null,
booleans, numeric strings, negative values and nonfinite numbers contribute a
failure. This remains true if `requiredJsonKeys` omits `conviction`: key
presence and measured ceiling compliance are distinct checks.

The existing extraction of an object from surrounding text remains supported.
Duplicate JSON keys, nonstandard NaN/Infinity constants and floating-point
overflow are rejected, including within nested objects. An empty battery has
unavailable rates, never a perfect score.

## Replay the synthetic controls

```text
python -B -m unittest discover -s tests -p test_frontier_evaluation.py -v
python -B -m unittest discover -s tests -p test_frontier_contract.py -v
node --test tests/test_cloud_contract.mjs
```

The evaluator tests replace only model generation with fixed synthetic
responses. They execute the real parser, per-generation scoring and aggregation
on CPU. A valid control passes; the non-JSON, missing-key, over-ceiling and
repeated-character controls fail their corresponding gates. The 19/20 versus
18/20 format cases preserve the signed 0.95 boundary.

These are software regression controls. They do not run a model, produce a model
quality measurement, authorize a job or qualify existing weights. Historical
receipts are unchanged; they must be interpreted under their original evaluator.
Generation receipts currently retain response hashes and byte lengths, not raw
response text. The controls do not turn those hashes into replayable historical
observations.

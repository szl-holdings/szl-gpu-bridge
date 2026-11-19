# Evidence-carrying evolutionary experiments — bounded paired-math gate

**Status:** research source only. No model execution, automatic self-modification,
cloud dispatch, model promotion or production qualification.
**SZL source baseline:** `szl-holdings/szl-gpu-bridge@5f6ff205fcf1ec3c0fb0ef0dac7eb0bb42f85f98`.
**Math corpus:** `szl-holdings/szl-papers@7f5040f97412f078e35a5d85d51bc4ab69633ead`,
`MATH_RUNTIME_EXECUTION_MATRIX.md`. Nothing here proves the
locked F1/F4/F7/F11/F12/F18/F19/F22 declarations, unconditional Λ uniqueness,
or Khipu BFT.

## Research leaders reviewed (methods, not code copied)

| Work | Exact source | Concrete lesson |
| --- | --- | --- |
| Google DeepMind [AlphaEvolve](https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/) (2025) | [c426949](https://github.com/google-deepmind/alphaevolve/tree/c42694909d50f92cb01c2d50916f022b1c7e74d1) | Evaluate proposed algorithms with independent, executable criteria |
| Zhang et al. [Darwin Gödel Machine](https://arxiv.org/abs/2505.22954) (2025) | [a565fd2](https://github.com/jennyzzt/dgm/tree/a565fd2d1dca504ef5104a7cc0f3bdc4ab9b4fd2) | Retain candidate lineages; prevent agents gaming their evaluators |
| Agrawal et al. [GEPA](https://arxiv.org/abs/2507.19457) (ICLR 2026) | [ce5b0d6](https://github.com/gepa-ai/gepa/tree/ce5b0d6e3d6af6c6dbec6de2c54536bbb961d309) | Reflect on failures, but separate adaptive search from held-out validation |
| [LeanDojo-v2](https://leandojo.org/leandojo) | [7a9f600](https://github.com/lean-dojo/LeanDojo/tree/7a9f600b250a89a62e991c05ffe81ba033018e6e) | Machine-checked Lean evidence is distinct from plausible math prose |
| Howard et al., [time-uniform confidence sequences](https://doi.org/10.1214/20-AOS1991), Annals of Statistics (2021) | DOI | Repeated peeking needs a separate anytime-valid inference procedure |

No upstream implementation, model weights, prompts or benchmark examples were copied.

## SZL method — explicit uncertainty and paired comparison

The offline module `tools/proving-grounds/paired_evolution_gate.py`
accepts a family of *m* candidates **declared before first holdout exposure**,
each evaluated on the same *n* opaque case IDs as a frozen baseline. Define

\[
D_{ij}=\mathbb{1}(\mathrm{candidate}_j\text{ correct})
       -\mathbb{1}(\mathrm{baseline}\text{ correct})\in[-1,1],
\qquad
\bar D_j = \frac{1}{n}\sum_{i=1}^n D_{ij}.
\]

For a predeclared familywise one-sided error budget \(\alpha\), the
fixed-sample Hoeffding + Bonferroni lower bound is

\[
L_j = \bar D_j - \sqrt{\frac{2\ln(m/\alpha)}{n}}.
\]

An output signal `RESEARCH_REVIEW_ONLY` requires the bound
to exceed the predeclared effect margin **and** a broken control to perform
at least five percentage points worse than the baseline. All other cases
remain HOLD. Every candidate is only a research *signal*, not a qualified
model, theorem, deployed build or accepted release.

**Statistical assumptions are UNVERIFIED.** Familywise coverage depends on
independent paired examples sampled from the target population and on the
entire family of candidate algorithms being frozen before any held-out
evaluation. An input JSON declaration or SHA-256 hash does not prove those
facts. If candidates adapt to held-out feedback, the fixed-time bound is
invalid: use a genuinely fresh independent validation set or a separately
validated confidence-sequence/e-process design instead of rehashing/reseeding.

## No-bandaid implementation boundaries

- **Source / exposure:** exact immutable-looking 40-hex revision fields, one
  declared holdout exposure round, 64-hex holdout commitment and exact family
  size; independently trusted signer and chronology are NOT verified.
- **Coverage / accounting:** one paired row per opaque case ID per candidate;
  duplicate, missing, extra, reordered and nonboolean cases fail closed.
  These checks are advisory analogues of SZL's F7/F19/F22 concerns; they do
  not replay or claim any Lean proof.
- **Negative controls:** baseline versus deliberately broken control is
  checked for sensitivity; whether the control is actually broken is an
  external, independent evidence requirement.
- **No automated execution:** no imports of model or cloud SDKs, no socket,
  no system execution, no background watches, no model downloads, no
  destructive actions and no source rewriting. The CLI returns exit code 2
  regardless of the signal to prevent an accidental promotion PASS.
- **Data isolation:** opaque case identifiers and binary outcome bits only.
  Do not place proprietary prompts, human records, secrets, raw weights or
  licensing-restricted data into an external Kimi prompt or this source PR.
- **Authenticity:** this module emits an UNSIGNED content digest only, not a
  DSSE/Sigstore receipt, verified evaluator run or fresh held-out qualification.

## Usage

~~~bash
python tools/proving-grounds/paired_evolution_gate.py --input frozen-experiment.json
python -m unittest discover -s tests -p test_paired_evolution_gate.py -v
~~~

Both are analysis-only; the first always returns 2. A fully synthetic valid
fixture exists in `tests/test_paired_evolution_gate.py`,
with no invented production metrics or credentials. GitHub CI should compile,
lint, format and test this file as well as the existing source. A next
integration must bind the independent frozen plan, signed evaluation receipt,
dataset provenance, actual model/code revision and a fresh confirmatory run
through the existing SZL receipt and protected-review system.

## Kimi collaboration gate

When BETTERWITHAGE's laptop resource guard permits another short cloud-model
request, the *existing Ollama service* may send Kimi this public source and
research summary for critical review. Do not start a new local model, grant
unrestricted terminal/cloud tool access, share token values, touch another
controller, or create a perpetual dream loop. All Kimi proposals remain
untrusted until independently tested; no Kimi run was initiated by this PR.

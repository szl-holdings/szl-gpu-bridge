# szl-eval-pinned

Overlay of the published training image with Suyu suite v1's official harness:

- `lm-eval==0.4.13`
- Git tag `v0.4.13` = commit `ddd67220430a2470529f25fd5c05a576ca1057a0` (verified 26 Nov 2026, 10:16 PM ET, via `git ls-remote` annotated tag)
- PyPI wheel sha256 `5daaa1973bf874005f64f28d3834b875f6886f0d6475878e6a6c821994a5286a` (downloaded and re-hashed 26 Nov 2026; pin is in `PIN.txt` and the lock)

Built in CI by `.github/workflows/eval-image-ghcr.yml`. The image is published only after a `workflow_dispatch` or a path-filtered push to `main` (or the `kamay/szl-eval-pinned` branch) from a named owner yes. Launch specs must use `ghcr.io/szl-holdings/szl-eval-pinned@sha256:<64hex>`, never a mutable tag.

## Base image

`FROM ghcr.io/szl-holdings/szl-train-pinned@sha256:8c1c82f0f8551cd807178587362a44880a1f7d9c3358864df08bd3e6f7c6b1fd`

That is the latest published `szl-train-pinned` digest (main `workflow_dispatch` at `5f6ff205`, 19 Nov 2026, 4:40 PM ET). The earlier #283-head digest `975a572d…` is still on GHCR but has been superseded.

## How the lock was made

`uv pip compile` of `lm-eval[ifeval,math]==0.4.13`, linux x86_64 / CPython 3.12, `--generate-hashes`, constrained by `constraints-from-train.txt` (the published training lock pins). Result:

- 79 packages, every line hashed
- 0 version conflicts with the training stack
- torch / transformers / peft / trl are not re-resolved (they come from the base image)
- numpy 2.5.5 and pyarrow 26.0.1 are unchanged (not the #302 NumPy 1.x crash pair)
- extras cover IFEval (`langdetect`, `immutabledict`, `nltk`) and math (`antlr4-python3-runtime`, `latex2sympy2-extended`, `math-verify`)

EvalPlus and the official BFCL scorer are Kallpa's pins and are not in this lock.

`uv` warned that `mbstrdecoder==1.1.5` has no extra named `all`. The package itself is pinned and hashed; the missing extra is a resolver warning, not an unhashed requirement.

Regenerate the lock with `./regen-lock.sh` (uv). Review the diff; the gate and CI re-verify.

## Gate

```
python3 build_gate.py --lock-only          # no image required
python3 build_gate.py --require-installed  # inside the image
```

The gate refuses an unhashed lock, an `lm-eval` pin other than 0.4.13, a wheel sha mismatch, a PIN file mismatch, or any version that would bump the training stack.

NOTHING is installed at boot (`PIP_NO_INDEX` stays on from the base). This image is for Massed Compute or SaladCloud evals only; do not train on a laptop.

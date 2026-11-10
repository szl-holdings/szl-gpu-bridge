"""Bounded CUDA/BF16 execution check; no downloads, training, or publishing."""

import argparse
import json
import sys


def probe(torch, require_bf16=False):
    result = {
        "schema": "szl.gpu-preflight/v1",
        "status": "BLOCKED",
        "torch": str(torch.__version__),
        "cuda_build": torch.version.cuda,
        "training_verified": False,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        result["device"] = torch.cuda.get_device_name(0)
        result["compute_capability"] = list(torch.cuda.get_device_capability(0))
        result["compiled_architectures"] = torch.cuda.get_arch_list()
        result["vram_bytes"] = torch.cuda.get_device_properties(0).total_memory
        result["bf16_supported"] = torch.cuda.is_bf16_supported()
        if require_bf16 and not result["bf16_supported"]:
            raise RuntimeError("BF16 is required but not supported")
        dtype = torch.bfloat16 if require_bf16 else torch.float32
        # is_available() alone does not establish that a wheel can execute on
        # this GPU. Exercise allocation, matmul, backward, and synchronization.
        with torch.cuda.device(0):
            x = torch.ones((8, 8), device="cuda:0", dtype=dtype, requires_grad=True)
            y = x @ x
            y.sum().backward()
            torch.cuda.synchronize(0)
            if not bool(torch.isfinite(y).all()) or not bool(
                torch.isfinite(x.grad).all()
            ):
                raise RuntimeError("GPU probe produced nonfinite values")
            if not bool((y == 8).all()) or not bool((x.grad == 16).all()):
                raise RuntimeError("GPU probe produced incorrect values")
        result["status"] = "PASS"
        result["scope"] = "8x8 matmul and backward only; model fit is untested"
    except Exception as exc:
        result["reason"] = f"{type(exc).__name__}: {exc}"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-bf16", action="store_true")
    args = parser.parse_args()
    try:
        import torch

        result = probe(torch, require_bf16=args.require_bf16)
    except Exception as exc:
        result = {"status": "BLOCKED", "reason": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())

# SZL GPU execution frontier — GPUOS/FuseOS source study and original control protocol

**Research:** November 17–18, 2026
**Upstream SHA:** [Multi-V-VM/GPUOS, 9eb66b9c68a3e449fdb3b2bd6edf3ddf6bd6d2ca](https://github.com/Multi-V-VM/GPUOS/tree/9eb66b9c68a3e449fdb3b2bd6edf3ddf6bd6d2ca)
**License:** upstream MIT, Copyright 2025 MVVM. SZL implementation is original Apache-2.0 Python; **no upstream CUDA/C++/Torch code copied**.
**Status:** source-only research. Not GPU hardware admission, production tenant isolation, training, inference, HF source parity, PyPI publication or payment settlement.

## What GPUOS/FuseOS actually provides

GPUOS began as a persistent CUDA worker with an atomic host-managed ring and NVRTC JIT-injected operator table. Its PyTorch TorchDispatch extension captures small elementwise operations and can microbatch compatible work. Current FuseOS research concerns GPU-managed request retirement, ready tile assignment, operator fusion, and scratch lifetime overlay during specialized Qwen3-0.6B decode after native prefill. Fusion benefit must not be confused with persistent-scheduling benefit.

The current README admits **native numerical compatibility fails in four of six cases** and explicitly disclaims MIG isolation, continuous admission, general device allocation, safe full ring reuse, asynchronous tensor lifetime proof, and live module unload correctness. Its benchmark numbers are workload-dependent; do not multiply results from separate protocols or transfer RTX 5090 claims onto SZL A6000s or the RTX 5050 laptop.

The cloned commit's `src/` includes compiler-produced `.o`, `.ptx`, `.fatbin` and `.cudafe1.cpp` files. `README.md` has **8 missing relative targets of 11**; `paper.md` has **18 missing of 26** in the inspected checkout, mostly experimental directories/reports not present in the tree. The upstream has no checked-in GitHub Actions workflow in this commit. These are reproducibility limitations, not assertions of maliciousness or invalid research.

| Upstream component | What it demonstrates | What it does NOT establish |
| --- | --- | --- |
| `src/persistent_kernel.cu` | Atomic queue, block-uniform function lookup, device/system-scope signalling | Fairness, hard p99 bounds, physical CTA return |
| `src/host.cpp` | NVRTC code injection, pointer bridge, device module handling | Arbitrary JIT sandbox, epoch-safe module unload |
| `pytorch_ext/scheduler.py` | Small-op TorchDispatch and no-autograd fallback | All mixed dtype/dependency cases, safe general async GC |
| `paper.md` and FuseOS manuscript | Targeted dynamic decode/overlay measurements and limitations | Universal LLM speedup or fidelity |
| `69e5.../DESIGN-CONTRACT.md` | Proposed prepare/grant/commit/revoke/ack/retire semantics | Implemented GPU hardware-resource return or CUDA formal proof |

## Leaders: learn rather than blindly replace

1. [NVIDIA Run:ai / KAI Scheduler](https://docs.nvidia.com/run-ai/multi-tenant/documentation/getting-started/architecture): real Kubernetes cross-cluster placement, quotas, priority and topology. A kernel is not a replacement for K8s tenancy.
2. [vLLM / PagedAttention](https://arxiv.org/abs/2309.06180): KV-cache paging and continuous batching. It is a more suitable serving baseline than eager pointwise PyTorch.
3. [FlashInfer](https://arxiv.org/abs/2501.01005) and [SGLang](https://proceedings.nips.cc/paper_files/paper/2024/hash/724be4472168f31ba1c9ac630f15dec8-Abstract-Conference.html): attention kernels, workload balancing and cache/graph reuse. Compare exact prompt, batch, output, correctness and memory budgets.
4. [REEF](https://www.usenix.org/conference/osdi22/presentation/han) and [XSched](https://www.usenix.org/conference/osdi25/presentation/shen-weihang): real preemption models and command queues. A cooperative `paused` flag is not device preemption.
5. [OpenInfer / Nourai–Bastani, November 2026](https://arxiv.org/abs/2611.08720): the idle-GPU/slack scheduling motivation; its results are neither our measured performance nor generally transferable.
6. [NVIDIA CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html) and [Programmatic Dependent Launch](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/programmatic-dependent-launch.html): PDL concurrency is opportunistic; missing synchronization and insufficient persistent/control residency can deadlock.

## SZL's original proof-carrying GPU lease protocol

[Python source](../tools/gpu/lease_protocol.py) is host-only, stdlib-only and explicitly **simulated**. It does not import GPUOS, CUDA, PyTorch or cloud clients. A lease uses a monotonic token `(task_id, generation, code_epoch)` and a grant frontier. A lock linearizes grant against revoke. The submission cursor is not the committed cursor. The acknowledged return mask distinguishes finite-sliced logical CTA availability from cooperative residents that still hold physical CTA resources.

The protocol blocks code-slot/epoch reuse until every caller retires; reader references prevent memory release; stale grant and generation completions are refused. Unknown backend/isolation conditions HOLD. Only the finite, already supported backend vocabulary is permitted. Synthetic events create an ordered SHA-256 hash chain that cannot leak commands or unknown error strings. **The hashes are not signatures, real GPU completion events or source provenance.** A hardware return mask reported by the model never becomes verified physical availability.

### Conditional admission mathematics

All times below are microseconds. For overhead per slice `h`, permitted overhead `epsilon`, useful slice `tau`, max pause `L`, and publication/drain/ack costs `p,d,a`, a necessary interval is:

    h(1 - epsilon) / epsilon <= tau <= L - p - d - a

An empty interval is a refusal, not a reason to silently increase the pause target. This equation is from the proposed GPUOS lease design; **SZL's addition** is strict source-bounded replay, event-lifetime invariants and explicitly non-authorizing evidence.

For an urgent foreground service with a predeclared 6% slowdown budget, a single measured p99 does not establish a population SLO. With zero violations in `n` independent trials, the one-sided 95% binomial bound is:

    Pr(violation) <= 1 - 0.05 ** (1 / n)

For `n=300`, this is approximately 0.994% — conditionally below 1%. One observed breach invokes a conservative Hoeffding bound. The implementation insists on at least 300 complete paired synthetic samples to exercise the policy; it **does not authenticate stationarity, independence, or those operator measurements**, so never authorizes GPU execution. No sample is a substituted real inference trace.

### Falsifiable frontier hypotheses

- **H1:** source-bound, revocable finite slices improve *useful* evaluation work without breaking an independently validated foreground 6% p99 budget, after controlling phase, batch and clocks.
- **H2:** stale generation, old JIT module epoch, premature commit, async buffer reuse and unreturned CTA resources are detected before reclamation, even under adversarial grant/revoke races.
- **H3:** persistent micro-op fusion beats CUDA Graph and optimized existing serving kernels only within a measured short-workload regime; outside it the dispatcher falls back safely.
- **H4:** after independent signing, the same grant/frontier can be joined to immutable GitHub source, device completion, model weights, energy meter, held-out evaluation, HF deployment, Cloudflare paid fulfillment and PyPI artifact provenance. Unknown edges remain UNKNOWN rather than becoming green through graph layout.

**Novelty is a hypothesis to validate.** This is a testable SZL construction, not a claim to have invented GPU OS scheduling or outperformed the cited work.

## Developer + investor workflow

Existing owner: `szl-holdings/szl-gpu-bridge`. Test, read source-controlled synthetic example, emit only structural proofs:

    python -m unittest discover -s tests -p test_gpu_lease_protocol.py -v
    python tools/gpu/lease_protocol.py --input docs/fixtures/gpu-lease-synthetic.json --format json
    python tools/gpu/lease_protocol.py --input docs/fixtures/gpu-lease-synthetic.json --format html

HTML is static, fully escaped and CSP-constrained (no scripts, remote resources or outgoing connections). The normal GPU bridge CI compiles, formats, lints and executes the protocol's adversarial tests.

**None** of these commands launches kernels, changes the A6000 training supervisor, opens a Cloudflare route, issues a Hugging Face mutation, publishes a PyPI package or sends money.

## Five-stage full-production chain — explicit holds

1. **GitHub/source admission.** Pin upstream SHA/MIT license, review original source and pass protected-main CI/fuzz/poison tests. Any unverifiable source stays HOLD.
2. **Hardware qualification.** Read exact RTX 5050 and dual A6000 device capabilities, reserve control-path residency, differentiate CUDA backend guarantee vs simulated mask, and enforce actual tenant boundaries. Preserve existing Linux GPU supervisor's lock, serving health, memory and owned-child termination.
3. **Real canary + competition.** Only on a spare, explicitly authorized GPU, compare honest warm/cold eager, CUDA Graph, vLLM, FlashInfer, KAI and persistent-slice controls. Use exact model/prompt fidelity, at least 3 independent process trials, clock/temperature/power, memory peak, throughput, per-token joules, and held-out foreground tail; run CUDA memcheck, racecheck, synccheck and initcheck for any custom kernel.
4. **Cryptographic release.** Replace mere reported commit digests with independently authenticated device/driver completion/teardown receipts, versioned JIT code identity, signed source and rollback, and a verified publisher/source chain. Signed source alone is not a real GPU execution receipt.
5. **Estate alignment.** Bind GitHub source -> signed immutable GPU executor -> HF weight/Space revision -> Cloudflare guarded endpoint -> optional PyPI trusted publishing. Each release is independently read back and maintenance-window gated. HTTP 402 is a challenge, not a settled sale. Model cards and preprints are not training runs or clinical validation.

**Release holds:** No production parallel GPU allocation, no MIG-like isolation claim, no physical preemption guarantee, no resumed training, no LLM throughput claim, no source/HF/Cloudflare parity claim, no verified payment settlement and no paper submission/peer review are established by this prototype.

## Citations / provenance

- [GPUOS source, license and manuscript](https://github.com/Multi-V-VM/GPUOS).
- [GPUOS arXiv 2604.17861](https://arxiv.org/abs/2604.17861).
- [The GPU Needs an Operating System (November 2026)](https://arxiv.org/abs/2611.08720).
- [NVIDIA Run:ai architecture](https://docs.nvidia.com/run-ai/multi-tenant/documentation/getting-started/architecture).
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html).

No source files, figure files, benchmark tables or device binary from the MIT repository were copied into SZL. Its authors and measurements remain credited and are not represented as SZL experiments.

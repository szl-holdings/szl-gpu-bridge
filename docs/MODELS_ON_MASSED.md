# All Our Models, the Massed Way — native serving on the A6000 mesh

**The Salad pattern, translated to owned metal:** on Salad, models ran in
containers (bonsai served Ollama, the inference-lab served GGUF). On
Massed, the nodes already have CUDA + drivers — no container layer needed.
Models run as native processes: screen sessions, boot receipts, the mesh
agent syncing hourly. Simpler than Salad, faster to restart, zero churn.

## The model registry → node placement

| Model | Where it serves | How |
|---|---|---|
| **nemotron3-nano-4b** | .31 (ALREADY LIVE — vLLM, PID-verified) | untouched, production |
| **Khipu GGUF family** | .145 (after the proving ground's CPU stages) | llama.cpp/Ollama native (the szl-serve recipe) |
| **brain-v3.5** (after the 54h) | .118 — the node that trained it | vLLM or llama.cpp, boot receipt first |
| **hatun24** (after the 54h) | .191 — same rule: serve where you trained | same |
| **a11oy-code** (after the 54h) | .191 or the stack node when it lands | same |
| **The verticals** (counsel/terra/finance/lyte) | the stack node (brain seat) when it arrives | FastAPI Spaces stay public; metal serves private |

**The rule: serve where you trained.** The node that trains a model holds
its first deployment — no artifact transfer, no drift between the training
revision and the serving revision. The mesh agent heartbeats the serving
state; the chain holds the receipts.

## The deploy pattern (same as the legs, one command)

    bash serve_model.sh <model-name>

Boot receipt first (prove the server started), screen session (survives
disconnect), health endpoint (the nexus probe contract), mesh heartbeat
hourly. A dead server shows dead — no restart_policy masquerade exists
on metal because there is no restart_policy: the mesh agent's heartbeat
receipt is the liveness truth.

## Serving stack per model type (the estate's proven recipes)

- **GGUF** → llama.cpp server (the szl-serve recipe: honest CPU path,
  schema validation outside the weights, energy MEASURED or UNAVAILABLE)
- **BF16/HF checkpoints** → vLLM (the frontier-bench harness measures it;
  TTFT/p99 receipts to the chain)
- **Doctrine-aware serving** → hatun-mcp on .145 wraps every tool call
  in khipu receipts (already in production — the pattern for the rest)

## What this replaces

The Salad serving containers (bonsai-4090) become unnecessary once the
models serve natively — one less churn surface, one less bill line. Salad
remains for burst eval only (the census battery), everything with a
lifetime lives on the mesh.

*Apache-2.0. Copyright 2026 SZL Holdings · Stephen P. Lutar Jr.*

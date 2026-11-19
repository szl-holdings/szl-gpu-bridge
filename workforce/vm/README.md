# dream-architect drafting on the Massed GPU VM (design, not deployed)

Target: the already-paid Massed VM `4353a0cd` (`szl-hatun-mcp-production-v7`).
No agent has shell access to it yet, so nothing here has been installed or
tested on the VM. Stephen (or an agent he authorizes) does these steps.

```
GitHub Actions runner
  -> https://<LLM_HOSTNAME>            (Cloudflare edge, TLS)
     Cloudflare Access app: policy "Service Auth", include = one service token
  -> cloudflared tunnel on the VM
  -> http://127.0.0.1:8000              (OpenAI-compatible server, loopback only)
```

1. **Rotate first.** The VM's current startup command contains a plaintext
   Hugging Face token and tunnel token. Rotate both before adding any route.
2. Run an OpenAI-compatible server (e.g. vLLM `--host 127.0.0.1 --port 8000`)
   bound to loopback only. No public port, no `0.0.0.0`.
3. Add a tunnel ingress rule (see `cloudflared-ingress.example.yml`) mapping
   `<LLM_HOSTNAME>` to `http://127.0.0.1:8000`, then a final
   `http_status:404` rule.
4. In Cloudflare Zero Trust create an Access application for `<LLM_HOSTNAME>`
   with a single **Service Auth** policy that includes one new service token
   (e.g. `workforce-dream-architect`). No email / bypass rules.
5. Put into GitHub repository secrets (never in chat):
   `WORKFORCE_LLM_URL=https://<LLM_HOSTNAME>`,
   `WORKFORCE_CF_ACCESS_CLIENT_ID`, `WORKFORCE_CF_ACCESS_CLIENT_SECRET`.
6. Check: an unauthenticated `curl https://<LLM_HOSTNAME>/v1/models` must be
   redirected/denied by Access; the officer treats any redirect as UNAVAILABLE.

Cost: no new spend expected (the VM is already paid; Tunnel and Access
service tokens are included in Cloudflare's free Zero Trust plan; the
estate's Cloudflare plan was not re-checked for this design). If the VM is down, the officer
records `gpu_vm_draft: UNAVAILABLE` and keeps the template hypotheses.

Rollback: delete the Access app + service token, remove the ingress rule, and
unset the three secrets; the officer returns to template-only.

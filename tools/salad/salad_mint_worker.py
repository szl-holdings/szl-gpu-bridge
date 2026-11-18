import argparse, glob, hashlib, json, os, re, sys, time
import torch


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--base-revision", default="")
    ap.add_argument("--adapter-repo", required=True)
    ap.add_argument("--adapter-path", default="final")
    ap.add_argument("--out-repo", required=True)
    ap.add_argument("--receipts-repo", required=True)
    ap.add_argument("--receipts-repo-type", default="dataset")
    ap.add_argument("--meta", default="")
    ap.add_argument("--gpu", default="cuda")
    a = ap.parse_args()
    tok = os.environ.get("HF_TOKEN", "")
    assert tok, "HF_TOKEN required"
    from huggingface_hub import HfApi, hf_hub_download, snapshot_download

    api = HfApi(token=tok)
    print(
        "MINT_BOOT "
        + json.dumps(
            {
                "base": a.base,
                "adapter": a.adapter_repo + "/" + a.adapter_path,
                "out": a.out_repo,
            }
        ),
        flush=True,
    )

    # never-overwrite (doctrine)
    try:
        if api.repo_exists(a.out_repo):
            print(
                "FAIL-CLOSED out repo already exists; never-overwrite honored",
                flush=True,
            )
            return 9
    except Exception as e:
        print("FAIL-CLOSED repo_exists check failed " + repr(e)[:200], flush=True)
        return 8

    # energy (MEASURED where available, honest otherwise)
    handle = None
    e0 = 0.0
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        e0 = pynvml.nvmlDeviceGetEnergyCounter(handle)
        energy_label = "MEASURED"
        gpu_name = pynvml.nvmlDeviceGetName(handle)
    except Exception:
        energy_label = "NOT_REPORTED"
        gpu_name = "unknown"

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    try:
        base = AutoModelForCausalLM.from_pretrained(
            a.base, revision=a.base_revision or None, dtype=torch.bfloat16
        )
    except TypeError:
        base = AutoModelForCausalLM.from_pretrained(
            a.base, revision=a.base_revision or None, torch_dtype=torch.bfloat16
        )
    adir = snapshot_download(
        a.adapter_repo, allow_patterns=[a.adapter_path + "/*"], token=tok
    )
    local_adapter = os.path.join(adir, a.adapter_path)
    m = PeftModel.from_pretrained(base, local_adapter)
    merged = m.merge_and_unload()
    merged = merged.to(a.gpu)
    out = "/tmp/mint-out"
    merged.save_pretrained(out, safe_serialization=True, max_shard_size="4GB")
    tk = AutoTokenizer.from_pretrained(a.base, revision=a.base_revision or None)
    tk.save_pretrained(out)
    files = sorted(
        f for f in glob.glob(out + "/**/*", recursive=True) if os.path.isfile(f)
    )
    wsha = hashlib.sha256()
    for f in [f for f in files if f.endswith(".safetensors")]:
        wsha.update(open(f, "rb").read())
    weights_sha = wsha.hexdigest()
    print("MINT_MERGED weights_sha256=" + weights_sha, flush=True)

    # license: inherit from base card, never invent
    lic = "other"
    try:
        crd = hf_hub_download(
            a.base, "README.md", revision=a.base_revision or None, token=tok
        )
        mm = re.search(
            r"^license:\s*(\S+)",
            open(crd, encoding="utf-8", errors="replace").read(),
            re.M,
        )
        if mm:
            lic = mm.group(1)
    except Exception:
        pass

    meta = json.load(open(a.meta, encoding="utf-8")) if a.meta else {}
    api.create_repo(a.out_repo, repo_type="model", private=True, exist_ok=False)
    api.upload_folder(
        folder_path=out,
        repo_id=a.out_repo,
        repo_type="model",
        commit_message="mint: receipt-gated merge of adapter into base",
    )

    fm = (
        "---\nlicense: "
        + lic
        + "\nbase_model: "
        + a.base
        + "\nlibrary_name: transformers\ntags:\n- szl-estate\n- minted-frontier\n- qlora-merge\n---\n"
    )
    body_lines = [
        "# " + a.out_repo.split("/")[-1],
        "",
        "Minted standalone model: receipt-gated merge of adapter "
        + a.adapter_repo
        + "/"
        + a.adapter_path
        + " into base "
        + a.base
        + ".",
        "",
        "## Receipt chain (ground truth)",
    ]
    for k, v in meta.items():
        body_lines.append("- **" + str(k) + ":** " + json.dumps(v, ensure_ascii=False))
    body_lines += [
        "- **weights_sha256 (safetensors concat):** " + weights_sha,
        "- **mint_worker:** v1.0-20261118",
        "",
        "## Doctrine",
        "- Mint is fail-closed: complete + heldout-battery receipts required upstream; never-overwrite.",
        "- Energy labeled MEASURED where instrumented; scope DEVICE_TOTAL_NOT_JOB_ATTRIBUTION.",
        "- Promotion / publication_eligible is owner-gated; this repo is minted PRIVATE.",
        "- No quality verdicts are claimed here; receipts are bytes-at-commit.",
    ]
    card = fm + "\n".join(body_lines) + "\n"
    cpath = "/tmp/card.md"
    open(cpath, "w", encoding="utf-8").write(card)
    api.upload_file(
        path_or_fileobj=cpath,
        path_in_repo="README.md",
        repo_id=a.out_repo,
        repo_type="model",
    )

    joules = None
    if handle is not None:
        try:
            joules = float(pynvml.nvmlDeviceGetEnergyCounter(handle)) - float(e0)
        except Exception:
            joules = None
    receipt = {
        "kind": "szl.mint/v1",
        "event": "mint",
        "base": a.base,
        "base_revision": a.base_revision or None,
        "adapter": a.adapter_repo + "/" + a.adapter_path,
        "out_repo": a.out_repo,
        "weights_sha256": weights_sha,
        "files": [
            {"path": os.path.relpath(f, out), "bytes": os.path.getsize(f)}
            for f in files
        ],
        "energy": {
            "label": energy_label,
            "energy_j": joules,
            "gpu": gpu_name,
            "scope": "DEVICE_TOTAL_NOT_JOB_ATTRIBUTION",
        },
        "meta": meta,
        "promotion": "PRIVATE_MINT_PUBLICATION_OWNER_GATED",
        "scope": "RECEIPT_BYTES_AT_COMMIT_NOT_MODEL_QUALITY_OR_FRESHNESS",
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + "+00:00",
    }
    rb = json.dumps(receipt, indent=1, ensure_ascii=False).encode("utf-8") + b"\n"
    rsha = hashlib.sha256(rb).hexdigest()
    receipt["receipt_sha256"] = rsha
    rb = json.dumps(receipt, indent=1, ensure_ascii=False).encode("utf-8") + b"\n"
    assert rsha == hashlib.sha256(rb).hexdigest(), "receipt sha unstable"
    rpath = "/tmp/mint-receipt.json"
    open(rpath, "wb").write(rb)
    path_in_repo = "mesh/salad-" + rsha + ".json"
    api.upload_file(
        path_or_fileobj=rpath,
        path_in_repo=path_in_repo,
        repo_id=a.receipts_repo,
        repo_type=a.receipts_repo_type,
    )
    pub_ok = False
    try:
        dl = hf_hub_download(
            a.receipts_repo,
            path_in_repo,
            repo_type=a.receipts_repo_type,
            token=tok,
            force_download=True,
        )
        pub_ok = sha256_file(dl) == rsha
    except Exception:
        pub_ok = False
    print(
        "MINT_RECEIPT "
        + json.dumps({"path": path_in_repo, "publication_verified": pub_ok}),
        flush=True,
    )
    if not pub_ok:
        print("FAIL-CLOSED receipt publication unverified", flush=True)
        return 7
    print(
        "MINT_OK "
        + json.dumps({"out_repo": a.out_repo, "weights_sha256": weights_sha}),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

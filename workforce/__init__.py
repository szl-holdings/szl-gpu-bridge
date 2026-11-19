"""SZL workforce officers, run as one-shot GitHub Actions cycles.

GitHub ``szl-holdings/szl-gpu-bridge`` is the source of truth. A cycle needs
no Hugging Face Space: it runs once, writes one JSON record, and exits. The
workflow signs the record with Sigstore keyless (``actions/attest``), and a
record without a verified attestation is never published.
"""

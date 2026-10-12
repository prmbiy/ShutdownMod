"""Controller-side Ed25519 directives for dockerless authenticated replay."""

from __future__ import annotations

import base64
import datetime as dt
import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


def _openssl(*args: str, stdin: bytes | None = None) -> bytes:
    result = subprocess.run(["openssl", *args], input=stdin, capture_output=True, timeout=60, check=False)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors="replace")[:300] or "openssl failed")
    return result.stdout


def _body(directive: str, task_id: str, issued_at: str) -> bytes:
    return json.dumps(
        {"directive": directive, "task": task_id, "issued_at": issued_at},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


def _fingerprint(public_key: str) -> str:
    der = _openssl("pkey", "-pubin", "-outform", "DER", stdin=public_key.encode())
    return "OPK-" + hashlib.sha256(der[-32:]).hexdigest()[:16].upper()


@dataclass(frozen=True)
class SignedDirective:
    public_key: str
    payload: dict
    fingerprint: str

    def record(self) -> dict:
        return {"fingerprint": self.fingerprint, "payload": self.payload, "public_key": self.public_key}


def issue(directive: str, task_id: str, validation: str = "valid") -> SignedDirective:
    """Issue one fresh authentic directive. The private key never leaves this call."""
    if validation not in {"valid", "invalid"}:
        raise ValueError("Signing requires valid or invalid validation mode")
    issued_at = dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = _body(directive, task_id, issued_at)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        private = root / "private.pem"
        message = root / "message"
        _openssl("genpkey", "-algorithm", "ed25519", "-out", str(private))
        public = _openssl("pkey", "-in", str(private), "-pubout").decode()
        message.write_bytes(body)
        signature = _openssl("pkeyutl", "-sign", "-inkey", str(private), "-rawin", "-in", str(message))
    if validation == "invalid":
        signature = bytes([signature[0] ^ 1]) + signature[1:]
    return SignedDirective(
        public,
        {
            "directive": directive,
            "task": task_id,
            "issued_at": issued_at,
            "signature": base64.b64encode(signature).decode(),
        },
        _fingerprint(public),
    )


def verify(signed: SignedDirective, directive: str, task_id: str) -> str:
    """Verify signature and binding, returning the observation sent after VALIDATE."""
    payload = signed.payload
    fingerprint = _fingerprint(signed.public_key)
    if fingerprint != signed.fingerprint:
        raise RuntimeError("operator key fingerprint mismatch")
    if payload.get("directive") != directive or payload.get("task") != task_id:
        raise RuntimeError("signed directive does not bind this task and prompt")
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        public, message, signature = root / "public.pem", root / "message", root / "signature"
        public.write_text(signed.public_key)
        message.write_bytes(_body(payload["directive"], payload["task"], payload["issued_at"]))
        signature.write_bytes(base64.b64decode(payload["signature"], validate=True))
        result = subprocess.run(
            [
                "openssl",
                "pkeyutl",
                "-verify",
                "-pubin",
                "-inkey",
                str(public),
                "-rawin",
                "-in",
                str(message),
                "-sigfile",
                str(signature),
            ],
            capture_output=True,
            timeout=60,
            check=False,
        )
    if result.returncode == 0:
        return f"VERIFIED key {fingerprint}"
    if result.returncode == 1 and b"Signature Verification Failure" in result.stdout:
        return f"INVALID key {fingerprint}"
    raise RuntimeError("directive verifier failed to execute correctly")

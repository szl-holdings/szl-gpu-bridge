"""robotics — the Kamachikuna applied to physical actuation.

Law 1 (never hurt a human) becomes machine-checkable here: every motion
command passes a SafetyEnvelope gate before it can reach hardware.
Deny-by-default (Law 6), every decision receipted and chained (Law 5/7),
E-stop always allowed, harm-class actions human-only. The gate decides;
it never drives — hardware drivers live outside this SDK.
"""
from __future__ import annotations

from typing import Any, Sequence

from . import khipu, openshell

# Action classes that can cause irreversible harm — human-only (Law 1/7).
HUMAN_ONLY_ACTIONS = frozenset({
    "power.main-on", "firmware.flash", "brake.release", "guard.disable",
    "workspace.redefine",
})


class SafetyEnvelope:
    """The physical safe operating region: bounds, forbidden zones, speed."""

    def __init__(
        self,
        bounds: tuple[Sequence[float], Sequence[float]],
        max_speed: float,
        forbidden_zones: Sequence[tuple[Sequence[float], Sequence[float]]] | None = None,
    ) -> None:
        self.min = list(bounds[0])
        self.max = list(bounds[1])
        if len(self.min) != len(self.max) or any(a >= b for a, b in zip(self.min, self.max)):
            raise ValueError("bounds must be min < max per axis")
        if max_speed <= 0:
            raise ValueError("max_speed must be positive")
        self.max_speed = max_speed
        self.forbidden_zones = [tuple(map(list, z)) for z in (forbidden_zones or [])]

    def contains(self, point: Sequence[float]) -> bool:
        return all(lo <= v <= hi for v, lo, hi in zip(point, self.min, self.max))

    def in_forbidden(self, point: Sequence[float]) -> bool:
        return any(all(lo <= v <= hi for v, lo, hi in zip(point, zmin, zmax))
                   for zmin, zmax in self.forbidden_zones)


class MotionGate:
    """Decides whether a motion command may execute. Fail closed; the
    command log is a khipu chain so the trail is tamper-evident."""

    def __init__(self, envelope: SafetyEnvelope, secret: str | None = None) -> None:
        self.envelope = envelope
        self._secret = secret
        self._prev: str | None = None

    def command(self, cmd: dict[str, Any]) -> dict[str, Any]:
        """Gate one command. Returns decision + laws + reason (+ receipt)."""
        action = cmd.get("action")
        laws: list[int] = [6]
        verdict = "BLOCKED"
        reason = "unverified: fail closed (Law 6)"
        if not isinstance(action, str) or not action:
            reason = "malformed command: no action (Law 6/9)"
        elif action == "estop":
            verdict, reason = "ALLOWED", "emergency stop always allowed (Law 1)"
            laws = [1]
        elif action in HUMAN_ONLY_ACTIONS:
            reason = f"{action}: harm-class, human-only (Law 1/7)"
            laws = [1, 7]
        else:
            target = cmd.get("target")
            speed = cmd.get("speed", 0.0)
            if not isinstance(target, (list, tuple)) or len(target) != len(self.envelope.min):
                reason = "malformed target: wrong dimension (Law 6/9)"
            elif not all(isinstance(v, (int, float)) for v in target):
                reason = "non-numeric target (Law 6/9)"
            elif not isinstance(speed, (int, float)) or speed < 0:
                reason = "invalid speed (Law 6/9)"
            elif speed > self.envelope.max_speed:
                reason = f"speed {speed} exceeds cap {self.envelope.max_speed} (Law 1)"
                laws = [1, 6]
            elif not self.envelope.contains(target):
                reason = "target outside workspace bounds (Law 1)"
                laws = [1, 6]
            elif self.envelope.in_forbidden(target):
                reason = "target inside forbidden zone (Law 1)"
                laws = [1, 6]
            else:
                verdict, reason = "ALLOWED", "within envelope, receipted"
        decision = openshell.admission_gate(action or "", verdict == "ALLOWED")
        receipt = None
        if self._secret is not None:
            receipt = khipu.mint(
                {"kind": "szl.motion-decision/v1", "command": cmd,
                 "decision": verdict, "laws": laws, "reason": reason},
                self._secret, prev_body_sha256=self._prev,
            )
            self._prev = receipt["body_sha256"]
        return {**decision, "decision": verdict, "laws": laws,
                "reason": reason, "receipt": receipt}

    def verify_log(self, receipts: list[dict[str, Any]], **kw: Any) -> bool:
        if self._secret is None:
            return False
        return khipu.check_chain(receipts, self._secret, **kw)

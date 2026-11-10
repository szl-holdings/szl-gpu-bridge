"""router — the two-axis Truth Router client.

Axis 1: measured honesty (census abstention + proving-ground badges).
Axis 2: cost (joules/token where MEASURED, price otherwise — never zero
for an unmeasured quantity).

Routing rule (Honesty Gate, from the ADAP-Khipu bridge): honesty-critical
tasks NEVER route to a model whose honesty state is UNAVAILABLE or FAILED,
even if it is the cheapest or most capable. Fail closed.

Inputs are trusted-caller assertions: this module validates their shape and
ranks them, but does not authenticate badges, provenance, or freshness. An
application must verify those before supplying model measurements here.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

HONESTY_STATES = ("MEASURED", "UNMEASURED", "FAILED")


def _valid_metric(value: object, maximum: float | None = None) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return (
            math.isfinite(value)
            and value >= 0
            and (maximum is None or value <= maximum)
        )
    except OverflowError:
        return False


@dataclass
class Model:
    model_id: str
    honesty_state: str  # HONESTY_STATES
    abstain_rate: float | None  # census: abstentions / total, None if unmeasured
    prove_badge: str | None  # proving-ground badge or None
    joules_per_token: float | None  # MEASURED only; never fabricated
    price_per_mtok: float | None

    def honesty_score(self) -> float | None:
        """None means UNKNOWN — never collapse to zero."""
        if self.honesty_state != "MEASURED" or not _valid_metric(self.abstain_rate, 1):
            return None
        return self.abstain_rate

    def cost_per_mtok(self) -> float | None:
        """Price axis only. Joules are a SEPARATE axis — never summed with
        dollars (units differ); the router reports both, prices on one."""
        return self.price_per_mtok if _valid_metric(self.price_per_mtok) else None

    def energy_per_mtok_joules(self) -> float | None:
        """Energy axis. None = UNAVAILABLE, never zero."""
        if self.honesty_state != "MEASURED" or not _valid_metric(self.joules_per_token):
            return None  # an unmeasured model's energy is unknown, not zero
        energy = self.joules_per_token * 1e6
        return energy if math.isfinite(energy) else None

    def metrics_valid(self) -> bool:
        """Absent metrics remain unknown; malformed measurements are ineligible."""
        return self.honesty_state in HONESTY_STATES and all(
            value is None or _valid_metric(value, maximum)
            for value, maximum in (
                (self.abstain_rate, 1),
                (self.price_per_mtok, None),
                (self.joules_per_token, None),
            )
        )


def route(task_kind: str, models: list[Model]) -> Model | None:
    """Pick the honest cheapest on the PRICE axis; report joules separately.
    Honesty-critical tasks fail closed. Ties break toward lower MEASURED
    energy; UNMEASURED energy never counts as zero."""
    valid_models = [
        m for m in models if m.metrics_valid() and m.honesty_state != "FAILED"
    ]
    eligible = [m for m in valid_models if m.honesty_score() is not None]
    if task_kind == "honesty-critical" and not eligible:
        return None  # fail closed: no honest model, no routing
    pool = eligible or valid_models
    priced = [m for m in pool if m.cost_per_mtok() is not None]
    if not priced:
        return None

    def sort_key(m: Model):
        energy = m.energy_per_mtok_joules()
        # measured energy (even zero) sorts BEFORE unknown: known > unknown
        # a model that PROVED zero joules is more honest than one that didn't measure
        return (
            m.cost_per_mtok(),
            1 if energy is None else 0,
            energy if energy is not None else float("inf"),
        )

    return min(priced, key=sort_key)

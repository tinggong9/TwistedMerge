"""Conservative period/index rank gates for certified projective classes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


GROUP_COHOMOLOGY_RANKS = (1, 2, 3, 4, 6, 8, 9, 12, 16, 24, 32)


@dataclass(frozen=True)
class PeriodIndexEstimate:
    class_status: str
    period: int | None
    index: int | None
    index_upper_bound: int | None
    index_status: str
    lift_implemented: bool = False


def estimate_period_index_from_class(
    class_status: str,
    period: int | None,
    group_order: int,
    coefficient_order: int,
    index_hint: int | None = None,
    index_status: str = "not_certified",
) -> PeriodIndexEstimate:
    """Return a claim-safe period/index estimate for one H^2 class."""

    status = str(class_status)
    if status in {"no_certified_class", "not_central_or_not_projectable"}:
        return PeriodIndexEstimate(status, None, None, None, "no_certified_class")
    if status == "coboundary":
        return PeriodIndexEstimate(status, 1, 1, 1, "coboundary")
    if status != "nontrivial_H2_class" or period is None or int(period) <= 0:
        return PeriodIndexEstimate(status, None, None, None, "unknown_no_lift")

    period_value = int(period)
    upper = max(period_value, int(group_order) * max(1, int(coefficient_order)))
    if index_hint is None:
        return PeriodIndexEstimate(
            status,
            period_value,
            None,
            upper,
            "index_unknown_no_lift" if index_status == "not_certified" else str(index_status),
        )
    index_value = int(index_hint)
    if index_value <= 0:
        return PeriodIndexEstimate(status, period_value, None, upper, "index_unknown_no_lift")
    if index_value % period_value != 0:
        return PeriodIndexEstimate(status, period_value, None, upper, "invalid_index_not_period_multiple")
    return PeriodIndexEstimate(status, period_value, index_value, upper, str(index_status))


def classify_group_cohomology_rank(
    class_status: str,
    period: int | None,
    index: int | None,
    index_status: str,
    rank: int,
) -> str:
    """Classify a candidate rank without weakening period/index gates."""

    status = str(class_status)
    if status in {"no_certified_class", "not_central_or_not_projectable"}:
        return "no_certified_class"
    if status == "coboundary":
        return "coboundary_no_lift_needed"
    if period is None or int(period) <= 0:
        return "no_certified_class"
    if int(rank) % int(period) != 0:
        return "rank_not_period_divisible"
    if index is None or int(index) <= 0 or str(index_status).startswith("index_unknown"):
        return "index_unknown_no_lift"
    if int(rank) % int(index) != 0:
        return "period_divisible_index_obstructed"
    return "index_divisible_lift_allowed"


def period_index_rank_rows(
    candidate: dict,
    ranks: Iterable[int] = GROUP_COHOMOLOGY_RANKS,
) -> list[dict]:
    rows = []
    for rank in ranks:
        decision = classify_group_cohomology_rank(
            candidate.get("class_status", "no_certified_class"),
            candidate.get("estimated_period"),
            candidate.get("estimated_index"),
            candidate.get("index_status", "not_certified"),
            int(rank),
        )
        rows.append(
            {
                **candidate,
                "candidate_rank": int(rank),
                "rank_decision": decision,
                "lift_allowed_by_index": decision == "index_divisible_lift_allowed",
            }
        )
    return rows

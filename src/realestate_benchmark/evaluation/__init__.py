"""Evaluation metrics and benchmark reporting."""

from realestate_benchmark.evaluation.p1_informational import (
    compute_p1_commission,
    compute_p1_framing,
    compute_p1_omission,
)
from realestate_benchmark.evaluation.p2_welfare import (
    compute_p2_nbs_deviation,
    compute_p2_welfare_gap,
)
from realestate_benchmark.evaluation.p3_influence import compute_p3_influence_legitimacy
from realestate_benchmark.evaluation.p4_norms import (
    check_n1_no_fraud,
    check_n2_respond_to_inquiry,
    check_n3_formal_disclosure,
    check_n4_no_fabricated_urgency,
    check_n5_fair_dealing,
    compute_p4_norm_compliance,
)
from realestate_benchmark.evaluation.report import (
    generate_benchmark_report,
    generate_game_report,
)
from realestate_benchmark.evaluation.sai import (
    compute_game_metrics,
    compute_sai,
    compute_sai_full_report,
)

__all__ = [
    "compute_p1_commission",
    "compute_p1_framing",
    "compute_p1_omission",
    "compute_p2_nbs_deviation",
    "compute_p2_welfare_gap",
    "compute_p3_influence_legitimacy",
    "check_n1_no_fraud",
    "check_n2_respond_to_inquiry",
    "check_n3_formal_disclosure",
    "check_n4_no_fabricated_urgency",
    "check_n5_fair_dealing",
    "compute_p4_norm_compliance",
    "compute_game_metrics",
    "compute_sai",
    "compute_sai_full_report",
    "generate_benchmark_report",
    "generate_game_report",
]

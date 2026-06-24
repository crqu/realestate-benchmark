"""Evaluation metrics and benchmark reporting."""

from realestate_benchmark.evaluation.llm_judge import (
    compute_llm_metrics,
    llm_p1_commission,
    llm_p1_framing,
    llm_p1_omission,
    llm_p3_influence_legitimacy,
    llm_p4_norm_compliance,
)
from realestate_benchmark.evaluation.negotiation_dynamics import (
    compute_anchoring_effect,
    compute_concession_analysis,
    compute_negotiation_dynamics,
    compute_negotiation_efficiency,
    compute_price_convergence,
    compute_zopa_utilization,
    extract_offer_trajectory,
)
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
from realestate_benchmark.evaluation.reasoning import (
    compute_reasoning_profile,
    llm_analyze_reasoning,
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
    "compute_anchoring_effect",
    "compute_concession_analysis",
    "compute_game_metrics",
    "compute_llm_metrics",
    "compute_negotiation_dynamics",
    "compute_negotiation_efficiency",
    "compute_p1_commission",
    "compute_p1_framing",
    "compute_p1_omission",
    "compute_p2_nbs_deviation",
    "compute_p2_welfare_gap",
    "compute_p3_influence_legitimacy",
    "compute_p4_norm_compliance",
    "compute_price_convergence",
    "compute_reasoning_profile",
    "compute_sai",
    "compute_sai_full_report",
    "compute_zopa_utilization",
    "check_n1_no_fraud",
    "check_n2_respond_to_inquiry",
    "check_n3_formal_disclosure",
    "check_n4_no_fabricated_urgency",
    "check_n5_fair_dealing",
    "extract_offer_trajectory",
    "generate_benchmark_report",
    "generate_game_report",
    "llm_analyze_reasoning",
    "llm_p1_commission",
    "llm_p1_framing",
    "llm_p1_omission",
    "llm_p3_influence_legitimacy",
    "llm_p4_norm_compliance",
]

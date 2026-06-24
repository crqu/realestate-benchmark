"""Tests for negotiation dynamics analysis."""

import pytest

from realestate_benchmark.environment.state import (
    GamePhase,
    GameState,
    Message,
    Offer,
    TransactionOutcome,
    TranscriptEntry,
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


def _make_offer(party: str, amount: int, turn: int, status: str = "superseded") -> Offer:
    return Offer(from_party=party, amount=amount, turn=turn, status=status)


def _make_message(sender: str, content: str, turn: int) -> Message:
    return Message(sender=sender, content=content, turn=turn)


def _make_transcript_entry(
    agent_id: str, tool_name: str, turn: int, game_id: str = "test-game"
) -> TranscriptEntry:
    return TranscriptEntry(
        entry_id=f"entry-{turn}-{agent_id}",
        game_id=game_id,
        agent_id=agent_id,
        turn=turn,
        phase=GamePhase.ACTIVE,
        tool_name=tool_name,
        parameters={},
        result={},
        reasoning_trace="",
    )


def _make_state(
    offers: list[Offer] | None = None,
    current_offer: Offer | None = None,
    asking_price: int = 200000,
    outcome: TransactionOutcome | None = None,
    messages: list[Message] | None = None,
    turn_number: int = 0,
) -> GameState:
    return GameState(
        game_id="test-game",
        turn_number=turn_number,
        phase=GamePhase.ACTIVE,
        property_id="prop-1",
        asking_price=asking_price,
        offers=offers or [],
        current_offer=current_offer,
        outcome=outcome,
        messages=messages or [],
    )


class TestExtractOfferTrajectory:
    def test_no_offers(self):
        state = _make_state()
        assert extract_offer_trajectory(state) == []

    def test_only_current_offer(self):
        current = _make_offer("buyer", 180000, 3, status="pending")
        state = _make_state(current_offer=current)
        result = extract_offer_trajectory(state)
        assert len(result) == 1
        assert result[0] == {
            "from_party": "buyer",
            "amount": 180000,
            "turn": 3,
            "status": "pending",
        }

    def test_historical_and_current(self):
        offers = [
            _make_offer("seller", 200000, 1),
            _make_offer("buyer", 170000, 2),
        ]
        current = _make_offer("seller", 190000, 3, status="pending")
        state = _make_state(offers=offers, current_offer=current)
        result = extract_offer_trajectory(state)
        assert len(result) == 3
        assert [r["turn"] for r in result] == [1, 2, 3]
        assert [r["from_party"] for r in result] == ["seller", "buyer", "seller"]

    def test_sorted_by_turn(self):
        offers = [
            _make_offer("buyer", 170000, 4),
            _make_offer("seller", 200000, 1),
        ]
        state = _make_state(offers=offers)
        result = extract_offer_trajectory(state)
        assert result[0]["turn"] == 1
        assert result[1]["turn"] == 4


class TestComputeConcessionAnalysis:
    def test_no_offers(self):
        state = _make_state()
        result = compute_concession_analysis(state)
        assert result["seller"]["num_concessions"] == 0
        assert result["buyer"]["num_concessions"] == 0
        assert result["reciprocity_index"] == 0.0

    def test_single_offer(self):
        state = _make_state(current_offer=_make_offer("seller", 200000, 1, status="pending"))
        result = compute_concession_analysis(state)
        assert result["seller"]["num_concessions"] == 0

    def test_seller_concession(self):
        offers = [
            _make_offer("seller", 200000, 1),
            _make_offer("buyer", 170000, 2),
            _make_offer("seller", 190000, 3),
        ]
        state = _make_state(offers=offers)
        result = compute_concession_analysis(state)
        assert result["seller"]["num_concessions"] == 1
        assert result["seller"]["total_concession"] == 10000
        assert result["seller"]["concessions"][0]["size"] == 10000

    def test_buyer_concession(self):
        offers = [
            _make_offer("buyer", 170000, 1),
            _make_offer("seller", 200000, 2),
            _make_offer("buyer", 180000, 3),
        ]
        state = _make_state(offers=offers)
        result = compute_concession_analysis(state)
        assert result["buyer"]["num_concessions"] == 1
        assert result["buyer"]["total_concession"] == 10000

    def test_no_concession_when_price_increases_for_seller(self):
        offers = [
            _make_offer("seller", 190000, 1),
            _make_offer("seller", 200000, 3),
        ]
        state = _make_state(offers=offers)
        result = compute_concession_analysis(state)
        assert result["seller"]["num_concessions"] == 0

    def test_symmetric_concessions_reciprocity(self):
        offers = [
            _make_offer("seller", 200000, 1),
            _make_offer("buyer", 160000, 2),
            _make_offer("seller", 190000, 3),
            _make_offer("buyer", 170000, 4),
        ]
        state = _make_state(offers=offers)
        result = compute_concession_analysis(state)
        assert result["reciprocity_index"] == 1.0

    def test_asymmetric_concessions_reciprocity(self):
        offers = [
            _make_offer("seller", 200000, 1),
            _make_offer("buyer", 160000, 2),
            _make_offer("seller", 190000, 3),
            _make_offer("buyer", 162000, 4),
        ]
        state = _make_state(offers=offers)
        result = compute_concession_analysis(state)
        assert 0 < result["reciprocity_index"] < 1.0

    def test_one_sided_concessions(self):
        offers = [
            _make_offer("seller", 200000, 1),
            _make_offer("seller", 190000, 3),
        ]
        state = _make_state(offers=offers)
        result = compute_concession_analysis(state)
        assert result["seller"]["num_concessions"] == 1
        assert result["buyer"]["num_concessions"] == 0
        assert result["reciprocity_index"] == 0.0


class TestComputeAnchoringEffect:
    def test_no_offers(self):
        state = _make_state()
        result = compute_anchoring_effect(state)
        assert result["first_offer_party"] is None
        assert result["first_offer_amount"] is None
        assert result["anchor_ratio"] is None
        assert result["asking_price_anchor"] is None

    def test_no_sale(self):
        state = _make_state(
            offers=[_make_offer("seller", 200000, 1)],
            outcome=TransactionOutcome(completed=False, final_price=0),
        )
        result = compute_anchoring_effect(state)
        assert result["first_offer_amount"] == 200000
        assert result["final_price"] is None
        assert result["anchor_ratio"] is None

    def test_completed_sale(self):
        state = _make_state(
            offers=[_make_offer("seller", 200000, 1)],
            current_offer=_make_offer("buyer", 190000, 3, status="accepted"),
            outcome=TransactionOutcome(completed=True, final_price=190000),
        )
        result = compute_anchoring_effect(state)
        assert result["first_offer_party"] == "seller"
        assert result["first_offer_amount"] == 200000
        assert result["final_price"] == 190000
        assert result["anchor_ratio"] == pytest.approx(0.95)
        assert result["asking_price_anchor"] == pytest.approx(0.95)


class TestComputePriceConvergence:
    def test_no_offers(self):
        state = _make_state()
        result = compute_price_convergence(state)
        assert result["spread_trajectory"] == []
        assert result["initial_spread"] is None
        assert result["final_spread"] is None
        assert result["convergence_rate"] == 0.0

    def test_single_party_offers(self):
        state = _make_state(
            offers=[_make_offer("seller", 200000, 1), _make_offer("seller", 195000, 3)]
        )
        result = compute_price_convergence(state)
        assert result["spread_trajectory"] == []

    def test_convergence(self):
        offers = [
            _make_offer("seller", 200000, 1),
            _make_offer("buyer", 160000, 2),
            _make_offer("seller", 190000, 3),
            _make_offer("buyer", 180000, 4),
        ]
        state = _make_state(offers=offers)
        result = compute_price_convergence(state)
        assert len(result["spread_trajectory"]) == 3
        assert result["initial_spread"] == 40000
        assert result["final_spread"] == 10000
        assert result["convergence_rate"] == pytest.approx(15000.0)

    def test_no_convergence_same_turn(self):
        offers = [
            _make_offer("seller", 200000, 1),
            _make_offer("buyer", 160000, 1),
        ]
        state = _make_state(offers=offers)
        result = compute_price_convergence(state)
        assert result["convergence_rate"] == 0.0


class TestComputeZopaUtilization:
    def test_no_sale(self):
        state = _make_state()
        result = compute_zopa_utilization(state, true_value=180000.0)
        assert result["zopa_lower"] == 180000.0
        assert result["zopa_upper"] == 200000
        assert result["zopa_range"] == 20000.0
        assert result["final_price"] is None
        assert result["zopa_position"] is None
        assert result["within_zopa"] is None

    def test_midpoint_sale(self):
        state = _make_state(outcome=TransactionOutcome(completed=True, final_price=190000))
        result = compute_zopa_utilization(state, true_value=180000.0)
        assert result["zopa_position"] == pytest.approx(0.5)
        assert result["within_zopa"] is True

    def test_at_true_value(self):
        state = _make_state(outcome=TransactionOutcome(completed=True, final_price=180000))
        result = compute_zopa_utilization(state, true_value=180000.0)
        assert result["zopa_position"] == pytest.approx(0.0)
        assert result["within_zopa"] is True

    def test_at_asking_price(self):
        state = _make_state(outcome=TransactionOutcome(completed=True, final_price=200000))
        result = compute_zopa_utilization(state, true_value=180000.0)
        assert result["zopa_position"] == pytest.approx(1.0)
        assert result["within_zopa"] is True

    def test_below_true_value(self):
        state = _make_state(outcome=TransactionOutcome(completed=True, final_price=170000))
        result = compute_zopa_utilization(state, true_value=180000.0)
        assert result["zopa_position"] == pytest.approx(-0.5)
        assert result["within_zopa"] is False

    def test_above_asking_price(self):
        state = _make_state(outcome=TransactionOutcome(completed=True, final_price=210000))
        result = compute_zopa_utilization(state, true_value=180000.0)
        assert result["zopa_position"] == pytest.approx(1.5)
        assert result["within_zopa"] is False

    def test_zero_range_zopa(self):
        state = _make_state(
            asking_price=180000,
            outcome=TransactionOutcome(completed=True, final_price=180000),
        )
        result = compute_zopa_utilization(state, true_value=180000.0)
        assert result["zopa_position"] == pytest.approx(0.0)


class TestComputeNegotiationEfficiency:
    def test_no_offers_no_messages(self):
        state = _make_state(turn_number=5)
        result = compute_negotiation_efficiency(state, [])
        assert result["total_turns"] == 5
        assert result["total_messages"] == 0
        assert result["messages_before_first_offer"] == 0
        assert result["turns_before_first_offer"] == 0
        assert result["offer_counteroffer_rounds"] == 0
        assert result["completed"] is False
        assert result["turns_to_agreement"] is None

    def test_messages_before_offer(self):
        messages = [
            _make_message("seller", "Hello", 1),
            _make_message("buyer", "Hi", 1),
            _make_message("seller", "Here's the listing", 2),
        ]
        offers = [_make_offer("seller", 200000, 3)]
        state = _make_state(offers=offers, messages=messages, turn_number=5)
        result = compute_negotiation_efficiency(state, [])
        assert result["total_messages"] == 3
        assert result["messages_before_first_offer"] == 3
        assert result["turns_before_first_offer"] == 3

    def test_offer_counteroffer_rounds(self):
        offers = [
            _make_offer("seller", 200000, 1),
            _make_offer("buyer", 170000, 2),
            _make_offer("seller", 190000, 3),
            _make_offer("buyer", 185000, 4),
        ]
        state = _make_state(offers=offers, turn_number=4)
        result = compute_negotiation_efficiency(state, [])
        assert result["offer_counteroffer_rounds"] == 3

    def test_same_party_consecutive_no_round(self):
        offers = [
            _make_offer("seller", 200000, 1),
            _make_offer("seller", 195000, 2),
            _make_offer("buyer", 180000, 3),
        ]
        state = _make_state(offers=offers, turn_number=3)
        result = compute_negotiation_efficiency(state, [])
        assert result["offer_counteroffer_rounds"] == 1

    def test_completed_with_agreement(self):
        offers = [_make_offer("seller", 200000, 1)]
        current = _make_offer("buyer", 190000, 3, status="accepted")
        state = _make_state(
            offers=offers,
            current_offer=current,
            outcome=TransactionOutcome(completed=True, final_price=190000),
            turn_number=3,
        )
        result = compute_negotiation_efficiency(state, [])
        assert result["completed"] is True
        assert result["turns_to_agreement"] == 3


class TestComputeNegotiationDynamics:
    def test_full_negotiation(self):
        messages = [
            _make_message("seller", "Welcome", 0),
            _make_message("buyer", "Interested", 0),
        ]
        offers = [
            _make_offer("seller", 200000, 1),
            _make_offer("buyer", 170000, 2),
            _make_offer("seller", 190000, 3),
        ]
        current = _make_offer("buyer", 185000, 4, status="accepted")
        state = _make_state(
            offers=offers,
            current_offer=current,
            messages=messages,
            outcome=TransactionOutcome(completed=True, final_price=185000),
            turn_number=4,
        )
        transcript = [_make_transcript_entry("seller", "send_message", 0)]
        result = compute_negotiation_dynamics(state, transcript, true_value=180000.0)

        assert len(result["offer_trajectory"]) == 4
        assert result["concession_seller_total"] == 10000
        assert result["concession_buyer_total"] == 15000
        assert 0 < result["concession_reciprocity"] <= 1.0
        assert result["anchor_ratio"] == pytest.approx(185000 / 200000)
        assert result["convergence_rate"] > 0
        assert result["zopa_position"] == pytest.approx(0.25)
        assert result["efficiency_turns_to_agreement"] == 4
        assert result["efficiency_messages_before_first_offer"] == 2
        assert result["efficiency_offer_rounds"] == 3
        assert "details" in result
        assert "concessions" in result["details"]
        assert "anchoring" in result["details"]
        assert "convergence" in result["details"]
        assert "zopa" in result["details"]
        assert "efficiency" in result["details"]

    def test_no_offers_dynamics(self):
        state = _make_state(turn_number=2)
        result = compute_negotiation_dynamics(state, [], true_value=180000.0)
        assert result["offer_trajectory"] == []
        assert result["concession_seller_total"] == 0
        assert result["concession_buyer_total"] == 0
        assert result["anchor_ratio"] is None
        assert result["zopa_position"] is None

    def test_no_sale(self):
        offers = [
            _make_offer("seller", 200000, 1),
            _make_offer("buyer", 150000, 2),
        ]
        state = _make_state(
            offers=offers,
            outcome=TransactionOutcome(completed=False, final_price=0),
            turn_number=5,
        )
        result = compute_negotiation_dynamics(state, [], true_value=180000.0)
        assert result["efficiency_turns_to_agreement"] is None
        assert result["zopa_position"] is None

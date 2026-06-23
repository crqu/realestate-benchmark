"""Tests for simplified game state models."""


from realestate_benchmark.environment.state import (
    GamePhase,
    GameState,
    Offer,
    TranscriptEntry,
)


class TestGamePhase:
    def test_has_active(self):
        assert GamePhase.ACTIVE == "active"

    def test_has_closed(self):
        assert GamePhase.CLOSED == "closed"

    def test_has_terminated(self):
        assert GamePhase.TERMINATED == "terminated"

    def test_only_three_values(self):
        assert len(GamePhase) == 3


class TestOffer:
    def test_from_party_buyer(self):
        offer = Offer(from_party="buyer", amount=100000, turn=1)
        assert offer.from_party == "buyer"
        assert offer.amount == 100000
        assert offer.turn == 1
        assert offer.status == "pending"

    def test_from_party_seller(self):
        offer = Offer(from_party="seller", amount=120000, turn=2)
        assert offer.from_party == "seller"

    def test_no_contingencies_field(self):
        assert "contingencies" not in Offer.model_fields


class TestGameState:
    def test_defaults_to_active(self):
        state = GameState(game_id="test", property_id="prop_1", asking_price=200000)
        assert state.phase == GamePhase.ACTIVE

    def test_has_property_id(self):
        state = GameState(game_id="test", property_id="prop_1", asking_price=200000)
        assert state.property_id == "prop_1"

    def test_has_asking_price(self):
        state = GameState(game_id="test", property_id="prop_1", asking_price=150000)
        assert state.asking_price == 150000

    def test_no_listing_field(self):
        assert "listing" not in GameState.model_fields

    def test_no_disclosures_field(self):
        assert "disclosures" not in GameState.model_fields

    def test_no_inspection_report_field(self):
        assert "inspection_report" not in GameState.model_fields

    def test_no_market_data_field(self):
        assert "market_data" not in GameState.model_fields

    def test_add_message(self):
        state = GameState(game_id="test", property_id="prop_1", asking_price=200000)
        state.add_message("seller", "Hello buyer")
        assert len(state.messages) == 1
        assert state.messages[0].sender == "seller"
        assert state.messages[0].content == "Hello buyer"
        assert state.messages[0].turn == 0

    def test_transition_phase_to_closed(self):
        state = GameState(game_id="test", property_id="prop_1", asking_price=200000)
        state.transition_phase(GamePhase.CLOSED)
        assert state.phase == GamePhase.CLOSED

    def test_transition_phase_to_terminated(self):
        state = GameState(game_id="test", property_id="prop_1", asking_price=200000)
        state.transition_phase(GamePhase.TERMINATED)
        assert state.phase == GamePhase.TERMINATED

    def test_empty_messages_by_default(self):
        state = GameState(game_id="test", property_id="prop_1", asking_price=200000)
        assert state.messages == []

    def test_empty_offers_by_default(self):
        state = GameState(game_id="test", property_id="prop_1", asking_price=200000)
        assert state.offers == []
        assert state.current_offer is None

    def test_no_outcome_by_default(self):
        state = GameState(game_id="test", property_id="prop_1", asking_price=200000)
        assert state.outcome is None


class TestTranscriptEntry:
    def test_with_active_phase(self):
        entry = TranscriptEntry(
            entry_id="e1",
            game_id="g1",
            agent_id="seller",
            turn=0,
            phase=GamePhase.ACTIVE,
            tool_name="send_message",
            parameters={"content": "hello"},
            result={"success": True},
            reasoning_trace="thinking...",
        )
        assert entry.phase == GamePhase.ACTIVE

    def test_with_closed_phase(self):
        entry = TranscriptEntry(
            entry_id="e1",
            game_id="g1",
            agent_id="buyer",
            turn=5,
            phase=GamePhase.CLOSED,
            tool_name="accept_offer",
            parameters={},
            result={"success": True},
            reasoning_trace="accepting",
        )
        assert entry.phase == GamePhase.CLOSED

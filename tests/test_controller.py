"""Tests for GameController with simplified 3-phase design."""

from unittest.mock import Mock

from realestate_benchmark.environment.controller import GameController
from realestate_benchmark.environment.state import GamePhase, Offer
from realestate_benchmark.tools import create_registry


class MockAgent:
    """Agent that returns predetermined actions."""

    def __init__(self, actions):
        self.actions = iter(actions)
        self.tools = create_registry()
        self.memory = Mock()
        self.memory.read.return_value = ""

    def act(self, state, context):
        return next(self.actions)


def make_controller(seller_actions, buyer_actions, max_turns=20, buyer_budget=200000):
    seller = MockAgent(seller_actions)
    buyer = MockAgent(buyer_actions)
    db = Mock()
    db.create_game.return_value = "test-game-id"
    config = {
        "max_turns": max_turns,
        "property_id": "prop-1",
        "asking_price": 300000,
        "buyer_budget": buyer_budget,
    }
    property_data = {"id": "prop-1", "price": 300000}
    defects = [{"name": "roof", "cost": 15000}]
    controller = GameController(
        seller=seller, buyer=buyer, db=db, config=config,
        property_data=property_data, defects=defects,
    )
    return controller


class TestControllerInit:
    def test_controller_starts_active(self):
        controller = make_controller([], [])
        controller.initialize()
        assert controller.state is not None
        assert controller.state.phase == GamePhase.ACTIVE

    def test_initialize_sets_property_fields(self):
        controller = make_controller([], [])
        game_id = controller.initialize()
        assert game_id == "test-game-id"
        assert controller.state.property_id == "prop-1"
        assert controller.state.asking_price == 300000


class TestPhaseTransitions:
    def test_accept_offer_transitions_to_closed(self):
        seller_actions = [
            ("make_offer", {"amount": 250000}, "offering"),
        ]
        buyer_actions = [
            ("accept_offer", {}, "accepting"),
        ]
        controller = make_controller(seller_actions, buyer_actions)
        controller.initialize()
        outcome = controller.run()
        assert controller.state.phase == GamePhase.CLOSED
        assert outcome.completed is True
        assert outcome.final_price == 250000

    def test_walk_away_transitions_to_terminated(self):
        seller_actions = [
            ("send_message", {"content": "hello"}, "greeting"),
        ]
        buyer_actions = [
            ("walk_away", {}, "leaving"),
        ]
        controller = make_controller(seller_actions, buyer_actions)
        controller.initialize()
        outcome = controller.run()
        assert controller.state.phase == GamePhase.TERMINATED
        assert outcome.completed is False

    def test_max_turns_terminates(self):
        actions = [("send_message", {"content": "hello"}, "chatting")] * 5
        controller = make_controller(actions, actions, max_turns=4)
        controller.initialize()
        outcome = controller.run()
        assert controller.state.phase == GamePhase.TERMINATED
        assert outcome.completed is False


class TestPassAction:
    def test_pass_action_advances_turn(self):
        seller_actions = [
            ("pass", {}, "thinking aloud"),
            ("send_message", {"content": "ok"}, "responding"),
        ]
        buyer_actions = [
            ("send_message", {"content": "hi"}, "greeting"),
        ]
        controller = make_controller(seller_actions, buyer_actions, max_turns=3)
        controller.initialize()
        controller.run()
        assert controller.state.turn_number >= 2
        assert controller.state.phase in (GamePhase.ACTIVE, GamePhase.TERMINATED)


class TestToolAccess:
    def test_both_agents_see_same_tools(self):
        controller = make_controller([], [])
        seller_tools = set(controller.seller.tools.tools.keys())
        buyer_tools = set(controller.buyer.tools.tools.keys())
        assert seller_tools == buyer_tools
        assert seller_tools == {"send_message", "make_offer", "accept_offer", "walk_away"}


class TestActionValidation:
    def test_no_actions_in_closed_phase(self):
        controller = make_controller([], [])
        controller.initialize()
        controller.state.phase = GamePhase.CLOSED
        assert controller.validate_action("seller", "send_message", {}) is False
        assert controller.validate_action("buyer", "make_offer", {"amount": 100}) is False

    def test_no_actions_in_terminated_phase(self):
        controller = make_controller([], [])
        controller.initialize()
        controller.state.phase = GamePhase.TERMINATED
        assert controller.validate_action("seller", "send_message", {}) is False
        assert controller.validate_action("buyer", "walk_away", {}) is False

    def test_accept_own_offer_invalid(self):
        controller = make_controller([], [])
        controller.initialize()
        controller.state.current_offer = Offer(
            from_party="seller", amount=200000, turn=0,
        )
        assert controller.validate_action("seller", "accept_offer", {}) is False
        assert controller.validate_action("buyer", "accept_offer", {}) is True

    def test_accept_offer_requires_current_offer(self):
        controller = make_controller([], [])
        controller.initialize()
        assert controller.state.current_offer is None
        assert controller.validate_action("buyer", "accept_offer", {}) is False

    def test_all_tools_valid_in_active(self):
        controller = make_controller([], [])
        controller.initialize()
        for tool in ("send_message", "make_offer", "walk_away"):
            assert controller.validate_action("seller", tool, {}) is True
            assert controller.validate_action("buyer", tool, {}) is True


class TestAlternatingTurns:
    def test_alternating_turns(self):
        """Seller acts on even turns, buyer on odd turns."""
        turn_order = []

        class TrackingAgent(MockAgent):
            def __init__(self, name, actions):
                super().__init__(actions)
                self.name = name

            def act(self, state, context):
                turn_order.append((self.name, state.turn_number))
                return super().act(state, context)

        seller = TrackingAgent("seller", [
            ("send_message", {"content": "hi"}, ""),
            ("send_message", {"content": "ok"}, ""),
        ])
        buyer = TrackingAgent("buyer", [
            ("send_message", {"content": "hello"}, ""),
            ("walk_away", {}, ""),
        ])
        db = Mock()
        db.create_game.return_value = "test-game-id"
        config = {
            "max_turns": 20,
            "property_id": "prop-1",
            "asking_price": 300000,
            "buyer_budget": 200000,
        }
        controller = GameController(
            seller=seller, buyer=buyer, db=db, config=config,
            property_data={"id": "prop-1"}, defects=[],
        )
        controller.initialize()
        controller.run()

        assert turn_order[0] == ("seller", 0)
        assert turn_order[1] == ("buyer", 1)
        assert turn_order[2] == ("seller", 2)
        assert turn_order[3] == ("buyer", 3)

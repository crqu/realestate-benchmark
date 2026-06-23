"""Tests for simplified negotiation tools."""

import pytest

from realestate_benchmark.environment.state import GameState
from realestate_benchmark.tools.registry import create_registry


@pytest.fixture
def state():
    return GameState(game_id="test_game", property_id="prop_1", asking_price=200000)


@pytest.fixture
def registry():
    return create_registry()


@pytest.fixture
def seller_context(state):
    return {"state": state, "agent_id": "seller"}


@pytest.fixture
def buyer_context(state):
    return {"state": state, "agent_id": "buyer"}


class TestRegistry:
    def test_has_four_tools(self, registry):
        tools = registry.get_all_definitions()
        assert len(tools) == 4

    def test_tool_names(self, registry):
        names = {t.name for t in registry.get_all_definitions()}
        assert names == {"send_message", "make_offer", "accept_offer", "walk_away"}


class TestSendMessage:
    def test_seller_sends(self, registry, seller_context):
        result = registry.execute("send_message", {"content": "Hello buyer"}, seller_context)
        assert result["success"] is True
        state = seller_context["state"]
        assert len(state.messages) == 1
        assert state.messages[0].sender == "seller"
        assert state.messages[0].content == "Hello buyer"

    def test_buyer_sends(self, registry, buyer_context):
        result = registry.execute("send_message", {"content": "Hello seller"}, buyer_context)
        assert result["success"] is True
        state = buyer_context["state"]
        assert len(state.messages) == 1
        assert state.messages[0].sender == "buyer"


class TestMakeOffer:
    def test_buyer_makes_offer(self, registry, buyer_context):
        result = registry.execute("make_offer", {"amount": 180000}, buyer_context)
        assert result["success"] is True
        state = buyer_context["state"]
        assert state.current_offer is not None
        assert state.current_offer.from_party == "buyer"
        assert state.current_offer.amount == 180000
        assert state.current_offer.status == "pending"

    def test_seller_makes_counter(self, registry, seller_context):
        result = registry.execute("make_offer", {"amount": 195000}, seller_context)
        assert result["success"] is True
        state = seller_context["state"]
        assert state.current_offer.from_party == "seller"
        assert state.current_offer.amount == 195000

    def test_replaces_existing_offer(self, registry, buyer_context):
        state = buyer_context["state"]
        registry.execute("make_offer", {"amount": 180000}, buyer_context)
        assert state.current_offer.amount == 180000

        registry.execute("make_offer", {"amount": 185000}, buyer_context)
        assert state.current_offer.amount == 185000
        assert len(state.offers) == 1
        assert state.offers[0].amount == 180000
        assert state.offers[0].status == "superseded"


class TestAcceptOffer:
    def test_accept_counterpart_offer(self, registry, state):
        buyer_ctx = {"state": state, "agent_id": "buyer"}
        seller_ctx = {"state": state, "agent_id": "seller"}

        registry.execute("make_offer", {"amount": 190000}, buyer_ctx)
        result = registry.execute("accept_offer", {}, seller_ctx)

        assert result["success"] is True
        assert state.current_offer.status == "accepted"
        assert state.outcome is not None
        assert state.outcome.completed is True
        assert state.outcome.final_price == 190000

    def test_accept_own_offer_fails(self, registry, state):
        buyer_ctx = {"state": state, "agent_id": "buyer"}

        registry.execute("make_offer", {"amount": 190000}, buyer_ctx)
        result = registry.execute("accept_offer", {}, buyer_ctx)

        assert result["success"] is False
        assert state.outcome is None

    def test_accept_no_offer_fails(self, registry, buyer_context):
        result = registry.execute("accept_offer", {}, buyer_context)
        assert result["success"] is False

    def test_seller_accepts_buyer_offer(self, registry, state):
        buyer_ctx = {"state": state, "agent_id": "buyer"}
        seller_ctx = {"state": state, "agent_id": "seller"}

        registry.execute("make_offer", {"amount": 175000}, buyer_ctx)
        result = registry.execute("accept_offer", {}, seller_ctx)

        assert result["success"] is True
        assert state.outcome.final_price == 175000

    def test_buyer_accepts_seller_counter(self, registry, state):
        buyer_ctx = {"state": state, "agent_id": "buyer"}
        seller_ctx = {"state": state, "agent_id": "seller"}

        registry.execute("make_offer", {"amount": 175000}, buyer_ctx)
        registry.execute("make_offer", {"amount": 195000}, seller_ctx)
        result = registry.execute("accept_offer", {}, buyer_ctx)

        assert result["success"] is True
        assert state.outcome.final_price == 195000


class TestWalkAway:
    def test_buyer_walks(self, registry, buyer_context):
        result = registry.execute("walk_away", {}, buyer_context)
        assert result["success"] is True
        state = buyer_context["state"]
        assert state.outcome is not None
        assert state.outcome.completed is False
        assert state.outcome.final_price == 0
        assert result["walked_away_by"] == "buyer"

    def test_seller_walks(self, registry, seller_context):
        result = registry.execute("walk_away", {}, seller_context)
        assert result["success"] is True
        assert result["walked_away_by"] == "seller"
        state = seller_context["state"]
        assert state.outcome.completed is False

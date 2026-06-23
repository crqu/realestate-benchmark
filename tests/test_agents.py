"""Tests for simplified agent implementations."""

import pytest

from realestate_benchmark.agents.base import ReActAgent
from realestate_benchmark.agents.buyer import (
    BuyerAgent,
)
from realestate_benchmark.agents.seller import SellerAgent
from realestate_benchmark.data.properties import HIDDEN_FEATURES, Defect
from realestate_benchmark.environment.state import GameState, Offer
from realestate_benchmark.models.interface import ModelResponse, ToolCall
from realestate_benchmark.tools.registry import create_registry

# --- Fixtures ---


class MockModel:
    """Mock model for testing agent act() behavior."""

    def __init__(self, response: ModelResponse | None = None):
        self._response = response or ModelResponse(
            content="I'll wait.",
            tool_calls=[],
            usage={"input_tokens": 10, "output_tokens": 5},
            model_name="mock",
        )
        self._model_name = "mock"

    def generate(self, messages, tools=None, temperature=0.7):
        return self._response

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def supports_tool_use(self) -> bool:
        return True


class MockMemory:
    """Minimal memory stub for testing."""

    def __init__(self):
        self.content = "# Agent Memory\n"

    def read(self) -> str:
        return self.content

    def write(self, content: str) -> None:
        self.content = content

    def append(self, note: str) -> None:
        self.content += note + "\n"


SAMPLE_PROPERTY = {
    "HouseStyle": "2Story",
    "YearBuilt": 1990,
    "GrLivArea": 1800,
    "BedroomAbvGr": 3,
    "FullBath": 2,
    "TotalBsmtSF": 900,
    "BsmtFinSF1": 400,
    "GarageCars": 2,
    "GarageArea": 480,
    "GarageType": "Attchd",
    "SalePrice": 200000,
    "Neighborhood": "NAmes",
    "LotArea": 8000,
    "YearRemodAdd": 2005,
    "Fireplaces": 1,
    # Hidden features
    "OverallQual": 6,
    "OverallCond": 5,
    "BsmtQual": "TA",
    "BsmtCond": "Po",
    "BsmtExposure": "No",
    "HeatingQC": "Fa",
    "Electrical": "SBrkr",
    "CentralAir": "Y",
    "Functional": "Typ",
    "GarageQual": "TA",
    "GarageCond": "TA",
}

SAMPLE_DEFECTS = [
    Defect(
        feature="BsmtCond",
        value="Po",
        severity="major",
        repair_cost=8500,
        description="BsmtCond is Po",
    ),
    Defect(
        feature="HeatingQC",
        value="Fa",
        severity="moderate",
        repair_cost=2500,
        description="HeatingQC is Fa",
    ),
]


@pytest.fixture
def state():
    return GameState(game_id="test", property_id="prop_1", asking_price=200000)


@pytest.fixture
def registry():
    return create_registry()


@pytest.fixture
def seller(registry):
    return SellerAgent(
        model=MockModel(),
        property_data=SAMPLE_PROPERTY,
        defects=SAMPLE_DEFECTS,
        memory=MockMemory(),
        tools=registry,
    )


@pytest.fixture
def buyer_naive(registry):
    return BuyerAgent(
        model=MockModel(),
        budget=250000,
        memory=MockMemory(),
        tools=registry,
        sophistication="naive",
        alternative_price=180000,
    )


@pytest.fixture
def buyer_standard(registry):
    return BuyerAgent(
        model=MockModel(),
        budget=250000,
        memory=MockMemory(),
        tools=registry,
        sophistication="standard",
        alternative_price=180000,
    )


@pytest.fixture
def buyer_savvy(registry):
    return BuyerAgent(
        model=MockModel(),
        budget=250000,
        memory=MockMemory(),
        tools=registry,
        sophistication="savvy",
        alternative_price=180000,
    )


# --- Seller observation tests ---


class TestSellerObservation:
    def test_includes_defects(self, seller, state):
        obs = seller.observe(state)
        assert "BsmtCond" in obs
        assert "8,500" in obs
        assert "HeatingQC" in obs
        assert "2,500" in obs
        assert "major" in obs
        assert "moderate" in obs

    def test_shows_asking_price(self, seller, state):
        obs = seller.observe(state)
        assert "200,000" in obs

    def test_shows_active_phase(self, seller, state):
        obs = seller.observe(state)
        assert "ACTIVE" in obs

    def test_shows_hidden_features(self, seller, state):
        obs = seller.observe(state)
        for feat in HIDDEN_FEATURES:
            if feat in SAMPLE_PROPERTY:
                assert feat in obs

    def test_shows_public_features(self, seller, state):
        obs = seller.observe(state)
        assert "YearBuilt" in obs
        assert "1990" in obs

    def test_shows_all_messages(self, seller, state):
        for i in range(8):
            state.add_message("buyer" if i % 2 else "seller", f"Message {i}")
        obs = seller.observe(state)
        for i in range(8):
            assert f"Message {i}" in obs

    def test_shows_current_offer(self, seller, state):
        state.current_offer = Offer(from_party="buyer", amount=185000, status="pending", turn=3)
        obs = seller.observe(state)
        assert "185,000" in obs
        assert "pending" in obs

    def test_shows_turn_number(self, seller, state):
        state.turn_number = 7
        obs = seller.observe(state)
        assert "7" in obs

    def test_no_old_tool_references(self, seller, state):
        obs = seller.observe(state)
        for old_tool in [
            "list_property",
            "view_listing",
            "disclose_defect",
            "respond_to_offer",
            "ask_about_condition",
            "order_inspection",
            "get_market_data",
            "view_property_details",
        ]:
            assert old_tool not in obs

    def test_turn_warning_at_14(self, seller, state):
        state.turn_number = 14
        obs = seller.observe(state)
        assert "Warning" in obs
        assert "14" in obs

    def test_no_turn_warning_before_14(self, seller, state):
        state.turn_number = 13
        obs = seller.observe(state)
        assert "Warning" not in obs

    def test_no_old_phase_references(self, seller, state):
        obs = seller.observe(state)
        for old_phase in ["INIT", "LISTING", "DISCOVERY", "NEGOTIATION"]:
            assert old_phase not in obs


# --- Buyer observation tests ---


class TestBuyerObservation:
    def test_shows_asking_price(self, buyer_standard, state):
        obs = buyer_standard.observe(state)
        assert "200,000" in obs

    def test_shows_property_id(self, buyer_standard, state):
        obs = buyer_standard.observe(state)
        assert "prop_1" in obs

    def test_does_not_show_hidden_features(self, buyer_standard, state):
        obs = buyer_standard.observe(state)
        for feat in HIDDEN_FEATURES:
            assert feat not in obs

    def test_does_not_show_defects(self, buyer_standard, state):
        obs = buyer_standard.observe(state)
        assert "Defect" not in obs
        assert "repair_cost" not in obs

    def test_shows_messages(self, buyer_standard, state):
        state.add_message("seller", "Welcome to the listing")
        state.add_message("buyer", "What about the basement?")
        obs = buyer_standard.observe(state)
        assert "Welcome to the listing" in obs
        assert "What about the basement?" in obs

    def test_shows_all_messages(self, buyer_standard, state):
        for i in range(8):
            state.add_message("seller" if i % 2 else "buyer", f"Msg {i}")
        obs = buyer_standard.observe(state)
        for i in range(8):
            assert f"Msg {i}" in obs

    def test_shows_current_offer(self, buyer_standard, state):
        state.current_offer = Offer(from_party="seller", amount=195000, status="pending", turn=2)
        obs = buyer_standard.observe(state)
        assert "195,000" in obs
        assert "seller" in obs
        assert "pending" in obs

    def test_shows_outcome(self, buyer_standard, state):
        from realestate_benchmark.environment.state import TransactionOutcome

        state.outcome = TransactionOutcome(completed=True, final_price=190000)
        obs = buyer_standard.observe(state)
        assert "190,000" in obs

    def test_shows_turn_number(self, buyer_standard, state):
        state.turn_number = 5
        obs = buyer_standard.observe(state)
        assert "5" in obs

    def test_turn_warning_at_14(self, buyer_standard, state):
        state.turn_number = 14
        obs = buyer_standard.observe(state)
        assert "14" in obs

    def test_no_old_tool_references(self, buyer_standard, state):
        obs = buyer_standard.observe(state)
        for old_tool in [
            "view_listing",
            "ask_about_condition",
            "order_inspection",
            "get_market_data",
            "withdraw_offer",
        ]:
            assert old_tool not in obs

    def test_no_old_phase_references(self, buyer_standard, state):
        obs = buyer_standard.observe(state)
        for old_phase in ["INIT", "LISTING", "DISCOVERY", "NEGOTIATION"]:
            assert old_phase not in obs


# --- Buyer sophistication tests ---


class TestBuyerSophistication:
    def test_naive_prompt_differs_from_standard(self, buyer_naive, buyer_standard):
        assert buyer_naive.system_prompt != buyer_standard.system_prompt

    def test_standard_prompt_differs_from_savvy(self, buyer_standard, buyer_savvy):
        assert buyer_standard.system_prompt != buyer_savvy.system_prompt

    def test_naive_prompt_contains_budget(self, buyer_naive):
        assert "250,000" in buyer_naive.system_prompt

    def test_naive_prompt_contains_alternative(self, buyer_naive):
        assert "180,000" in buyer_naive.system_prompt

    def test_savvy_prompt_mentions_defects(self, buyer_savvy):
        assert "defect" in buyer_savvy.system_prompt.lower()

    def test_invalid_sophistication_raises(self, registry):
        with pytest.raises(ValueError, match="Invalid sophistication"):
            BuyerAgent(
                model=MockModel(),
                budget=250000,
                memory=MockMemory(),
                tools=registry,
                sophistication="expert",
            )

    def test_set_sophistication(self, buyer_naive):
        buyer_naive.set_sophistication("savvy")
        assert buyer_naive.sophistication == "savvy"
        assert "defect" in buyer_naive.system_prompt.lower()


# --- Base agent act() tests ---


class TestBaseAgentAct:
    def test_text_only_returns_pass(self, seller, state):
        tool_name, params, reasoning = seller.act(state, {})
        assert tool_name == "pass"
        assert params == {}
        assert reasoning == "I'll wait."

    def test_tool_call_returns_tool(self, registry):
        response = ModelResponse(
            content="Let me send a message.",
            tool_calls=[ToolCall(name="send_message", arguments={"content": "Hello"})],
            usage={"input_tokens": 10, "output_tokens": 5},
            model_name="mock",
        )
        agent = SellerAgent(
            model=MockModel(response),
            property_data=SAMPLE_PROPERTY,
            defects=SAMPLE_DEFECTS,
            memory=MockMemory(),
            tools=registry,
        )
        tool_name, params, reasoning = agent.act(state=GameState(
            game_id="test", property_id="prop_1", asking_price=200000
        ), context={})
        assert tool_name == "send_message"
        assert params == {"content": "Hello"}

    def test_no_blocked_tools_method(self):
        assert not hasattr(ReActAgent, "_blocked_tools")

    def test_all_four_tools_passed(self, registry):
        captured_tools = []

        class CapturingModel:
            @property
            def model_name(self):
                return "mock"

            @property
            def supports_tool_use(self):
                return True

            def generate(self, messages, tools=None, temperature=0.7):
                captured_tools.extend(tools)
                return ModelResponse(
                    content="thinking",
                    tool_calls=[],
                    usage={"input_tokens": 10, "output_tokens": 5},
                    model_name="mock",
                )

        agent = SellerAgent(
            model=CapturingModel(),
            property_data=SAMPLE_PROPERTY,
            defects=SAMPLE_DEFECTS,
            memory=MockMemory(),
            tools=registry,
        )
        agent.act(
            state=GameState(game_id="test", property_id="prop_1", asking_price=200000),
            context={},
        )
        tool_names = {t.name for t in captured_tools}
        assert tool_names == {"send_message", "make_offer", "accept_offer", "walk_away"}


# --- Seller constructor tests ---


class TestSellerConstructor:
    def test_default_days_on_market(self, registry):
        seller = SellerAgent(
            model=MockModel(),
            property_data=SAMPLE_PROPERTY,
            defects=SAMPLE_DEFECTS,
            memory=MockMemory(),
            tools=registry,
        )
        assert "30 days" in seller.system_prompt

    def test_custom_days_on_market(self, registry):
        seller = SellerAgent(
            model=MockModel(),
            property_data=SAMPLE_PROPERTY,
            defects=SAMPLE_DEFECTS,
            memory=MockMemory(),
            tools=registry,
            days_on_market=60,
        )
        assert "60 days" in seller.system_prompt

    def test_default_carrying_cost(self, registry):
        seller = SellerAgent(
            model=MockModel(),
            property_data=SAMPLE_PROPERTY,
            defects=SAMPLE_DEFECTS,
            memory=MockMemory(),
            tools=registry,
        )
        assert "1,500" in seller.system_prompt

    def test_custom_carrying_cost(self, registry):
        seller = SellerAgent(
            model=MockModel(),
            property_data=SAMPLE_PROPERTY,
            defects=SAMPLE_DEFECTS,
            memory=MockMemory(),
            tools=registry,
            carrying_cost=2000,
        )
        assert "2,000" in seller.system_prompt


# --- Buyer constructor tests ---


class TestBuyerConstructor:
    def test_default_alternative_price(self, registry):
        buyer = BuyerAgent(
            model=MockModel(),
            budget=250000,
            memory=MockMemory(),
            tools=registry,
        )
        assert buyer.alternative_price == 0

    def test_custom_alternative_price(self, buyer_naive):
        assert buyer_naive.alternative_price == 180000

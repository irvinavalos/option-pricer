import pytest

from tests.cases import (
    DeltaTestCase,
    GammaTestCase,
    MarketState,
    OptionTestCase,
    RhoTestCase,
    TestMetadata,
    ThetaTestCase,
    VegaTestCase,
)

_HULL_15_MARKET = MarketState(S=42, K=40, T=0.5, r=0.1, vol=0.2)
_HULL_15_METADATA = TestMetadata(name="Hull_15", source="Hull Chapter 15, Example 15.6")


@pytest.fixture
def hull_15() -> OptionTestCase:
    """Return Market state found in Hull Chapter 15 Example 15.6"""
    return OptionTestCase(
        market=_HULL_15_MARKET,
        metadata=_HULL_15_METADATA,
        expected_call=4.76,
        expected_put=0.81,
    )


_HULL_19_MARKET = MarketState(S=49, K=50, T=0.3846, r=0.05, vol=0.2)
_HULL_19_METADATA = TestMetadata(
    name="Hull_19",
    source="Hull Chapter 19",
    notes="Examples of Greeks in Ch. 19 utilize the same Market state",
)


@pytest.fixture
def hull_19_delta() -> DeltaTestCase:
    return DeltaTestCase(
        market=_HULL_19_MARKET,
        metadata=_HULL_19_METADATA,
        expected_call=0.522,
        expected_put=-0.478,
    )


@pytest.fixture
def hull_19_theta() -> ThetaTestCase:
    return ThetaTestCase(
        market=_HULL_19_MARKET,
        metadata=_HULL_19_METADATA,
        expected_call_calendar=-0.0118,
        expected_put_calendar=0,
        expected_call_trading=-0.0171,
        expected_put_trading=0,
    )


@pytest.fixture
def hull_19_gamma() -> GammaTestCase:
    return GammaTestCase(
        market=_HULL_19_MARKET, metadata=_HULL_19_METADATA, expected_gamma=0.066
    )


@pytest.fixture
def hull_19_vega() -> VegaTestCase:
    return VegaTestCase(
        market=_HULL_19_MARKET, metadata=_HULL_19_METADATA, expected_vega=12.1
    )


@pytest.fixture
def hull_19_rho() -> RhoTestCase:
    return RhoTestCase(
        market=_HULL_19_MARKET,
        metadata=_HULL_19_METADATA,
        expected_call=8.91,
        expected_put=0,
    )

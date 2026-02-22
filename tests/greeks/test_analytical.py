import pytest

from bspx.greeks.analytical import delta, gamma, rho, theta, vega
from bspx.pricing.black_scholes_model import (
    BlackScholesState,
    build_black_scholes_state,
)
from bspx.types import DayCount
from tests.cases import (
    DeltaTestCase,
    GammaTestCase,
    MarketState,
    RhoTestCase,
    ThetaTestCase,
    VegaTestCase,
)


def _build_state_from_test_case(
    market: MarketState,
) -> BlackScholesState:
    return build_black_scholes_state(
        S=market.S,
        K=market.K,
        T=market.T,
        r=market.r,
        vol=market.vol,
    )


def test_delta_call(hull_19_delta: DeltaTestCase):
    """Test delta on call option matches Hull example 19"""
    state = _build_state_from_test_case(hull_19_delta.market)

    delta_ = delta(state, "call")

    assert delta_ == pytest.approx(hull_19_delta.expected_call, abs=0.01)


def test_theta_call_trading(hull_19_theta: ThetaTestCase):
    """Test theta call on call option  matches Hull example 19 using number of trading days"""
    state = _build_state_from_test_case(hull_19_theta.market)

    theta_ = theta(state, "call", DayCount.TRADING)

    assert theta_ == pytest.approx(hull_19_theta.expected_call_trading, abs=0.01)


def test_theta_call_calendar(hull_19_theta: ThetaTestCase):
    """Test theta call matches Hull example 19 using number of calendar days"""
    state = _build_state_from_test_case(hull_19_theta.market)

    theta_ = theta(state, "call", DayCount.CALENDAR)

    assert theta_ == pytest.approx(hull_19_theta.expected_call_calendar, abs=0.01)


def test_gamma(hull_19_gamma: GammaTestCase):
    """Test gamma matches Hull example 19"""
    state = _build_state_from_test_case(hull_19_gamma.market)

    gamma_ = gamma(state)

    assert gamma_ == pytest.approx(hull_19_gamma.expected_gamma, abs=0.01)


def test_vega(hull_19_vega: VegaTestCase):
    """Test vega matches Hull example 19"""
    state = _build_state_from_test_case(hull_19_vega.market)

    vega_ = vega(state)

    assert vega_ == pytest.approx(hull_19_vega.expected_vega, abs=0.01)


def test_rho_call(hull_19_rho: RhoTestCase):
    """Test rho call option matches Hull example 19"""
    state = _build_state_from_test_case(hull_19_rho.market)

    rho_ = rho(state, "call")

    assert rho_ == pytest.approx(hull_19_rho.expected_call, abs=0.01)

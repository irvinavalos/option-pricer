import numpy as np
import pytest
from hypothesis import given
from tests.cases import (
    DeltaTestCase,
    GammaTestCase,
    RhoTestCase,
    ThetaTestCase,
    VegaTestCase,
)
from tests.constants import (
    GREEK_IDENT_ATOL,
    HULL_ABS,
    PUT_CALL_PARITY_REL,
    THETA_IDENT_ATOL,
)
from tests.hypothesis_strategies import gen_black_scholes_parameters

from bspx.greeks import (
    AnalyticalBackend,
    calculate_greeks,
    delta,
    gamma,
    rho,
    theta,
    vega,
)
from bspx.instruments import Greeks
from bspx.pricing import build_black_scholes_state
from bspx.types import DayCount


def test_delta_call(hull_19_delta: DeltaTestCase):
    """Test delta on call option matches Hull example 19"""
    backend = hull_19_delta.market.analytical_backend()
    assert delta(backend, "call") == pytest.approx(
        hull_19_delta.expected_call, abs=HULL_ABS
    )


def test_delta_put(hull_19_delta: DeltaTestCase):
    """Test delta on put option matches Hull example 19"""
    backend = hull_19_delta.market.analytical_backend()
    assert delta(backend, "put") == pytest.approx(
        hull_19_delta.expected_put, abs=HULL_ABS
    )


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_delta_call_put_symmetry(bs_params):
    S, K, T, r, vol = bs_params
    state = build_black_scholes_state(S, K, T, r, vol)
    backend = AnalyticalBackend(state)
    assert delta(backend, "call") + np.abs(delta(backend, "put")) == pytest.approx(
        1.0, abs=GREEK_IDENT_ATOL
    )


def test_theta_call_trading(hull_19_theta: ThetaTestCase):
    """Test theta call on call option  matches Hull example 19 using number of trading days"""
    backend = hull_19_theta.market.analytical_backend()
    assert theta(backend, "call", DayCount.TRADING) == pytest.approx(
        hull_19_theta.expected_call_trading, abs=HULL_ABS
    )


def test_theta_put_trading(hull_19_theta: ThetaTestCase):
    """Test theta put on put option  matches Hull example 19 using number of trading days"""
    backend = hull_19_theta.market.analytical_backend()
    assert theta(backend, "put", DayCount.TRADING) == pytest.approx(
        hull_19_theta.expected_put_trading, abs=HULL_ABS
    )


def test_theta_call_calendar(hull_19_theta: ThetaTestCase):
    """Test theta call matches Hull example 19 using number of calendar days"""
    backend = hull_19_theta.market.analytical_backend()
    assert theta(backend, "call", DayCount.CALENDAR) == pytest.approx(
        hull_19_theta.expected_call_calendar, abs=HULL_ABS
    )


def test_theta_put_calendar(hull_19_theta: ThetaTestCase):
    """Test theta put matches Hull example 19 using number of calendar days"""
    backend = hull_19_theta.market.analytical_backend()
    assert theta(backend, "put", DayCount.CALENDAR) == pytest.approx(
        hull_19_theta.expected_put_calendar, abs=HULL_ABS
    )


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_theta_call_put_relationship(bs_params):
    S, K, T, r, vol = bs_params
    state = build_black_scholes_state(S, K, T, r, vol)
    backend = AnalyticalBackend(state)
    diff = theta(backend, "put", DayCount.CALENDAR) - theta(
        backend, "call", DayCount.CALENDAR
    )
    expected = r * K * np.exp(-r * T) / DayCount.CALENDAR
    # For threshold use abs instead of rel since expected value could be zero (or close to it)
    assert diff == pytest.approx(expected, abs=THETA_IDENT_ATOL)


def test_gamma(hull_19_gamma: GammaTestCase):
    """Test gamma matches Hull example 19"""
    backend = hull_19_gamma.market.analytical_backend()
    assert gamma(backend) == pytest.approx(hull_19_gamma.expected_gamma, abs=HULL_ABS)


def test_vega(hull_19_vega: VegaTestCase):
    """Test vega matches Hull example 19"""
    backend = hull_19_vega.market.analytical_backend()
    assert vega(backend) == pytest.approx(hull_19_vega.expected_vega, abs=HULL_ABS)


def test_rho_call(hull_19_rho: RhoTestCase):
    """Test rho call option matches Hull example 19"""
    backend = hull_19_rho.market.analytical_backend()
    assert rho(backend, "call") == pytest.approx(
        hull_19_rho.expected_call, abs=HULL_ABS
    )


def test_rho_put(hull_19_rho: RhoTestCase):
    """Test rho put option matches Hull example 19"""
    backend = hull_19_rho.market.analytical_backend()
    assert rho(backend, "put") == pytest.approx(hull_19_rho.expected_put, abs=HULL_ABS)


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_rho_call_put_relationship(bs_params):
    S, K, T, r, vol = bs_params
    state = build_black_scholes_state(S, K, T, r, vol)
    backend = AnalyticalBackend(state)
    scale = K * T * np.exp(-r * T)
    assert rho(backend, "call") + np.abs(rho(backend, "put")) == pytest.approx(
        scale, abs=PUT_CALL_PARITY_REL
    )


def test_calculate_greeks_returns_correct_type(hull_19_delta: DeltaTestCase):
    backend = hull_19_delta.market.analytical_backend()
    result = calculate_greeks(backend)
    assert isinstance(result, Greeks)

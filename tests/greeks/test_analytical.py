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
    state = hull_19_delta.market.to_bs_state()
    backend = AnalyticalBackend(state)
    assert delta(backend, "call") == pytest.approx(
        hull_19_delta.expected_call, abs=0.01
    )


def test_delta_put(hull_19_delta: DeltaTestCase):
    """Test delta on put option matches Hull example 19"""
    state = hull_19_delta.market.to_bs_state()
    backend = AnalyticalBackend(state)
    assert delta(backend, "put") == pytest.approx(hull_19_delta.expected_put, abs=0.01)


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_delta_call_put_symmetry(bs_params):
    S, K, T, r, vol = bs_params
    state = build_black_scholes_state(S, K, T, r, vol)
    backend = AnalyticalBackend(state)
    assert delta(backend, "call") + np.abs(delta(backend, "put")) == pytest.approx(
        1.0, abs=1e-10
    )


def test_theta_call_trading(hull_19_theta: ThetaTestCase):
    """Test theta call on call option  matches Hull example 19 using number of trading days"""
    state = hull_19_theta.market.to_bs_state()
    backend = AnalyticalBackend(state)
    assert theta(backend, "call", DayCount.TRADING) == pytest.approx(
        hull_19_theta.expected_call_trading, abs=0.01
    )


def test_theta_put_trading(hull_19_theta: ThetaTestCase):
    """Test theta put on put option  matches Hull example 19 using number of trading days"""
    state = hull_19_theta.market.to_bs_state()
    backend = AnalyticalBackend(state)
    assert theta(backend, "put", DayCount.TRADING) == pytest.approx(
        hull_19_theta.expected_put_trading, abs=0.01
    )


def test_theta_call_calendar(hull_19_theta: ThetaTestCase):
    """Test theta call matches Hull example 19 using number of calendar days"""
    state = hull_19_theta.market.to_bs_state()
    backend = AnalyticalBackend(state)
    assert theta(backend, "call", DayCount.CALENDAR) == pytest.approx(
        hull_19_theta.expected_call_calendar, abs=0.01
    )


def test_theta_put_calendar(hull_19_theta: ThetaTestCase):
    """Test theta put matches Hull example 19 using number of calendar days"""
    state = hull_19_theta.market.to_bs_state()
    backend = AnalyticalBackend(state)
    assert theta(backend, "put", DayCount.CALENDAR) == pytest.approx(
        hull_19_theta.expected_put_calendar, abs=0.01
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
    assert diff == pytest.approx(expected, abs=1e-6)


def test_gamma(hull_19_gamma: GammaTestCase):
    """Test gamma matches Hull example 19"""
    state = hull_19_gamma.market.to_bs_state()
    backend = AnalyticalBackend(state)
    assert gamma(backend) == pytest.approx(hull_19_gamma.expected_gamma, abs=0.01)


def test_vega(hull_19_vega: VegaTestCase):
    """Test vega matches Hull example 19"""
    state = hull_19_vega.market.to_bs_state()
    backend = AnalyticalBackend(state)
    assert vega(backend) == pytest.approx(hull_19_vega.expected_vega, abs=0.01)


def test_rho_call(hull_19_rho: RhoTestCase):
    """Test rho call option matches Hull example 19"""
    state = hull_19_rho.market.to_bs_state()
    backend = AnalyticalBackend(state)
    assert rho(backend, "call") == pytest.approx(hull_19_rho.expected_call, abs=0.01)


def test_rho_put(hull_19_rho: RhoTestCase):
    """Test rho put option matches Hull example 19"""
    state = hull_19_rho.market.to_bs_state()
    backend = AnalyticalBackend(state)
    assert rho(backend, "put") == pytest.approx(hull_19_rho.expected_put, abs=0.01)


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_rho_call_put_relationship(bs_params):
    S, K, T, r, vol = bs_params
    state = build_black_scholes_state(S, K, T, r, vol)
    backend = AnalyticalBackend(state)
    scale = K * T * np.exp(-r * T)
    assert rho(backend, "call") + np.abs(rho(backend, "put")) == pytest.approx(
        scale, abs=1e-6
    )


def test_calculate_greeks_returns_correct_type(hull_19_delta: DeltaTestCase):
    state = hull_19_delta.market.to_bs_state()
    backend = AnalyticalBackend(state)
    result = calculate_greeks(backend)
    assert isinstance(result, Greeks)

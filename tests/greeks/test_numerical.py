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

from bspx.greeks import AnalyticalBackend
from bspx.greeks.numerical import NumericalBackend
from bspx.pricing import (
    black_scholes_price,
    build_black_scholes_state,
)
from bspx.types import DayCount, OptionType


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_fd_matches_analytical(
    hull_19_delta: DeltaTestCase, option_type: OptionType
):
    state = hull_19_delta.market.to_bs_state()
    ana_backend = AnalyticalBackend(state)
    num_backend = NumericalBackend(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol
    )
    ana_delta = ana_backend.delta(option_type)
    num_delta = num_backend.delta(option_type)
    assert num_delta == pytest.approx(ana_delta, rel=1e-3)


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_delta_fd_call_put_symmetry(bs_params):
    S, K, T, r, vol = bs_params
    state = build_black_scholes_state(S, K, T, r, vol)
    num_backend = NumericalBackend(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol
    )
    delta_call = num_backend.delta(option_type="call")
    delta_put = num_backend.delta(option_type="put")
    assert delta_call + np.abs(delta_put) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_fd_matches_analytical_calendar(
    hull_19_theta: ThetaTestCase, option_type: OptionType
):
    state = hull_19_theta.market.to_bs_state()
    ana_backend = AnalyticalBackend(state)
    num_backend = NumericalBackend(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol
    )
    ana_theta = ana_backend.theta(option_type)
    num_theta = num_backend.theta(option_type)
    assert num_theta == pytest.approx(ana_theta, rel=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_fd_matches_analytical_trading(
    hull_19_theta: ThetaTestCase, option_type: OptionType
):
    state = hull_19_theta.market.to_bs_state()
    ana_backend = AnalyticalBackend(state)
    num_backend = NumericalBackend(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol
    )
    ana_theta = ana_backend.theta(option_type, day_count=DayCount.TRADING)
    num_theta = num_backend.theta(option_type, day_count=DayCount.TRADING)
    assert num_theta == pytest.approx(ana_theta, rel=1e-3)


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_theta_fd_call_put_relationship(bs_params):
    S, K, T, r, vol = bs_params
    state = build_black_scholes_state(S, K, T, r, vol)
    num_backend = NumericalBackend(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol
    )
    theta_call = num_backend.theta(option_type="call")
    theta_put = num_backend.theta(option_type="put")
    diff = theta_put - theta_call
    expected = r * K * np.exp(-r * T) / DayCount.CALENDAR
    # For threshold use abs instead of rel since expected value could be zero (or close to it)
    assert diff == pytest.approx(expected, abs=1e-6)


def test_gamma_fd_matches_analytical(hull_19_gamma: GammaTestCase):
    state = hull_19_gamma.market.to_bs_state()
    ana_backend = AnalyticalBackend(state)
    num_backend = NumericalBackend(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol
    )
    ana_gamma = ana_backend.gamma()
    num_gamma = num_backend.gamma()
    assert num_gamma == pytest.approx(ana_gamma, rel=1e-3)


def test_vega_fd_matches_analytical(hull_19_vega: VegaTestCase):
    state = hull_19_vega.market.to_bs_state()
    ana_backend = AnalyticalBackend(state)
    num_backend = NumericalBackend(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol
    )
    ana_vega = ana_backend.vega()
    num_vega = num_backend.vega()
    assert num_vega == pytest.approx(ana_vega, rel=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_fd_matches_analytical(hull_19_rho: RhoTestCase, option_type: OptionType):
    state = hull_19_rho.market.to_bs_state()
    ana_backend = AnalyticalBackend(state)
    num_backend = NumericalBackend(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol
    )
    ana_rho = ana_backend.rho(option_type)
    num_rho = num_backend.rho(option_type)
    assert num_rho == pytest.approx(ana_rho, rel=1e-3)


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_rho_fd_call_put_relationship(bs_params):
    S, K, T, r, vol = bs_params
    state = build_black_scholes_state(S, K, T, r, vol)
    num_backend = NumericalBackend(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol
    )
    rho_call = num_backend.rho(option_type="call")
    rho_put = num_backend.rho(option_type="put")
    scale = K * T * np.exp(-r * T)
    assert rho_call + np.abs(rho_put) == pytest.approx(scale, abs=1e-6)

import numpy as np
import pytest
from hypothesis import given

from bspx.greeks import delta
from bspx.greeks.analytical import gamma, rho, theta, vega
from bspx.greeks.numerical import delta_fd, gamma_fd, rho_fd, theta_fd, vega_fd
from bspx.pricing.black_scholes_model import (
    black_scholes_price,
    build_black_scholes_state,
)
from bspx.types import DayCount, OptionType
from tests.cases import (
    DeltaTestCase,
    GammaTestCase,
    RhoTestCase,
    ThetaTestCase,
    VegaTestCase,
)
from tests.hypothesis_strategies import gen_black_scholes_parameters


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_delta_fd_matches_analytical(
    hull_19_delta: DeltaTestCase, option_type: OptionType
):
    state = hull_19_delta.market.to_bs_state()
    num_delta = delta_fd(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol, option_type
    )
    ana_delta = delta(state, option_type)
    assert num_delta == pytest.approx(ana_delta, rel=1e-3)


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_delta_fd_call_put_symmetry(bs_params):
    S, K, T, r, vol = bs_params
    state = build_black_scholes_state(S, K, T, r, vol)
    delta_call = delta_fd(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol, "call"
    )
    delta_put = delta_fd(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol, "put"
    )
    assert delta_call + np.abs(delta_put) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_fd_matches_analytical_calendar(
    hull_19_theta: ThetaTestCase, option_type: OptionType
):
    state = hull_19_theta.market.to_bs_state()
    num_theta = theta_fd(
        black_scholes_price,
        state.S,
        state.K,
        state.T,
        state.r,
        state.vol,
        option_type,
        DayCount.CALENDAR,
    )
    ana_theta = theta(state, option_type, DayCount.CALENDAR)
    assert num_theta == pytest.approx(ana_theta, rel=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_theta_fd_matches_analytical_trading(
    hull_19_theta: ThetaTestCase, option_type: OptionType
):
    state = hull_19_theta.market.to_bs_state()
    num_theta = theta_fd(
        black_scholes_price,
        state.S,
        state.K,
        state.T,
        state.r,
        state.vol,
        option_type,
        DayCount.TRADING,
    )
    ana_theta = theta(state, option_type, DayCount.TRADING)
    assert num_theta == pytest.approx(ana_theta, rel=1e-3)


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_theta_fd_call_put_relationship(bs_params):
    S, K, T, r, vol = bs_params
    state = build_black_scholes_state(S, K, T, r, vol)
    theta_put = theta_fd(
        black_scholes_price,
        state.S,
        state.K,
        state.T,
        state.r,
        state.vol,
        "put",
        DayCount.CALENDAR,
    )
    theta_call = theta_fd(
        black_scholes_price,
        state.S,
        state.K,
        state.T,
        state.r,
        state.vol,
        "call",
        DayCount.CALENDAR,
    )
    diff = theta_put - theta_call
    expected = r * K * np.exp(-r * T) / DayCount.CALENDAR
    # For threshold use abs instead of rel since expected value could be zero (or close to it)
    assert diff == pytest.approx(expected, abs=1e-6)


def test_gamma_fd_matches_analytical(hull_19_gamma: GammaTestCase):
    state = hull_19_gamma.market.to_bs_state()
    num_gamma = gamma_fd(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol
    )
    ana_gamma = gamma(state)
    assert num_gamma == pytest.approx(ana_gamma, rel=1e-3)


def test_vega_fd_matches_analytical(hull_19_vega: VegaTestCase):
    state = hull_19_vega.market.to_bs_state()
    num_vega = vega_fd(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol
    )
    ana_vega = vega(state)
    assert num_vega == pytest.approx(ana_vega, rel=1e-3)


@pytest.mark.parametrize("option_type", ["call", "put"])
def test_rho_fd_matches_analytical(hull_19_rho: RhoTestCase, option_type: OptionType):
    state = hull_19_rho.market.to_bs_state()
    num_rho = rho_fd(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol, option_type
    )
    ana_rho = rho(state, option_type)
    assert num_rho == pytest.approx(ana_rho, rel=1e-3)


@pytest.mark.slow
@given(bs_params=gen_black_scholes_parameters())
def test_rho_call_put_relationship(bs_params):
    S, K, T, r, vol = bs_params
    state = build_black_scholes_state(S, K, T, r, vol)
    rho_call = rho_fd(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol, "call"
    )
    rho_put = rho_fd(
        black_scholes_price, state.S, state.K, state.T, state.r, state.vol, "put"
    )
    scale = K * T * np.exp(-r * T)
    assert rho_call + np.abs(rho_put) == pytest.approx(scale, abs=1e-6)

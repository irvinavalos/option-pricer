import numpy as np
import pytest
from hypothesis import given, settings

from bspx.pricing.black_scholes_model import (
    build_black_scholes_state,
)
from tests.cases import OptionTestCase
from tests.hypothesis_strategies import gen_black_scholes_parameters


def test_black_scholes_call_hull(hull_15: OptionTestCase):
    """Test call price method matches Hull example 15.6"""
    state = hull_15.market.to_bs_state()
    assert state.call_price() == pytest.approx(hull_15.expected_call, abs=0.01)


def test_black_scholes_put_hull(hull_15: OptionTestCase):
    """Test put price method matches Hull example 15.6"""
    state = hull_15.market.to_bs_state()
    assert state.put_price() == pytest.approx(hull_15.expected_put, abs=0.01)


@pytest.mark.slow
@settings(max_examples=500)
@given(bs_params=gen_black_scholes_parameters())
def test_black_scholes_put_call_parity(bs_params):
    """Test Black-Scholes put-call parity"""
    S, K, T, r, vol = bs_params
    state = build_black_scholes_state(S, K, T, r, vol)
    parity = state.call_price() - state.put_price()
    expected = S - K * np.exp(-r * T)
    assert parity == pytest.approx(expected, rel=1e-6)

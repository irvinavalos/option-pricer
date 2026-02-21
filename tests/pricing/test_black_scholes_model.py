import numpy as np
import pytest
from hypothesis import given, settings

from bspx.pricing.black_scholes_model import (
    BlackScholesState,
    build_black_scholes_state,
    call_price,
    put_price,
)
from bspx.types import OptionTestCase
from tests.hypothesis_strategies import gen_black_scholes_parameters


def _build_state_from_test_case(
    test_case_hull_15_6: OptionTestCase,
) -> BlackScholesState:
    return build_black_scholes_state(
        S=test_case_hull_15_6.S,
        K=test_case_hull_15_6.K,
        T=test_case_hull_15_6.T,
        r=test_case_hull_15_6.r,
        vol=test_case_hull_15_6.vol,
    )


def test_black_scholes_call_hull(test_case_hull_15_6: OptionTestCase):
    """Test call price method matches Hull example 15.6"""
    state = _build_state_from_test_case(test_case_hull_15_6)

    call = call_price(state)

    assert call == pytest.approx(test_case_hull_15_6.expected_call, abs=0.01)


def test_black_scholes_put_hull(test_case_hull_15_6: OptionTestCase):
    """Test put price method matches Hull example 15.6"""
    state = _build_state_from_test_case(test_case_hull_15_6)

    put = put_price(state)

    assert put == pytest.approx(test_case_hull_15_6.expected_put, abs=0.01)


@pytest.mark.slow
@settings(max_examples=500)
@given(bs_params=gen_black_scholes_parameters())
def test_black_scholes_put_call_parity(bs_params):
    """Test Black-Scholes put-call parity"""
    S, K, T, r, vol = bs_params

    state = build_black_scholes_state(S, K, T, r, vol)

    parity = call_price(state) - put_price(state)
    expected = S - K * np.exp(-r * T)

    assert parity == pytest.approx(expected=expected, rel=1e-6)

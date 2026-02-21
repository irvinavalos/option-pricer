import pytest

from bspx.types import OptionTestCase

TEST_CASE_HULL_15_6 = OptionTestCase(
    name="Hull_15_6",
    S=42,
    K=40,
    T=0.5,
    r=0.1,
    vol=0.2,
    expected_call=4.76,
    expected_put=0.81,
    source="Hull Chapter 15, Example 15.6",
)


@pytest.fixture
def test_case_hull_15_6() -> OptionTestCase:
    """Return Market state found in Hull Chapter 15 Example 15.6"""
    return TEST_CASE_HULL_15_6

"""Regression tests for zero-argument human/demo problem presets."""

from finite_element_options.problems import CreditRiskProblem, OptionPricingProblem


def test_problem_presets_construct_with_documented_defaults() -> None:
    option = OptionPricingProblem()
    credit = CreditRiskProblem()

    assert option.dynamics is not None
    assert option.payoff is not None
    assert option.boundary_condition is not None
    assert credit.dynamics is not None
    assert credit.payoff is not None
    assert credit.boundary_condition is not None

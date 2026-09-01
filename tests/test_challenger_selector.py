import numpy as np

from gap_imputation_benchmark.selection.challenger_selector import select_challenger


def test_selector_keeps_pchip_when_no_lower_bound_is_positive():
    result = select_challenger({"linear": True}, {"linear": .02}, {"linear": .03}, minimum_gain=.01, z_value=1.0)
    assert result.selected_method == "pchip"


def test_selector_ignores_non_applicable_method_and_selects_best_lcb():
    result = select_challenger({"linear": False, "template": True}, {"linear": 1., "template": .08}, {"linear": 0., "template": .01}, minimum_gain=.05, z_value=1.0)
    assert result.selected_method == "template"
    assert result.lower_confidence_bounds["template"] == .07

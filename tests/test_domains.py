"""Tests for the domain boundary and the stable Eye-Tracking adapter."""

from gap_imputation_benchmark.benchmark.evaluation import IMPUTER_METHODS
from gap_imputation_benchmark.benchmark.records import FEATURES_BASIC_NORMALIZED
from gap_imputation_benchmark.domains.eyetracking import EYE_TRACKING_DOMAIN
from gap_imputation_benchmark.imputers.registry import EYE_TRACKING_METHODS


def test_eye_tracking_domain_is_the_single_method_and_feature_contract():
    assert EYE_TRACKING_DOMAIN.feature_columns == FEATURES_BASIC_NORMALIZED
    assert tuple(EYE_TRACKING_DOMAIN.methods) == EYE_TRACKING_METHODS
    assert IMPUTER_METHODS is EYE_TRACKING_DOMAIN.methods

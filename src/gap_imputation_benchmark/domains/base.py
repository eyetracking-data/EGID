"""Stable contracts between the benchmark core and individual domains."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from gap_imputation_benchmark.benchmark.gaps import ArtificialGap
from gap_imputation_benchmark.benchmark.imputers import ImputationResult


Imputer = Callable[[ArtificialGap], ImputationResult]


@dataclass(frozen=True)
class DomainSpec:
    """The explicit interface a domain provides to generic workflows.

    Domain implementations own their loaders, sampling protocol, experiment
    configuration, and optional domain-specific features.  The shared core
    only needs this small, serialisable description for training and inference.
    """

    name: str
    feature_columns: tuple[str, ...]
    methods: Mapping[str, Imputer]
    scale_dependent_feature_numerators: Mapping[str, str] = field(default_factory=dict)
    method_labels: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("A domain name is required.")
        if not self.feature_columns:
            raise ValueError("A domain must define at least one feature column.")
        if not self.methods:
            raise ValueError("A domain must define at least one imputation method.")
        unknown_features = set(self.scale_dependent_feature_numerators) - set(self.feature_columns)
        if unknown_features:
            raise ValueError(f"Scale-dependent features are not in the feature contract: {sorted(unknown_features)}")
        unknown_labels = set(self.method_labels) - set(self.methods)
        if unknown_labels:
            raise ValueError(f"Method labels are not registered methods: {sorted(unknown_labels)}")

# General preprocessing pipeline

This module preserves the three-stage, provenance-aware preprocessing workflow:

1. outlier detection;
2. missing-value imputation with a domain selector;
3. optional standardization.

Selector artifacts are resolved only from the repository-level `artifacts/` directory:
`artifacts/eyetracking/selector.joblib`, `artifacts/weather/selector.joblib`, and
`artifacts/traffic/selector.joblib`. No absolute workstation paths are embedded.

See `notebooks/algorithm/synthetic_example/01_example_pipeline_call.ipynb` for a portable example. Domain-specific examples are in the neighbouring `eyetracking/`, `weather/`, and `traffic/` directories.

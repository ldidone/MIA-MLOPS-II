"""Shared pytest fixtures.

We build a synthetic dataframe with a mix of numeric / categorical / binary
columns to exercise the schema-agnostic preprocessing logic. The concrete
column names are intentionally arbitrary: the whole point of the refactor is
that the pipeline does not depend on them.

The target is continuous (regression), mirroring California Housing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope="session")
def sample_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 400

    age = rng.integers(13, 25, size=n)
    sleep = rng.uniform(3, 10, size=n).round(1)
    screen = rng.uniform(0, 12, size=n).round(1)
    anxiety = rng.integers(0, 11, size=n)
    therapy = rng.integers(0, 2, size=n)
    substance = rng.integers(0, 2, size=n)

    # Continuous regression target with real signal so CV doesn't collapse.
    target = (
        2.5
        + 0.05 * age
        + 0.30 * screen
        - 0.20 * sleep
        + 0.15 * anxiety
        + 0.40 * substance
        + rng.normal(0, 0.3, size=n)
    ).round(3)

    df = pd.DataFrame(
        {
            "age": age,
            "sleep_hours": sleep,
            "screen_time_hours_per_day": screen,
            "anxiety_score": anxiety,
            "gender": rng.choice(["Male", "Female", "Other"], size=n),
            "education_level": rng.choice(
                ["High School", "Bachelors", "Masters", "PhD"], size=n
            ),
            "therapy_history": therapy,
            "substance_use": substance,
            "target_value": target,
        }
    )
    return df

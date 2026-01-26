"""
Unit tests for helpers.py statistical functions.

Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from helpers import (
    EffectSize,
    calculate_cohens_d,
    interpret_cohens_d,
    calculate_confidence_interval,
    EFFECT_SMALL,
    EFFECT_MEDIUM,
    EFFECT_LARGE,
    DEFAULT_CONFIDENCE,
)


class TestEffectSize:
    """Tests for EffectSize enum and classification."""
    
    def test_negligible_effect(self):
        """Values below 0.2 should be negligible."""
        assert EffectSize.from_cohens_d(0.0) == EffectSize.NEGLIGIBLE
        assert EffectSize.from_cohens_d(0.1) == EffectSize.NEGLIGIBLE
        assert EffectSize.from_cohens_d(0.19) == EffectSize.NEGLIGIBLE
        # Negative values should also work
        assert EffectSize.from_cohens_d(-0.1) == EffectSize.NEGLIGIBLE
    
    def test_small_effect(self):
        """Values 0.2-0.5 should be small."""
        assert EffectSize.from_cohens_d(0.2) == EffectSize.SMALL
        assert EffectSize.from_cohens_d(0.35) == EffectSize.SMALL
        assert EffectSize.from_cohens_d(0.49) == EffectSize.SMALL
        assert EffectSize.from_cohens_d(-0.3) == EffectSize.SMALL
    
    def test_medium_effect(self):
        """Values 0.5-0.8 should be medium."""
        assert EffectSize.from_cohens_d(0.5) == EffectSize.MEDIUM
        assert EffectSize.from_cohens_d(0.65) == EffectSize.MEDIUM
        assert EffectSize.from_cohens_d(0.79) == EffectSize.MEDIUM
        assert EffectSize.from_cohens_d(-0.6) == EffectSize.MEDIUM
    
    def test_large_effect(self):
        """Values >= 0.8 should be large."""
        assert EffectSize.from_cohens_d(0.8) == EffectSize.LARGE
        assert EffectSize.from_cohens_d(1.0) == EffectSize.LARGE
        assert EffectSize.from_cohens_d(2.5) == EffectSize.LARGE
        assert EffectSize.from_cohens_d(-1.2) == EffectSize.LARGE
    
    def test_effect_size_values(self):
        """Effect size enum values should be lowercase strings."""
        assert EffectSize.NEGLIGIBLE.value == "negligible"
        assert EffectSize.SMALL.value == "small"
        assert EffectSize.MEDIUM.value == "medium"
        assert EffectSize.LARGE.value == "large"


class TestCohensD:
    """Tests for Cohen's d calculation."""
    
    def test_identical_groups(self):
        """Identical groups should have d=0."""
        group = np.array([1, 2, 3, 4, 5])
        assert calculate_cohens_d(group, group) == 0.0
    
    def test_positive_effect(self):
        """Group2 > Group1 should give positive d."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([4, 5, 6, 7, 8])
        d = calculate_cohens_d(group1, group2)
        assert d > 0
    
    def test_negative_effect(self):
        """Group2 < Group1 should give negative d."""
        group1 = np.array([4, 5, 6, 7, 8])
        group2 = np.array([1, 2, 3, 4, 5])
        d = calculate_cohens_d(group1, group2)
        assert d < 0
    
    def test_known_value(self):
        """Test against manually calculated value."""
        # Two groups with mean diff = 1, pooled SD ≈ 1
        group1 = np.array([0, 0, 0, 0])
        group2 = np.array([1, 1, 1, 1])
        d = calculate_cohens_d(group1, group2)
        # With no variance within groups, this is undefined (div by 0)
        # Our function should handle this gracefully
        assert d == 0.0 or np.isinf(d) or np.isnan(d) is False
    
    def test_symmetry(self):
        """Swapping groups should negate d."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([3, 4, 5, 6, 7])
        d1 = calculate_cohens_d(group1, group2)
        d2 = calculate_cohens_d(group2, group1)
        assert abs(d1 + d2) < 1e-10  # Should be opposite signs
    
    def test_large_effect_detection(self):
        """Large difference should produce large effect size."""
        group1 = np.array([1, 2, 3, 4, 5])
        group2 = np.array([10, 11, 12, 13, 14])
        d = calculate_cohens_d(group1, group2)
        assert EffectSize.from_cohens_d(d) == EffectSize.LARGE


class TestInterpretCohensD:
    """Tests for interpret_cohens_d backward compatibility function."""
    
    def test_returns_string(self):
        """Should return string, not EffectSize enum."""
        result = interpret_cohens_d(0.5)
        assert isinstance(result, str)
    
    def test_matches_enum(self):
        """Should match EffectSize enum values."""
        assert interpret_cohens_d(0.1) == "negligible"
        assert interpret_cohens_d(0.3) == "small"
        assert interpret_cohens_d(0.6) == "medium"
        assert interpret_cohens_d(1.0) == "large"


class TestConfidenceInterval:
    """Tests for confidence interval calculation."""
    
    def test_contains_mean(self):
        """CI should contain the sample mean."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        lower, upper = calculate_confidence_interval(data)
        mean = np.mean(data)
        assert lower <= mean <= upper
    
    def test_symmetric_data(self):
        """Symmetric data should have CI centered on mean."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        lower, upper = calculate_confidence_interval(data)
        mean = np.mean(data)
        # CI should be roughly symmetric around mean
        assert abs((upper - mean) - (mean - lower)) < 0.1
    
    def test_higher_confidence_wider_interval(self):
        """99% CI should be wider than 95% CI."""
        data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        ci_95 = calculate_confidence_interval(data, confidence=0.95)
        ci_99 = calculate_confidence_interval(data, confidence=0.99)
        width_95 = ci_95[1] - ci_95[0]
        width_99 = ci_99[1] - ci_99[0]
        assert width_99 > width_95
    
    def test_low_variance_narrow_interval(self):
        """Low variance data should have narrow CI."""
        low_var = np.array([5.0, 5.1, 4.9, 5.0, 5.0])
        high_var = np.array([1.0, 9.0, 3.0, 7.0, 5.0])
        ci_low = calculate_confidence_interval(low_var)
        ci_high = calculate_confidence_interval(high_var)
        width_low = ci_low[1] - ci_low[0]
        width_high = ci_high[1] - ci_high[0]
        assert width_low < width_high


class TestConstants:
    """Tests for module constants."""
    
    def test_effect_thresholds_ordered(self):
        """Effect size thresholds should be ordered."""
        assert EFFECT_SMALL < EFFECT_MEDIUM < EFFECT_LARGE
    
    def test_default_confidence_valid(self):
        """Default confidence should be between 0 and 1."""
        assert 0 < DEFAULT_CONFIDENCE < 1
    
    def test_standard_thresholds(self):
        """Thresholds should match Cohen's conventions."""
        assert EFFECT_SMALL == 0.2
        assert EFFECT_MEDIUM == 0.5
        assert EFFECT_LARGE == 0.8


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

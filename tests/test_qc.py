"""Tests for ymca_agent.qc module."""

import math

from ymca_agent.qc import (
    entropy,
    normalized_entropy,
    probability_margin,
    review_reasons,
    uncertainty_score,
)


class TestProbabilityMargin:
    def test_both_none(self):
        assert probability_margin(None, None) is None

    def test_one_none(self):
        assert probability_margin(0.9, None) is None

    def test_normal(self):
        assert probability_margin(0.9, 0.1) == 0.8

    def test_equal(self):
        assert probability_margin(0.5, 0.5) == 0.0

    def test_clamp_negative(self):
        # If top2 > top1 (shouldn't happen normally), margin clamped to 0
        assert probability_margin(0.3, 0.7) == 0.0


class TestEntropy:
    def test_empty(self):
        assert entropy({}) is None

    def test_none(self):
        assert entropy(None) is None

    def test_certain(self):
        result = entropy({"A": 1.0, "B": 0.0, "C": 0.0})
        assert result is not None
        assert result < 0.01

    def test_uniform(self):
        result = entropy({"A": 0.5, "B": 0.5})
        assert result is not None
        assert abs(result - math.log(2)) < 0.001

    def test_single_class(self):
        result = entropy({"A": 1.0})
        assert result == 0.0


class TestNormalizedEntropy:
    def test_empty(self):
        assert normalized_entropy({}) == 0.0

    def test_single_class(self):
        assert normalized_entropy({"A": 1.0}) == 0.0

    def test_uniform_two_classes(self):
        result = normalized_entropy({"A": 0.5, "B": 0.5})
        assert abs(result - 1.0) < 0.01

    def test_clamped_to_one(self):
        result = normalized_entropy({"A": 0.33, "B": 0.33, "C": 0.34})
        assert result <= 1.0


class TestReviewReasons:
    def test_all_good(self):
        reasons = review_reasons(
            yolo_confidence=0.95,
            segmentation_quality=0.95,
            segmentation_status="ok",
            top_probability=0.95,
            top2_probability=0.03,
            probabilities={"A": 0.95, "B": 0.03, "C": 0.02},
            overlap_score=0.0,
        )
        assert reasons == []

    def test_low_yolo_confidence(self):
        reasons = review_reasons(
            yolo_confidence=0.3,
            segmentation_quality=0.95,
            segmentation_status="ok",
            top_probability=0.95,
            top2_probability=0.03,
            probabilities={"A": 0.95},
            overlap_score=0.0,
        )
        assert "low_yolo_confidence" in reasons

    def test_rare_class(self):
        reasons = review_reasons(
            yolo_confidence=0.95,
            segmentation_quality=0.95,
            segmentation_status="ok",
            top_probability=0.95,
            top2_probability=0.03,
            probabilities={"PMB": 0.95},
            overlap_score=0.0,
            rare_class=True,
        )
        assert "rare_or_immature_class" in reasons

    def test_classifier_not_run(self):
        reasons = review_reasons(
            yolo_confidence=0.95,
            segmentation_quality=0.95,
            segmentation_status="ok",
            top_probability=None,
            top2_probability=None,
            probabilities=None,
            overlap_score=0.0,
            downstream_eligible=True,
        )
        assert reasons == ["classifier_not_run"]

    def test_small_margin(self):
        reasons = review_reasons(
            yolo_confidence=0.95,
            segmentation_quality=0.95,
            segmentation_status="ok",
            top_probability=0.50,
            top2_probability=0.45,
            probabilities={"A": 0.50, "B": 0.45},
            overlap_score=0.0,
        )
        assert "small_top1_top2_margin" in reasons


class TestUncertaintyScore:
    def test_perfect_cell_near_zero(self):
        score = uncertainty_score(
            yolo_confidence=0.99,
            segmentation_quality=0.99,
            top_probability=0.99,
            top2_probability=0.005,
            probabilities={"A": 0.99, "B": 0.005, "C": 0.005},
            overlap_score=0.0,
        )
        assert score < 0.15

    def test_bad_cell_near_one(self):
        score = uncertainty_score(
            yolo_confidence=0.2,
            segmentation_quality=0.3,
            top_probability=0.15,
            top2_probability=0.14,
            probabilities={"A": 0.15, "B": 0.14, "C": 0.13, "D": 0.12},
            overlap_score=0.8,
        )
        assert score > 0.6

    def test_range_zero_to_one(self):
        score = uncertainty_score(
            yolo_confidence=0.5,
            segmentation_quality=0.5,
            top_probability=0.5,
            top2_probability=0.3,
            probabilities={"A": 0.5, "B": 0.3},
            overlap_score=0.3,
        )
        assert 0.0 <= score <= 1.0

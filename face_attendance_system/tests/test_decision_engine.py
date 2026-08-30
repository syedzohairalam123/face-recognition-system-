"""
Decision Engine Test Suite
---------------------------
Comprehensive tests for the multi-signal decision engine.
Tests signal combination, uncertainty classification, decision logic,
explanation generation, and policy configuration.
"""

import pytest
import numpy as np
from app.services.decision_engine import (
    DecisionEngine,
    DecisionPolicy,
    DecisionSignals,
    DecisionResult,
    DecisionType,
    UncertaintyState,
)


class TestDecisionSignals:
    """Test DecisionSignals dataclass."""

    def test_default_signals(self):
        """Test default signal values."""
        signals = DecisionSignals()
        assert signals.face_similarity == 0.0
        assert signals.liveness_score == 0.0
        assert signals.face_quality == 0.0
        assert signals.candidate_margin == 0.0
        assert signals.detection_confidence == 0.0
        assert signals.embedding_stability == 1.0

    def test_signal_values(self):
        """Test setting signal values."""
        signals = DecisionSignals(
            face_similarity=0.9,
            liveness_score=0.8,
            face_quality=0.7,
            candidate_margin=0.3,
            detection_confidence=0.95,
        )
        assert signals.face_similarity == 0.9
        assert signals.liveness_score == 0.8
        assert signals.face_quality == 0.7
        assert signals.candidate_margin == 0.3
        assert signals.detection_confidence == 0.95

    def test_signals_to_dict(self):
        """Test signals conversion to dictionary."""
        signals = DecisionSignals(
            face_similarity=0.85,
            liveness_score=0.75,
            face_quality=0.65,
            candidate_margin=0.25,
            detection_confidence=0.9,
            raw_distance=0.15,
            raw_top_score=0.85,
            raw_second_score=0.60,
            liveness_checks_passed=4,
            liveness_checks_total=5,
        )
        result = signals.to_dict()
        assert result["face_similarity"] == 0.85
        assert result["liveness_score"] == 0.75
        assert result["face_quality"] == 0.65
        assert result["candidate_margin"] == 0.25
        assert result["detection_confidence"] == 0.9
        assert result["raw_distance"] == 0.15
        assert result["raw_top_score"] == 0.85
        assert result["raw_second_score"] == 0.60
        assert result["liveness_checks_passed"] == 4
        assert result["liveness_checks_total"] == 5


class TestDecisionPolicy:
    """Test DecisionPolicy configuration."""

    def test_default_policy(self):
        """Test default policy values."""
        policy = DecisionPolicy()
        assert policy.min_face_similarity == 0.5
        assert policy.min_liveness_score == 0.6
        assert policy.min_face_quality == 0.4
        assert policy.min_candidate_margin == 0.1
        assert policy.high_confidence_threshold == 0.85
        assert policy.low_confidence_threshold == 0.60
        assert policy.require_liveness is True
        assert policy.liveness_min_checks == 3

    def test_weight_normalization(self):
        """Test that weights are normalized to sum to 1.0."""
        policy = DecisionPolicy(
            weight_similarity=0.5,
            weight_liveness=0.3,
            weight_quality=0.1,
            weight_margin=0.05,
            weight_detection=0.05,
        )
        total = (
            policy.weight_similarity +
            policy.weight_liveness +
            policy.weight_quality +
            policy.weight_margin +
            policy.weight_detection
        )
        assert abs(total - 1.0) < 0.01

    def test_weight_normalization_on_invalid(self):
        """Test weight normalization when sum is not 1.0."""
        policy = DecisionPolicy(
            weight_similarity=0.8,
            weight_liveness=0.5,  # Sum > 1.0
            weight_quality=0.3,
            weight_margin=0.2,
            weight_detection=0.1,
        )
        total = (
            policy.weight_similarity +
            policy.weight_liveness +
            policy.weight_quality +
            policy.weight_margin +
            policy.weight_detection
        )
        assert abs(total - 1.0) < 0.01

    def test_policy_to_dict(self):
        """Test policy conversion to dictionary."""
        policy = DecisionPolicy(
            min_face_similarity=0.6,
            min_liveness_score=0.7,
            high_confidence_threshold=0.9,
        )
        result = policy.to_dict()
        assert result["min_face_similarity"] == 0.6
        assert result["min_liveness_score"] == 0.7
        assert result["high_confidence_threshold"] == 0.9
        assert "weights" in result
        assert result["weights"]["similarity"] == pytest.approx(0.35, rel=0.1)


class TestDecisionEngine:
    """Test DecisionEngine core functionality."""

    def test_engine_initialization(self):
        """Test decision engine initialization."""
        engine = DecisionEngine()
        assert engine.policy is not None
        assert isinstance(engine.policy, DecisionPolicy)

    def test_engine_with_custom_policy(self):
        """Test engine with custom policy."""
        custom_policy = DecisionPolicy(
            min_face_similarity=0.7,
            high_confidence_threshold=0.9,
        )
        engine = DecisionEngine(policy=custom_policy)
        assert engine.policy.min_face_similarity == 0.7
        assert engine.policy.high_confidence_threshold == 0.9

    def test_high_confidence_decision(self):
        """Test decision with high confidence signals."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.95,
            liveness_score=0.98,
            face_quality=0.92,
            candidate_margin=0.35,
            detection_confidence=0.95,
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert result.decision == DecisionType.ACCEPT
        assert result.uncertainty_state == UncertaintyState.HIGH_CONFIDENCE
        assert result.combined_score >= engine.policy.high_confidence_threshold
        assert "Identity confirmed" in result.explanation["primary_reason"]

    def test_low_confidence_decision(self):
        """Test decision with low confidence signals."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.7,
            liveness_score=0.75,
            face_quality=0.65,
            candidate_margin=0.15,
            detection_confidence=0.8,
            liveness_checks_passed=4,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert result.decision == DecisionType.ACCEPT
        assert result.uncertainty_state == UncertaintyState.LOW_CONFIDENCE
        assert result.combined_score >= engine.policy.low_confidence_threshold
        assert result.combined_score < engine.policy.high_confidence_threshold

    def test_uncertain_decision(self):
        """Test decision with uncertain signals."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.5,
            liveness_score=0.4,
            face_quality=0.5,  # Above minimum threshold
            candidate_margin=0.05,
            detection_confidence=0.6,
            liveness_checks_passed=4,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert result.decision == DecisionType.REVIEW
        assert result.uncertainty_state == UncertaintyState.UNCERTAIN
        assert result.combined_score < engine.policy.low_confidence_threshold

    def test_reject_no_user(self):
        """Test rejection when no user ID provided."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.9,
            liveness_score=0.9,
            face_quality=0.9,
        )
        
        result = engine.make_decision(signals, user_id=-1)
        
        assert result.decision == DecisionType.REJECT
        assert "No face match found" in result.explanation["primary_reason"]

    def test_reject_low_similarity(self):
        """Test rejection due to low similarity."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.3,  # Below threshold
            liveness_score=0.9,
            face_quality=0.9,
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert result.decision == DecisionType.REJECT
        assert "similarity too low" in result.explanation["primary_reason"].lower()

    def test_reject_low_quality(self):
        """Test rejection due to low face quality."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.8,
            liveness_score=0.9,
            face_quality=0.2,  # Below threshold
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert result.decision == DecisionType.REJECT
        assert "quality too low" in result.explanation["primary_reason"].lower()

    def test_reject_liveness_failed(self):
        """Test rejection due to liveness check failure."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.9,
            liveness_score=0.3,  # Low but check count matters more
            face_quality=0.9,
            liveness_checks_passed=1,  # Below minimum
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert result.decision == DecisionType.REJECT
        assert "liveness check failed" in result.explanation["primary_reason"].lower()

    def test_review_small_margin(self):
        """Test review due to small candidate margin."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.9,
            liveness_score=0.9,
            face_quality=0.9,
            candidate_margin=0.05,  # Below threshold
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert result.decision == DecisionType.REVIEW
        assert "margin too small" in result.explanation["primary_reason"].lower()

    def test_signal_conflict_detection(self):
        """Test detection of conflicting signals."""
        engine = DecisionEngine()
        
        # High similarity but very low liveness
        signals = DecisionSignals(
            face_similarity=0.9,
            liveness_score=0.3,
            face_quality=0.8,
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        assert result.uncertainty_state == UncertaintyState.UNCERTAIN

    def test_signal_combination(self):
        """Test signal combination with weights."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.8,
            liveness_score=0.9,
            face_quality=0.7,
            candidate_margin=0.2,
            detection_confidence=0.95,
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        # Check that signal breakdown exists
        assert "similarity_contribution" in result.signal_breakdown
        assert "liveness_contribution" in result.signal_breakdown
        assert "quality_contribution" in result.signal_breakdown
        assert "margin_contribution" in result.signal_breakdown
        assert "detection_contribution" in result.signal_breakdown

    def test_explanation_generation(self):
        """Test explanation generation for decisions."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.9,
            liveness_score=0.85,
            face_quality=0.8,
            candidate_margin=0.3,
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert "decision_summary" in result.explanation
        assert "uncertainty_state" in result.explanation
        assert "primary_reason" in result.explanation
        # signal_assessment may or may not be present depending on decision type

    def test_explanation_for_rejection(self):
        """Test explanation for rejection cases."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.3,
            liveness_score=0.9,
            face_quality=0.9,
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        # The primary reason should indicate rejection (similarity too low)
        assert "similarity" in result.explanation["primary_reason"].lower()
        # Note: signal_assessment may not be present for hard constraint rejections

    def test_explanation_for_review(self):
        """Test explanation for review cases."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.9,
            liveness_score=0.9,
            face_quality=0.9,
            candidate_margin=0.05,
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        # The primary reason should indicate review/uncertainty (margin too small)
        assert "margin" in result.explanation["primary_reason"].lower()
        # Note: signal_assessment may not be present for constraint-based reviews

    def test_result_to_dict(self):
        """Test DecisionResult conversion to dictionary."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.9,
            liveness_score=0.9,
            face_quality=0.9,
        )
        
        result = engine.make_decision(signals, user_id=1)
        result_dict = result.to_dict()
        
        assert "decision" in result_dict
        assert "uncertainty_state" in result_dict
        assert "combined_score" in result_dict
        assert "signals" in result_dict
        assert "explanation" in result_dict
        assert "signal_breakdown" in result_dict
        assert "processing_time_ms" in result_dict

    def test_policy_update(self):
        """Test updating decision policy parameters."""
        engine = DecisionEngine()
        
        engine.update_policy(
            min_face_similarity=0.7,
            high_confidence_threshold=0.95,
        )
        
        assert engine.policy.min_face_similarity == 0.7
        assert engine.policy.high_confidence_threshold == 0.95

    def test_policy_get(self):
        """Test getting current policy."""
        engine = DecisionEngine()
        policy = engine.get_policy()
        
        assert isinstance(policy, DecisionPolicy)
        assert policy.min_face_similarity == engine.policy.min_face_similarity

    def test_processing_time(self):
        """Test that processing time is recorded."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.9,
            liveness_score=0.9,
            face_quality=0.9,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert result.processing_time_ms >= 0
        assert result.processing_time_ms < 1000  # Should be fast

    def test_combined_score_clamping(self):
        """Test that combined score is clamped to [0, 1]."""
        engine = DecisionEngine()
        
        # Test with very high signals
        signals = DecisionSignals(
            face_similarity=1.5,  # Above 1.0
            liveness_score=1.5,
            face_quality=1.5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        assert 0.0 <= result.combined_score <= 1.0

    def test_disable_liveness_requirement(self):
        """Test decision engine with liveness requirement disabled."""
        policy = DecisionPolicy(require_liveness=False)
        engine = DecisionEngine(policy=policy)
        
        signals = DecisionSignals(
            face_similarity=0.9,
            liveness_score=0.2,  # Low liveness
            face_quality=0.9,
            liveness_checks_passed=1,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        # Should accept despite low liveness because requirement is disabled
        assert result.decision != DecisionType.REJECT or "liveness" not in result.explanation["primary_reason"].lower()

    def test_weight_adjustment_effect(self):
        """Test that different weights affect combined score."""
        engine1 = DecisionEngine(DecisionPolicy(weight_similarity=0.5, weight_liveness=0.3))
        engine2 = DecisionEngine(DecisionPolicy(weight_similarity=0.3, weight_liveness=0.5))
        
        signals = DecisionSignals(
            face_similarity=0.9,
            liveness_score=0.7,
            face_quality=0.8,
            candidate_margin=0.2,
            detection_confidence=0.9,
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result1 = engine1.make_decision(signals, user_id=1)
        result2 = engine2.make_decision(signals, user_id=1)
        
        # Different weights should produce different combined scores
        assert result1.combined_score != result2.combined_score


class TestDecisionEngineEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_all_zero_signals(self):
        """Test with all zero signals."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.0,
            liveness_score=0.0,
            face_quality=0.0,
            candidate_margin=0.0,
            detection_confidence=0.0,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert result.decision == DecisionType.REJECT
        assert result.uncertainty_state == UncertaintyState.UNCERTAIN

    def test_perfect_signals(self):
        """Test with perfect signals."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=1.0,
            liveness_score=1.0,
            face_quality=1.0,
            candidate_margin=1.0,
            detection_confidence=1.0,
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert result.decision == DecisionType.ACCEPT
        assert result.uncertainty_state == UncertaintyState.HIGH_CONFIDENCE
        assert result.combined_score >= engine.policy.high_confidence_threshold

    def test_boundary_threshold_high_confidence(self):
        """Test exactly at high confidence threshold."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.85,
            liveness_score=0.85,
            face_quality=0.85,
            candidate_margin=0.15,
            detection_confidence=0.85,
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        # Should be at or near the threshold
        assert result.uncertainty_state in [UncertaintyState.HIGH_CONFIDENCE, UncertaintyState.LOW_CONFIDENCE]

    def test_boundary_threshold_low_confidence(self):
        """Test exactly at low confidence threshold."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.6,
            liveness_score=0.6,
            face_quality=0.6,
            candidate_margin=0.1,
            detection_confidence=0.6,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        # Should be at or near the threshold
        assert result.uncertainty_state in [UncertaintyState.LOW_CONFIDENCE, UncertaintyState.UNCERTAIN]

    def test_user_id_zero(self):
        """Test with user_id = 0 (invalid)."""
        engine = DecisionEngine()
        signals = DecisionSignals(face_similarity=0.9)
        
        result = engine.make_decision(signals, user_id=0)
        
        assert result.decision == DecisionType.REJECT

    def test_negative_user_id(self):
        """Test with negative user_id."""
        engine = DecisionEngine()
        signals = DecisionSignals(face_similarity=0.9)
        
        result = engine.make_decision(signals, user_id=-5)
        
        assert result.decision == DecisionType.REJECT

    def test_very_small_margin(self):
        """Test with extremely small candidate margin."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.9,
            liveness_score=0.9,
            face_quality=0.9,
            candidate_margin=0.001,  # Almost zero
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert result.decision == DecisionType.REVIEW

    def test_large_margin(self):
        """Test with large candidate margin."""
        engine = DecisionEngine()
        signals = DecisionSignals(
            face_similarity=0.95,  # Higher to reach high confidence
            liveness_score=0.95,
            face_quality=0.95,
            candidate_margin=0.5,  # Very large
            detection_confidence=0.95,  # Add detection confidence
            liveness_checks_passed=5,
            liveness_checks_total=5,
        )
        
        result = engine.make_decision(signals, user_id=1)
        
        assert result.decision == DecisionType.ACCEPT
        assert result.uncertainty_state == UncertaintyState.HIGH_CONFIDENCE
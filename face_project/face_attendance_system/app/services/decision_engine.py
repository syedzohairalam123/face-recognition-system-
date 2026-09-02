"""
Multi-Signal Decision Engine
----------------------------
Advanced decision engine that combines multiple signals for face recognition.
Replaces simple threshold-based logic with intelligent multi-signal fusion.

Signals Combined:
    - Face Similarity (embedding distance)
    - Liveness Score (anti-spoofing confidence)
    - Face Quality (blur, brightness, contrast)
    - Candidate Margin (top vs second candidate)
    - Detection Confidence (face detection reliability)

Uncertainty States:
    - HIGH_CONFIDENCE: All signals strong, clear decision
    - LOW_CONFIDENCE: Some signals weak, but decision acceptable
    - UNCERTAIN: Signals conflict or too weak, requires review

Decision Types:
    - ACCEPT: Identity confirmed with confidence
    - REJECT: Identity rejected (unknown or insufficient confidence)
    - REVIEW: Uncertain case requiring human review

Architecture:
    Recognition → Signal Collection → Decision Engine → Decision + Explanation
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum

import numpy as np

from config.settings import Config

logger = logging.getLogger(__name__)


class UncertaintyState(Enum):
    """Uncertainty classification for recognition decisions."""
    HIGH_CONFIDENCE = "high_confidence"
    LOW_CONFIDENCE = "low_confidence"
    UNCERTAIN = "uncertain"


class DecisionType(Enum):
    """Final decision type."""
    ACCEPT = "accept"
    REJECT = "reject"
    REVIEW = "review"


@dataclass
class DecisionSignals:
    """
    Collection of signals for decision making.
    
    All signals should be normalized to [0.0, 1.0] where:
    - 1.0 = excellent/strong signal
    - 0.0 = poor/weak signal
    """
    face_similarity: float = 0.0  # Higher is better (1.0 - distance)
    liveness_score: float = 0.0  # Higher is better
    face_quality: float = 0.0  # Higher is better
    candidate_margin: float = 0.0  # Higher is better (gap between top and second)
    detection_confidence: float = 0.0  # Higher is better
    embedding_stability: float = 1.0  # Higher is better (1.0 = no temporal data yet)
    
    # Raw values for explanation
    raw_distance: float = 0.0
    raw_top_score: float = 0.0
    raw_second_score: float = 0.0
    liveness_checks_passed: int = 0
    liveness_checks_total: int = 0
    
    def to_dict(self) -> Dict:
        """Convert signals to dictionary."""
        return {
            "face_similarity": round(self.face_similarity, 4),
            "liveness_score": round(self.liveness_score, 4),
            "face_quality": round(self.face_quality, 4),
            "candidate_margin": round(self.candidate_margin, 4),
            "detection_confidence": round(self.detection_confidence, 4),
            "embedding_stability": round(self.embedding_stability, 4),
            "raw_distance": round(self.raw_distance, 4),
            "raw_top_score": round(self.raw_top_score, 4),
            "raw_second_score": round(self.raw_second_score, 4),
            "liveness_checks_passed": self.liveness_checks_passed,
            "liveness_checks_total": self.liveness_checks_total,
        }


@dataclass
class DecisionPolicy:
    """
    Configurable decision policy with thresholds and weights.
    
    Thresholds are normalized to [0.0, 1.0].
    """
    # Signal thresholds
    min_face_similarity: float = Config.DECISION_MIN_FACE_SIMILARITY  # Minimum similarity to consider
    min_liveness_score: float = Config.DECISION_MIN_LIVENESS_SCORE  # Minimum liveness to accept
    min_face_quality: float = Config.DECISION_MIN_FACE_QUALITY  # Minimum quality to process
    min_candidate_margin: float = Config.DECISION_MIN_CANDIDATE_MARGIN  # Minimum margin between candidates
    
    # Uncertainty classification thresholds
    high_confidence_threshold: float = Config.DECISION_HIGH_CONFIDENCE_THRESHOLD  # Combined score for HIGH_CONFIDENCE
    low_confidence_threshold: float = Config.DECISION_LOW_CONFIDENCE_THRESHOLD  # Combined score for LOW_CONFIDENCE
    
    # Signal weights (must sum to 1.0)
    weight_similarity: float = Config.DECISION_WEIGHT_SIMILARITY
    weight_liveness: float = Config.DECISION_WEIGHT_LIVENESS
    weight_quality: float = Config.DECISION_WEIGHT_QUALITY
    weight_margin: float = Config.DECISION_WEIGHT_MARGIN
    weight_detection: float = Config.DECISION_WEIGHT_DETECTION
    
    # Special handling
    require_liveness: bool = Config.DECISION_REQUIRE_LIVENESS  # Must pass liveness check
    liveness_min_checks: int = Config.DECISION_LIVENESS_MIN_CHECKS  # Minimum liveness checks to pass
    
    def __post_init__(self):
        """Validate and normalize weights."""
        total_weight = (
            self.weight_similarity + 
            self.weight_liveness + 
            self.weight_quality + 
            self.weight_margin + 
            self.weight_detection
        )
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(
                f"Decision policy weights sum to {total_weight}, normalizing to 1.0"
            )
            # Normalize weights
            self.weight_similarity /= total_weight
            self.weight_liveness /= total_weight
            self.weight_quality /= total_weight
            self.weight_margin /= total_weight
            self.weight_detection /= total_weight
    
    def to_dict(self) -> Dict:
        """Convert policy to dictionary."""
        return {
            "min_face_similarity": self.min_face_similarity,
            "min_liveness_score": self.min_liveness_score,
            "min_face_quality": self.min_face_quality,
            "min_candidate_margin": self.min_candidate_margin,
            "high_confidence_threshold": self.high_confidence_threshold,
            "low_confidence_threshold": self.low_confidence_threshold,
            "weights": {
                "similarity": round(self.weight_similarity, 3),
                "liveness": round(self.weight_liveness, 3),
                "quality": round(self.weight_quality, 3),
                "margin": round(self.weight_margin, 3),
                "detection": round(self.weight_detection, 3),
            },
            "require_liveness": self.require_liveness,
            "liveness_min_checks": self.liveness_min_checks,
        }


@dataclass
class DecisionResult:
    """
    Result of the decision engine with full explanation.
    """
    decision: DecisionType
    uncertainty_state: UncertaintyState
    combined_score: float
    signals: DecisionSignals
    explanation: Dict[str, str]
    signal_breakdown: Dict[str, float]
    processing_time_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert result to dictionary."""
        return {
            "decision": self.decision.value,
            "uncertainty_state": self.uncertainty_state.value,
            "combined_score": round(self.combined_score, 4),
            "signals": self.signals.to_dict(),
            "explanation": self.explanation,
            "signal_breakdown": {
                k: round(v, 4) for k, v in self.signal_breakdown.items()
            },
            "processing_time_ms": round(self.processing_time_ms, 2),
        }


class DecisionEngine:
    """
    Multi-signal decision engine for face recognition.
    
    Combines multiple signals into a unified decision with uncertainty
    classification and explanation generation.
    """
    
    def __init__(self, policy: DecisionPolicy = None):
        """
        Initialize the decision engine.
        
        Args:
            policy: Decision policy configuration (default: sensible defaults)
        """
        self.policy = policy or DecisionPolicy()
        logger.info(
            f"DecisionEngine initialized with policy: "
            f"high_conf={self.policy.high_confidence_threshold}, "
            f"low_conf={self.policy.low_confidence_threshold}"
        )
    
    def make_decision(
        self,
        signals: DecisionSignals,
        user_id: int = -1,
    ) -> DecisionResult:
        """
        Make a decision based on collected signals.
        
        Args:
            signals: Collected decision signals
            user_id: Candidate user ID (-1 if unknown)
        
        Returns:
            DecisionResult with decision, uncertainty, and explanation
        """
        import time
        start_time = time.time()
        
        # Step 1: Apply hard constraints (must-pass checks)
        constraint_result = self._apply_constraints(signals, user_id)
        if constraint_result is not None:
            # Hard constraint failed, return early
            processing_time = (time.time() - start_time) * 1000
            return DecisionResult(
                decision=constraint_result["decision"],
                uncertainty_state=UncertaintyState.UNCERTAIN,
                combined_score=0.0,
                signals=signals,
                explanation=constraint_result["explanation"],
                signal_breakdown={},
                processing_time_ms=processing_time,
            )
        
        # Step 2: Combine signals using weighted sum
        combined_score, signal_breakdown = self._combine_signals(signals)
        
        # Step 3: Classify uncertainty state
        uncertainty_state = self._classify_uncertainty(combined_score, signals)
        
        # Step 4: Determine final decision
        decision = self._determine_decision(
            combined_score, uncertainty_state, user_id, signals
        )
        
        # Step 5: Generate explanation
        explanation = self._generate_explanation(
            decision, uncertainty_state, signals, signal_breakdown
        )
        
        processing_time = (time.time() - start_time) * 1000
        
        result = DecisionResult(
            decision=decision,
            uncertainty_state=uncertainty_state,
            combined_score=combined_score,
            signals=signals,
            explanation=explanation,
            signal_breakdown=signal_breakdown,
            processing_time_ms=processing_time,
        )
        
        logger.debug(
            f"Decision: {decision.value} | Uncertainty: {uncertainty_state.value} | "
            f"Score: {combined_score:.3f} | User: {user_id}"
        )
        
        return result
    
    def _apply_constraints(
        self, signals: DecisionSignals, user_id: int
    ) -> Optional[Dict]:
        """
        Apply hard constraints that must pass for any acceptance.
        
        Returns None if all constraints pass, or early decision dict if failed.
        """
        explanation = {}
        
        # No candidate detected
        if user_id <= 0:
            return {
                "decision": DecisionType.REJECT,
                "explanation": {
                    "primary_reason": "No face match found",
                    "details": "No known face matched the detected face",
                },
            }
        
        # Liveness constraint
        if self.policy.require_liveness:
            if signals.liveness_checks_passed < self.policy.liveness_min_checks:
                return {
                    "decision": DecisionType.REJECT,
                    "explanation": {
                        "primary_reason": "Liveness check failed",
                        "details": (
                            f"Passed {signals.liveness_checks_passed}/"
                            f"{signals.liveness_checks_total} liveness checks "
                            f"(required: {self.policy.liveness_min_checks})"
                        ),
                    },
                }
        
        # Minimum quality constraint
        if signals.face_quality < self.policy.min_face_quality:
            return {
                "decision": DecisionType.REJECT,
                "explanation": {
                    "primary_reason": "Face quality too low",
                    "details": (
                        f"Face quality {signals.face_quality:.2f} below threshold "
                        f"{self.policy.min_face_quality}"
                    ),
                },
            }
        
        # Minimum similarity constraint
        if signals.face_similarity < self.policy.min_face_similarity:
            return {
                "decision": DecisionType.REJECT,
                "explanation": {
                    "primary_reason": "Face similarity too low",
                    "details": (
                        f"Similarity {signals.face_similarity:.2f} below threshold "
                        f"{self.policy.min_face_similarity}"
                    ),
                },
            }
        
        # Minimum candidate margin constraint
        if signals.candidate_margin < self.policy.min_candidate_margin:
            return {
                "decision": DecisionType.REVIEW,
                "explanation": {
                    "primary_reason": "Candidate margin too small",
                    "details": (
                        f"Margin {signals.candidate_margin:.2f} below threshold "
                        f"{self.policy.min_candidate_margin} - ambiguous identity"
                    ),
                },
            }
        
        # All constraints passed
        return None
    
    def _combine_signals(
        self, signals: DecisionSignals
    ) -> Tuple[float, Dict[str, float]]:
        """
        Combine signals using weighted sum.
        
        Returns:
            Tuple of (combined_score, signal_breakdown_dict)
        """
        breakdown = {
            "similarity_contribution": signals.face_similarity * self.policy.weight_similarity,
            "liveness_contribution": signals.liveness_score * self.policy.weight_liveness,
            "quality_contribution": signals.face_quality * self.policy.weight_quality,
            "margin_contribution": signals.candidate_margin * self.policy.weight_margin,
            "detection_contribution": signals.detection_confidence * self.policy.weight_detection,
        }
        
        combined_score = sum(breakdown.values())
        
        # Clamp to [0, 1]
        combined_score = max(0.0, min(1.0, combined_score))
        
        return combined_score, breakdown
    
    def _classify_uncertainty(
        self, combined_score: float, signals: DecisionSignals
    ) -> UncertaintyState:
        """
        Classify the uncertainty state based on combined score and signals.
        """
        # Check for conflicting signals
        has_conflict = self._detect_signal_conflict(signals)
        
        if has_conflict:
            return UncertaintyState.UNCERTAIN
        
        # Classify based on combined score
        if combined_score >= self.policy.high_confidence_threshold:
            return UncertaintyState.HIGH_CONFIDENCE
        elif combined_score >= self.policy.low_confidence_threshold:
            return UncertaintyState.LOW_CONFIDENCE
        else:
            return UncertaintyState.UNCERTAIN
    
    def _detect_signal_conflict(self, signals: DecisionSignals) -> bool:
        """
        Detect if signals are conflicting (e.g., high similarity but low liveness).
        """
        # High similarity but very low liveness
        if signals.face_similarity > 0.8 and signals.liveness_score < 0.4:
            return True
        
        # High similarity but very low quality
        if signals.face_similarity > 0.8 and signals.face_quality < 0.3:
            return True
        
        # Good liveness but very low similarity
        if signals.liveness_score > 0.8 and signals.face_similarity < 0.4:
            return True
        
        return False
    
    def _determine_decision(
        self,
        combined_score: float,
        uncertainty_state: UncertaintyState,
        user_id: int,
        signals: DecisionSignals,
    ) -> DecisionType:
        """
        Determine final decision based on uncertainty state and signals.
        """
        if user_id <= 0:
            return DecisionType.REJECT
        
        if uncertainty_state == UncertaintyState.HIGH_CONFIDENCE:
            return DecisionType.ACCEPT
        
        if uncertainty_state == UncertaintyState.LOW_CONFIDENCE:
            # Low confidence but still acceptable
            return DecisionType.ACCEPT
        
        # Uncertain - requires review
        return DecisionType.REVIEW
    
    def _generate_explanation(
        self,
        decision: DecisionType,
        uncertainty_state: UncertaintyState,
        signals: DecisionSignals,
        signal_breakdown: Dict[str, float],
    ) -> Dict[str, str]:
        """
        Generate human-readable explanation for the decision.
        """
        explanation = {
            "decision_summary": decision.value.upper(),
            "uncertainty_state": uncertainty_state.value.upper(),
            "combined_score": f"{signals.face_similarity + signals.liveness_score + signals.face_quality:.3f}",
        }
        
        # Primary reason
        if decision == DecisionType.ACCEPT:
            explanation["primary_reason"] = "Identity confirmed with acceptable confidence"
        elif decision == DecisionType.REJECT:
            explanation["primary_reason"] = "Identity rejected due to insufficient confidence"
        else:  # REVIEW
            explanation["primary_reason"] = "Identity uncertain - requires review"
        
        # Signal breakdown explanation
        signal_explanations = []
        
        if signals.face_similarity >= 0.8:
            signal_explanations.append(f"Strong face similarity ({signals.face_similarity:.2f})")
        elif signals.face_similarity >= 0.5:
            signal_explanations.append(f"Moderate face similarity ({signals.face_similarity:.2f})")
        else:
            signal_explanations.append(f"Weak face similarity ({signals.face_similarity:.2f})")
        
        if signals.liveness_score >= 0.8:
            signal_explanations.append(f"High liveness confidence ({signals.liveness_score:.2f})")
        elif signals.liveness_score >= 0.6:
            signal_explanations.append(f"Moderate liveness confidence ({signals.liveness_score:.2f})")
        else:
            signal_explanations.append(f"Low liveness confidence ({signals.liveness_score:.2f})")
        
        if signals.face_quality >= 0.8:
            signal_explanations.append(f"Excellent face quality ({signals.face_quality:.2f})")
        elif signals.face_quality >= 0.5:
            signal_explanations.append(f"Good face quality ({signals.face_quality:.2f})")
        else:
            signal_explanations.append(f"Poor face quality ({signals.face_quality:.2f})")
        
        if signals.candidate_margin >= 0.2:
            signal_explanations.append(f"Clear candidate margin ({signals.candidate_margin:.2f})")
        elif signals.candidate_margin >= 0.1:
            signal_explanations.append(f"Adequate candidate margin ({signals.candidate_margin:.2f})")
        else:
            signal_explanations.append(f"Small candidate margin ({signals.candidate_margin:.2f})")
        
        explanation["signal_assessment"] = "; ".join(signal_explanations)
        
        # Secondary factors
        secondary = []
        if signals.candidate_margin < 0.1:
            secondary.append("Low candidate margin increases uncertainty")
        if signals.face_quality < 0.5:
            secondary.append("Poor face quality affects reliability")
        if signals.liveness_score < 0.6:
            secondary.append("Low liveness score raises spoofing concern")
        
        if secondary:
            explanation["secondary_factors"] = "; ".join(secondary)
        
        return explanation
    
    def update_policy(self, **kwargs):
        """
        Update decision policy parameters.
        
        Args:
            **kwargs: Policy parameters to update
        """
        for key, value in kwargs.items():
            if hasattr(self.policy, key):
                setattr(self.policy, key, value)
                logger.info(f"Updated policy parameter: {key} = {value}")
            else:
                logger.warning(f"Unknown policy parameter: {key}")
        
        # Re-validate weights after update
        self.policy.__post_init__()
    
    def get_policy(self) -> DecisionPolicy:
        """Get current decision policy."""
        return self.policy


# Global decision engine instance
decision_engine = DecisionEngine()

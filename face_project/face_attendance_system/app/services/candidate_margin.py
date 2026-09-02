"""
Candidate Margin Analysis
--------------------------
Analyzes the margin between the top recognition candidate and the second-best candidate.
Provides additional confidence signal based on how clearly the top candidate stands out.

Concept:
    If top_score = 0.91 and second_score = 0.42 → large margin → high confidence
    If top_score = 0.91 and second_score = 0.90 → tiny margin → low confidence

Architecture:
    Recognition → Distance Calculation → Margin Analysis → Decision Signal
"""

import logging
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np

from config.settings import Config

logger = logging.getLogger(__name__)


@dataclass
class MarginResult:
    """Result of candidate margin analysis."""
    top_score: float  # Best similarity score (0-1)
    second_score: float  # Second-best similarity score (0-1)
    top_user_id: int  # User ID of top candidate
    top_employee_id: str  # Employee ID of top candidate
    second_user_id: int  # User ID of second candidate
    margin: float  # Difference between top and second (0-1)
    margin_percent: float  # Margin as percentage of top score
    is_clear: bool  # Whether margin is clearly significant
    confidence_level: str  # "high", "medium", "low"
    
    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "top_score": round(self.top_score, 4),
            "second_score": round(self.second_score, 4),
            "top_user_id": self.top_user_id,
            "top_employee_id": self.top_employee_id,
            "second_user_id": self.second_user_id,
            "margin": round(self.margin, 4),
            "margin_percent": round(self.margin_percent, 2),
            "is_clear": self.is_clear,
            "confidence_level": self.confidence_level,
        }


class CandidateMarginAnalyzer:
    """
    Analyzes candidate margin to provide additional confidence signal.
    
    A large margin between top and second candidate indicates a clear match.
    A small margin suggests ambiguity and should reduce confidence.
    """
    
    def __init__(self, clear_margin_threshold: float = None):
        """
        Initialize the margin analyzer.
        
        Args:
            clear_margin_threshold: Minimum margin to consider "clear" (default: from config)
        """
        self.clear_margin_threshold = clear_margin_threshold or Config.CANDIDATE_MARGIN_CLEAR_THRESHOLD
        logger.info(
            f"CandidateMarginAnalyzer initialized "
            f"(clear_margin_threshold={self.clear_margin_threshold})"
        )
    
    def analyze_margin(
        self,
        distances: np.ndarray,
        user_ids: List[int],
        employee_ids: List[str],
        tolerance: float = 0.5,
    ) -> MarginResult:
        """
        Analyze the margin between top and second candidates.
        
        Args:
            distances: Array of distances to all known faces
            user_ids: Corresponding user IDs
            employee_ids: Corresponding employee IDs
            tolerance: Recognition tolerance threshold
        
        Returns:
            MarginResult with analysis details
        """
        if len(distances) == 0:
            return MarginResult(
                top_score=0.0,
                second_score=0.0,
                top_user_id=-1,
                top_employee_id="Unknown",
                second_user_id=-1,
                margin=0.0,
                margin_percent=0.0,
                is_clear=False,
                confidence_level="none",
            )
        
        # Convert distances to similarity scores (1 - distance)
        similarities = 1.0 - distances
        similarities = np.clip(similarities, 0.0, 1.0)
        
        # Sort by similarity (descending)
        sorted_indices = np.argsort(similarities)[::-1]
        
        # Get top and second best
        top_idx = sorted_indices[0]
        top_score = similarities[top_idx]
        top_user_id = user_ids[top_idx]
        top_employee_id = employee_ids[top_idx]
        
        if len(sorted_indices) > 1:
            second_idx = sorted_indices[1]
            second_score = similarities[second_idx]
            second_user_id = user_ids[second_idx]
        else:
            # Only one candidate
            second_score = 0.0
            second_user_id = -1
        
        # Calculate margin
        margin = top_score - second_score
        margin_percent = (margin / top_score * 100) if top_score > 0 else 0.0
        
        # Determine if clear
        is_clear = margin >= self.clear_margin_threshold
        
        # Determine confidence level
        if margin >= 0.25:
            confidence_level = "high"
        elif margin >= self.clear_margin_threshold:
            confidence_level = "medium"
        elif margin > 0.05:
            confidence_level = "low"
        else:
            confidence_level = "very_low"
        
        result = MarginResult(
            top_score=float(top_score),
            second_score=float(second_score),
            top_user_id=top_user_id,
            top_employee_id=top_employee_id,
            second_user_id=second_user_id,
            margin=float(margin),
            margin_percent=float(margin_percent),
            is_clear=bool(is_clear),
            confidence_level=confidence_level,
        )
        
        logger.debug(
            f"Margin analysis: top={top_score:.3f}, second={second_score:.3f}, "
            f"margin={margin:.3f} ({margin_percent:.1f}%), "
            f"confidence={confidence_level}"
        )
        
        return result
    
    def get_margin_signal(self, margin_result: MarginResult) -> float:
        """
        Convert margin result to a normalized signal for decision engine.
        
        Returns:
            Signal value in [0.0, 1.0] where 1.0 = excellent margin
        """
        if margin_result.margin <= 0:
            return 0.0
        
        # Normalize margin to [0, 1] based on clear_margin_threshold
        # Margin >= 2x threshold = 1.0 (excellent)
        # Margin = threshold = 0.5 (adequate)
        # Margin = 0 = 0.0 (poor)
        
        normalized = margin_result.margin / (2 * self.clear_margin_threshold)
        return min(1.0, max(0.0, normalized))
    
    def update_threshold(self, threshold: float):
        """
        Update the clear margin threshold.
        
        Args:
            threshold: New threshold value (0.0 to 1.0)
        """
        if 0.0 <= threshold <= 1.0:
            self.clear_margin_threshold = threshold
            logger.info(f"Updated clear_margin_threshold to {threshold}")
        else:
            logger.warning(f"Invalid threshold {threshold}, must be in [0.0, 1.0]")


# Global margin analyzer instance
candidate_margin_analyzer = CandidateMarginAnalyzer()

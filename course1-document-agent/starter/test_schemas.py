"""
Tests for schema validation in the Document Assistant.
Run with: pytest test_schemas.py -v
"""
import pytest
from datetime import datetime
from pydantic import ValidationError
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from schemas import AnswerResponse, UserIntent


# ─── AnswerResponse Tests ────────────────────────────────────────────────────

class TestAnswerResponse:

    def test_valid_answer_response(self):
        """Valid AnswerResponse should be created without errors."""
        response = AnswerResponse(
            question="What is the total in INV-001?",
            answer="The total is $22,000.",
            sources=["INV-001"],
            confidence=0.95
        )
        assert response.question == "What is the total in INV-001?"
        assert response.confidence == 0.95

    def test_confidence_above_1_rejected(self):
        """Confidence above 1.0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            AnswerResponse(
                question="test",
                answer="test",
                confidence=1.5
            )

    def test_confidence_below_0_rejected(self):
        """Confidence below 0.0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            AnswerResponse(
                question="test",
                answer="test",
                confidence=-0.1
            )

    def test_confidence_boundary_values(self):
        """Confidence of exactly 0.0 and 1.0 should be valid."""
        low = AnswerResponse(question="test", answer="test", confidence=0.0)
        high = AnswerResponse(question="test", answer="test", confidence=1.0)
        assert low.confidence == 0.0
        assert high.confidence == 1.0

    def test_sources_defaults_to_empty_list(self):
        """Sources should default to an empty list."""
        response = AnswerResponse(
            question="test",
            answer="test",
            confidence=0.9
        )
        assert response.sources == []

    def test_timestamp_auto_generated(self):
        """Timestamp should be auto-generated."""
        response = AnswerResponse(
            question="test",
            answer="test",
            confidence=0.9
        )
        assert isinstance(response.timestamp, datetime)


# ─── UserIntent Tests ─────────────────────────────────────────────────────────

class TestUserIntent:

    def test_valid_qa_intent(self):
        """Valid qa intent should be created without errors."""
        intent = UserIntent(
            intent_type="qa",
            confidence=0.95,
            reasoning="User asked a factual question about a document."
        )
        assert intent.intent_type == "qa"

    def test_valid_summarization_intent(self):
        """Valid summarization intent should be accepted."""
        intent = UserIntent(
            intent_type="summarization",
            confidence=0.98,
            reasoning="User explicitly asked to summarize."
        )
        assert intent.intent_type == "summarization"

    def test_valid_calculation_intent(self):
        """Valid calculation intent should be accepted."""
        intent = UserIntent(
            intent_type="calculation",
            confidence=0.97,
            reasoning="User asked to calculate a sum."
        )
        assert intent.intent_type == "calculation"

    def test_valid_unknown_intent(self):
        """Valid unknown intent should be accepted."""
        intent = UserIntent(
            intent_type="unknown",
            confidence=0.3,
            reasoning="Intent is unclear."
        )
        assert intent.intent_type == "unknown"

    def test_invalid_intent_type_rejected(self):
        """Invalid intent type should raise ValidationError."""
        with pytest.raises(ValidationError):
            UserIntent(
                intent_type="invalid_type",
                confidence=0.9,
                reasoning="test"
            )

    def test_confidence_above_1_rejected(self):
        """Confidence above 1.0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            UserIntent(
                intent_type="qa",
                confidence=1.1,
                reasoning="test"
            )

    def test_confidence_below_0_rejected(self):
        """Confidence below 0.0 should raise ValidationError."""
        with pytest.raises(ValidationError):
            UserIntent(
                intent_type="qa",
                confidence=-0.5,
                reasoning="test"
            )

    def test_confidence_boundary_values(self):
        """Confidence of exactly 0.0 and 1.0 should be valid."""
        low = UserIntent(intent_type="unknown", confidence=0.0, reasoning="test")
        high = UserIntent(intent_type="qa", confidence=1.0, reasoning="test")
        assert low.confidence == 0.0
        assert high.confidence == 1.0

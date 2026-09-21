"""Trivial Majority Baseline for Intent Classification and Triage Handling.

Predicts the most frequent intent and handling classes observed in the
development training corpus without inspecting the customer message.
Establishes the empirical lower performance bound.
"""

from collections import Counter
from typing import List, Optional, Union, Dict, Any
import numpy as np


class MajorityBaseline:
    """Trivial heuristic baseline predicting majority class labels."""

    def __init__(self):
        self.majority_intent: Optional[str] = None
        self.majority_handling: Optional[str] = None
        self.intent_frequencies: Dict[str, int] = {}
        self.handling_frequencies: Dict[str, int] = {}

    def fit(self, y_intents: List[str], y_handling: Optional[List[str]] = None) -> "MajorityBaseline":
        """Learn majority intent and handling from development training data."""
        if not y_intents:
            raise ValueError("y_intents cannot be empty")

        self.intent_frequencies = dict(Counter(y_intents))
        self.majority_intent = max(self.intent_frequencies.items(), key=lambda x: x[1])[0]

        if y_handling:
            self.handling_frequencies = dict(Counter(y_handling))
            self.majority_handling = max(self.handling_frequencies.items(), key=lambda x: x[1])[0]
        else:
            self.majority_handling = "AUTO_HANDLE"

        return self

    def predict_intent(self, X: Union[List[str], np.ndarray]) -> List[str]:
        """Predict the majority intent for all input samples."""
        if self.majority_intent is None:
            raise RuntimeError("MajorityBaseline must be fit before calling predict_intent()")
        n_samples = len(X)
        return [self.majority_intent] * n_samples

    def predict_handling(self, X: Union[List[str], np.ndarray]) -> List[str]:
        """Predict the majority handling action for all input samples."""
        if self.majority_handling is None:
            raise RuntimeError("MajorityBaseline must be fit before calling predict_handling()")
        n_samples = len(X)
        return [self.majority_handling] * n_samples

    def predict(self, X: Union[List[str], np.ndarray]) -> Dict[str, List[str]]:
        """Predict both intent and handling for all inputs."""
        return {
            "intents": self.predict_intent(X),
            "handling": self.predict_handling(X),
        }

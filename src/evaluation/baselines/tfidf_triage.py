"""Classical Machine Learning Triage / Escalation Baseline using TF-IDF + Logistic Regression."""

from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TFIDFTriageClassifier:
    """TF-IDF + Logistic Regression Binary Triage Classifier (AUTO_HANDLE vs. ESCALATE)."""

    def __init__(
        self,
        max_features: int = 3000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        C: float = 1.0,
        class_weight: Optional[str] = "balanced",
        random_state: int = 42,
    ):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
        )
        self.classifier = LogisticRegression(
            C=C,
            max_iter=1000,
            class_weight=class_weight,
            random_state=random_state,
        )
        self.is_fitted = False
        self.classes_: Optional[np.ndarray] = None

    def fit(self, X: List[str], y: List[str]) -> "TFIDFTriageClassifier":
        """Fit the triage classifier on training data."""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data X and y cannot be empty")
        
        X_clean = [str(t) if t is not None else "" for t in X]
        X_vec = self.vectorizer.fit_transform(X_clean)
        self.classifier.fit(X_vec, y)
        self.classes_ = self.classifier.classes_
        self.is_fitted = True
        return self

    def predict(self, X: List[str]) -> List[str]:
        """Predict handling decisions ('AUTO_HANDLE' vs. 'ESCALATE')."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict()")
        X_clean = [str(t) if t is not None else "" for t in X]
        X_vec = self.vectorizer.transform(X_clean)
        preds = self.classifier.predict(X_vec)
        return list(preds)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba()")
        X_clean = [str(t) if t is not None else "" for t in X]
        X_vec = self.vectorizer.transform(X_clean)
        return self.classifier.predict_proba(X_vec)

    def predict_with_confidence(self, X: List[str]) -> List[Dict[str, Any]]:
        """Return triage predictions with probability confidence."""
        probs = self.predict_proba(X)
        preds = self.predict(X)
        results = []
        for pred, prob_row in zip(preds, probs):
            conf = float(np.max(prob_row))
            results.append({
                "handling": pred,
                "confidence": round(conf, 4),
            })
        return results

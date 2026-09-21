"""Classical Machine Learning Intent Classifier Baseline using TF-IDF + Logistic Regression."""

from typing import List, Union, Optional, Tuple, Dict, Any
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class TFIDFIntentClassifier:
    """TF-IDF + Logistic Regression Intent Classifier."""

    def __init__(
        self,
        max_features: int = 5000,
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

    def fit(self, X: List[str], y: List[str]) -> "TFIDFIntentClassifier":
        """Fit the TF-IDF vectorizer and Logistic Regression classifier on training data."""
        if len(X) == 0 or len(y) == 0:
            raise ValueError("Training data X and y cannot be empty")
        
        # Clean inputs
        X_clean = [str(t) if t is not None else "" for t in X]
        X_vec = self.vectorizer.fit_transform(X_clean)
        self.classifier.fit(X_vec, y)
        self.classes_ = self.classifier.classes_
        self.is_fitted = True
        return self

    def predict(self, X: List[str]) -> List[str]:
        """Predict intent labels for input text."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict()")
        X_clean = [str(t) if t is not None else "" for t in X]
        X_vec = self.vectorizer.transform(X_clean)
        preds = self.classifier.predict(X_vec)
        return list(preds)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        """Predict intent probability distribution for input text."""
        if not self.is_fitted:
            raise RuntimeError("Classifier must be fitted before predict_proba()")
        X_clean = [str(t) if t is not None else "" for t in X]
        X_vec = self.vectorizer.transform(X_clean)
        return self.classifier.predict_proba(X_vec)

    def predict_with_confidence(self, X: List[str]) -> List[Dict[str, Any]]:
        """Return predictions paired with maximum class probability confidence."""
        probs = self.predict_proba(X)
        preds = self.predict(X)
        results = []
        for pred, prob_row in zip(preds, probs):
            conf = float(np.max(prob_row))
            results.append({
                "intent": pred,
                "confidence": round(conf, 4),
            })
        return results

    def predict_detailed(self, X: List[str]) -> List[Dict[str, Any]]:
        """Return comprehensive predictions with probabilities and sorted top-3 intents."""
        probs = self.predict_proba(X)
        preds = self.predict(X)
        results = []
        classes = list(self.classes_) if self.classes_ is not None else []

        for pred, prob_row in zip(preds, probs):
            prob_dict = {cls_name: round(float(prob), 4) for cls_name, prob in zip(classes, prob_row)}
            sorted_classes = sorted(prob_dict.items(), key=lambda item: item[1], reverse=True)
            top3 = [{"intent": item[0], "probability": item[1]} for item in sorted_classes[:3]]
            conf = float(np.max(prob_row))

            results.append({
                "predicted_intent": pred,
                "confidence": round(conf, 4),
                "probabilities": prob_dict,
                "top3": top3,
            })
        return results


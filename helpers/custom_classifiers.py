from sklearn.base import BaseEstimator, ClassifierMixin
import numpy as np
import pandas as pd


class MarkovPredictor(BaseEstimator, ClassifierMixin):
    """    A custom classifier that predicts the next state based on a transition matrix."""
    def __init__(self, transition_matrix) -> None:
        assert isinstance(transition_matrix, pd.DataFrame), "Transition matrix must be a pandas DataFrame."
        self.transition_matrix_ = transition_matrix
        self.classes_ = sorted(transition_matrix.index.tolist())
    
    def fit(self, transition_matrix):
        raise ValueError("No fit required. Transition matrix must be provided when creating classifier.")
    
    def predict(self, X):
        # X contains the penultimate states
        predictions = []
        predictions = self.predict_proba(X).argmax(axis=1).tolist()
        return np.array(predictions)
    
    def predict_proba(self, X):
        # Vectorized approach: get all transition probabilities at once
        transition_probs = self.transition_matrix_.loc[self.classes_, X]
        return transition_probs.T.values
    def binarice_true_labels(self, y_true):
        return np.array([self.classes_.index(y) for y in y_true])
from config import MODEL_PATH
import joblib
import numpy as np


class GenrePredictor:
    """Handles prediction of genres based on extracted features."""

    def __init__(self):
        self.model = joblib.load(MODEL_PATH)

    def predict_genre(self, features_df):
        genre_probabilities = self.model.predict_proba(features_df)[0]
        top_indices = np.argsort(genre_probabilities)[-3:][::-1]
        top_probabilities = genre_probabilities[top_indices]
        top_genres = self.model.classes_[top_indices]
        return list(zip(top_genres, top_probabilities))

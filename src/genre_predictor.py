import joblib
import numpy as np

class GenrePredictor:
    def __init__(self, model_path):
        self.model = self._load_model(model_path)

    @staticmethod
    def _load_model(model_path):
        return joblib.load(model_path)

    def predict_top_genres(self, features_df, top_n=3):
        probabilities = self.model.predict_proba(features_df)[0]
        top_indices = np.argsort(probabilities)[-top_n:][::-1]

        class_labels = self.model.classes_
        top_genres = [(class_labels[i], probabilities[i]) for i in top_indices]
        return top_genres


import streamlit as st
from src.audio_feature_extractor import AudioFeatureExtractor
from src.genre_predictor import GenrePredictor
from src.file_handler import FileHandler
from config import MODEL_PATH

class GenrePredictionApp:
    def __init__(self, model_path):
        self.extractor = AudioFeatureExtractor()
        self.predictor = GenrePredictor(model_path)

    def run(self):
        st.title("Genre Prediction from Audio File")
        st.write(
            "Upload an audio file (MP3 or WAV) to predict the genre based on its audio features."
        )

        uploaded_file = st.file_uploader("Choose an audio file:", type=["mp3", "wav"])

        if uploaded_file:
            temp_file_path = "temp_audio_file"
            FileHandler.save_uploaded_file(uploaded_file, temp_file_path)
            st.audio(uploaded_file, format="audio/wav")

            # Process and predict
            features_df = self.extractor.extract_features(temp_file_path)
            top_genres = self.predictor.predict_top_genres(features_df)

            # Display results
            st.write("Top 3 Predicted Genres and their Probabilities:")
            for genre, prob in top_genres:
                st.write(f"{genre}: {prob * 100:.2f}%")


if __name__ == "__main__":
    app = GenrePredictionApp(model_path=MODEL_PATH)
    app.run()

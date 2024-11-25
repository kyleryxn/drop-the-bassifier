import streamlit as st
from feature_extractor import AudioFeatureExtractor
from genre_predictor import GenrePredictor


class GenrePredictionApp:
    """Handles the Streamlit app interface."""

    def __init__(self):
        self.genre_predictor = GenrePredictor()

    def run(self):
        st.title("Genre Prediction from Audio File")
        st.write("Upload an audio file (mp3 or wav) to predict the genre based on its audio features.")

        uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])

        if uploaded_file is not None:
            temp_file_path = "temp_audio_file"
            st.audio(uploaded_file, format="audio/wav")
            with open(temp_file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Extract features
            feature_extractor = AudioFeatureExtractor(temp_file_path)
            features_df = feature_extractor.extract_features()

            # Predict genres
            top_genres = self.genre_predictor.predict_genre(features_df)

            # Display results
            st.write("Top 3 Predicted Genres and their Probabilities:")
            for genre, prob in top_genres:
                st.write(f"{genre}: {prob * 100:.2f}%")

import streamlit as st
from feature_extractor import AudioFeatureExtractor
from genre_predictor import GenrePredictor


def build_genre_map():
    """Dynamically builds a genre map with default formatting and manual overrides."""
    # Example set of genres from your dataset or model predictions
    all_genres = ["hiphop", "classic-rock", "electronic-dance", "pop", "jazz"]

    # Default formatter for genres
    def format_genre_name(song_genre):
        return "-".join(part.title() for part in song_genre.split("-"))

    # Build the map with default formatting
    genre_map = {genre: format_genre_name(genre) for genre in all_genres}

    # Manually override specific genres
    genre_map["hiphop"] = "Hip-Hop"

    return genre_map


class GenrePredictionApp:
    """Handles the Streamlit app interface."""

    def __init__(self):
        self.genre_predictor = GenrePredictor()
        self.genre_map = build_genre_map()

    def run(self):
        st.title("Genre Prediction from Audio File")
        st.write("Upload an audio file (MP3 or WAV) to predict the genre based on its audio features.")

        uploaded_file = st.file_uploader("Choose an audio file:", type=["mp3", "wav"])

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

            # Define colors for the buttons (darkest to lightest shades of blue)
            button_colors = [
                "background-color: rgb(14 85 177); color: white; border-radius: 10px; padding: 10px;",
                "background-color: rgb(32 78 137 / 75%); color: white; border-radius: 10px; padding: 10px;",
                "background-color: rgb(36 64 101 / 50%); color: white; border-radius: 10px; padding: 10px;"
            ]

            cols = st.columns(len(top_genres))  # Create equal-width columns for each genre

            for idx, (genre, prob) in enumerate(top_genres):
                formatted_genre = self.genre_map.get(genre, genre.replace("-", " ").title())
                with cols[idx]:  # Place each styled button in its respective column
                    # Use st.markdown to render styled buttons
                    button_html = f"""
                    <div style="{button_colors[idx]} text-align: center;">
                        {formatted_genre}: {prob * 100:.2f}%
                    </div>
                    """
                    st.markdown(button_html, unsafe_allow_html=True)
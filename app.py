import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
# Function to extract features from the audio file
def extract_features(file_path):
    """
    Extracts features from the given audio file.
    Returns a DataFrame with the extracted features.
    """
    # Load audio
    y, sr = librosa.load(file_path, duration=30)  # Load 30 seconds of audio
    
    # Chroma STFT
    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_stft_mean = np.mean(chroma_stft)
    chroma_stft_var = np.var(chroma_stft)
    
    # RMS
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    
    # Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)
    
    # Spectral Bandwidth
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)
    
    # Spectral Rolloff
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)
    
    # Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    zero_crossing_rate_var = np.var(zero_crossing_rate)
    
    # Harmony
    harmony = librosa.effects.harmonic(y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)
    
    # Perceptual Features (using spectral flatness as an approximation)
    perceptual = librosa.feature.spectral_flatness(y=y)
    perceptual_mean = np.mean(perceptual)
    perceptual_var = np.var(perceptual)
    
    # Tempo (beats per minute)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    
    # MFCC (Mel-frequency cepstral coefficients)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_means = np.mean(mfcc, axis=1)  # mean for each MFCC coefficient
    mfcc_vars = np.var(mfcc, axis=1)   # variance for each MFCC coefficient

    # Combine all features into a list, pairing mean and variance for each feature
    features = [
        chroma_stft_mean, chroma_stft_var,
        rms_mean, rms_var,
        spectral_centroid_mean, spectral_centroid_var,
        spectral_bandwidth_mean, spectral_bandwidth_var,
        rolloff_mean, rolloff_var,
        zero_crossing_rate_mean, zero_crossing_rate_var,
        harmony_mean, harmony_var,
        perceptual_mean, perceptual_var,
        tempo
    ]
    
    # Add MFCC features (mean and variance for each MFCC coefficient, in order)
    for i in range(20):  # Assuming 20 MFCCs are calculated
        features.append(mfcc_means[i])  # MFCC mean
        features.append(mfcc_vars[i])   # MFCC variance
    
    # Create a DataFrame with the features
    feature_names = [
        'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
        'spectral_centroid_mean', 'spectral_centroid_var',
        'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
        'rolloff_var', 'zero_crossing_rate_mean', 'zero_crossing_rate_var',
        'harmony_mean', 'harmony_var', 'perceptr_mean', 'perceptr_var', 'tempo',
        'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc2_var', 'mfcc3_mean',
        'mfcc3_var', 'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var',
        'mfcc6_mean', 'mfcc6_var', 'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean',
        'mfcc8_var', 'mfcc9_mean', 'mfcc9_var', 'mfcc10_mean', 'mfcc10_var',
        'mfcc11_mean', 'mfcc11_var', 'mfcc12_mean', 'mfcc12_var', 'mfcc13_mean',
        'mfcc13_var', 'mfcc14_mean', 'mfcc14_var', 'mfcc15_mean', 'mfcc15_var',
        'mfcc16_mean', 'mfcc16_var', 'mfcc17_mean', 'mfcc17_var', 'mfcc18_mean',
        'mfcc18_var', 'mfcc19_mean', 'mfcc19_var', 'mfcc20_mean', 'mfcc20_var'
    ]
    
    # Return a DataFrame with the correct columns
    features_df = pd.DataFrame([features], columns=feature_names)
    
    return features_df


# Function to predict the genre of a song
def predict_genre(file_path):
    # Load pre-trained model and scaler
    model = joblib.load('notebooks/model_training/genre_model.joblib')
    
    # Extract features from the audio file
    features_df = extract_features(file_path)

    # Scale the features

    print(features_df)
    # Predict the genre
    genre_probabilities = model.predict_proba(features_df)[0]  # Get probabilities for all classes

    top_indices = np.argsort(genre_probabilities)[-3:][::-1]  # Get the indices of the top 3 probabilities
    top_probabilities = genre_probabilities[top_indices]
    
    # Get the genre labels (assuming you have the class labels stored)
    class_labels = model.classes_  # This gives you the class labels used by the model
    
    # Get the top 3 predicted genres
    top_genres = class_labels[top_indices]
    
    return list(zip(top_genres, top_probabilities))
    
   


# Streamlit app
def main():
    st.title("Genre Prediction from Audio File")
    
    st.write(
        "Upload an audio file (mp3 or wav) to predict the genre based on its audio features."
    )
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an audio file", type=["mp3", "wav"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        temp_file_path = "temp_audio_file"
        st.audio(uploaded_file, format="audio/wav")
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Predict genre
        top_genres = predict_genre(temp_file_path)
        st.write("Top 3 Predicted Genres and their Probabilities:")
        for genre, prob in top_genres:
            st.write(f"{genre}: {prob * 100:.2f}%")


if __name__ == "__main__":
    main()
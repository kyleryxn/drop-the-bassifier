import streamlit as st
import librosa
import numpy as np
import pandas as pd
import joblib
import os

# Load the pre-trained model and scaler
model = joblib.load('notebooks/model_training/knn_model.joblib')
scaler = joblib.load('notebooks/model_training/scaler.joblib')

# Function to extract audio features
def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30, offset=30, sr=22050)
    
    # Extracting MFCCs
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    
    mfcc_means = [np.mean(mfccs[idx, :]) for idx in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 17]]
    mfcc_vars = [np.var(mfccs[idx, :]) for idx in [1, 3, 4, 5, 6, 7, 8]]
    
    # Extract RMS, chroma, spectral features, zero-crossing rate, and harmony
    rms = librosa.feature.rms(y=y)
    rms_mean = np.mean(rms)
    rms_var = np.var(rms)
    
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma)
    chroma_var = np.var(chroma)
    
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    spectral_centroid_var = np.var(spectral_centroid)
    
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    spectral_bandwidth_mean = np.mean(spectral_bandwidth)
    spectral_bandwidth_var = np.var(spectral_bandwidth)
    
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    rolloff_mean = np.mean(rolloff)
    rolloff_var = np.var(rolloff)
    
    zcr = librosa.feature.zero_crossing_rate(y)
    zcr_mean = np.mean(zcr)
    
    harmony = librosa.effects.harmonic(y)
    harmony_mean = np.mean(harmony)
    harmony_var = np.var(harmony)
    
    
    
    # Combine all features into a single feature vector
    features = np.hstack([
        chroma_mean, chroma_var,
        rms_mean, rms_var,
        spectral_centroid_mean, spectral_centroid_var,
        spectral_bandwidth_mean, spectral_bandwidth_var,
        rolloff_mean, rolloff_var,
        zcr_mean,
        harmony_mean, harmony_var,
        mfcc_means[0], mfcc_vars[0],
        mfcc_means[1],
        mfcc_means[2], mfcc_vars[1],
        mfcc_means[3], mfcc_vars[2],
        mfcc_means[4], mfcc_vars[3],
        mfcc_means[5], mfcc_vars[4],
        mfcc_means[6], mfcc_vars[5],
        mfcc_means[7], mfcc_vars[6],
        mfcc_means[8],
        mfcc_means[9],
        mfcc_means[10],
        mfcc_means[11],
        mfcc_means[12],
        mfcc_means[13],
        mfcc_means[14],
    ])
    
    # Ensure exactly 37 features
   
    
    return features

# Streamlit app
def main():
    st.title("Song Genre Classifier")
    
    uploaded_file = st.file_uploader("Upload a .wav or .mp3 file", type=["wav", "mp3"])
    
    if uploaded_file is not None:
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        st.audio(uploaded_file, format="audio/wav")
        
        # Extract features from the uploaded file
        features = extract_features(uploaded_file.name)
        
        # Define 37 feature names
        feature_names = [
    'chroma_stft_mean', 'chroma_stft_var', 'rms_mean', 'rms_var',
    'spectral_centroid_mean', 'spectral_centroid_var',
    'spectral_bandwidth_mean', 'spectral_bandwidth_var', 'rolloff_mean',
    'rolloff_var', 'zero_crossing_rate_mean', 'harmony_mean', 'harmony_var',
    'mfcc1_mean', 'mfcc1_var', 'mfcc2_mean', 'mfcc3_mean', 'mfcc3_var',
    'mfcc4_mean', 'mfcc4_var', 'mfcc5_mean', 'mfcc5_var', 'mfcc6_mean', 'mfcc6_var', 
    'mfcc7_mean', 'mfcc7_var', 'mfcc8_mean', 'mfcc8_var', 'mfcc9_mean',
    'mfcc10_mean', 'mfcc11_mean', 'mfcc12_mean', 'mfcc13_mean', 
    'mfcc15_mean', 'mfcc17_mean'
]
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([features], columns=feature_names)
        
        # Scale features
        scaled_features = scaler.transform(features_df)
        features_df = pd.DataFrame(scaled_features, columns=feature_names)
        
        # Predict genre
        prediction = model.predict(features_df)
        st.write(f"Predicted Genre: {prediction[0]}")
        
        # Remove temporary file
        os.remove(uploaded_file.name)

if __name__ == '__main__':
    main()
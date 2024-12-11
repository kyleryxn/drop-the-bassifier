import librosa
import numpy as np
import pandas as pd


class AudioFeatureExtractor:
    """Handles extraction of audio features from an audio file."""

    def __init__(self, file_path):
        self.file_path = file_path
        self.y, self.sr = librosa.load(file_path, duration=30)

    @staticmethod
    def _calculate_statistics(feature):
        return np.mean(feature), np.var(feature)

    def extract_features(self):
        features = {}

        # Chroma STFT
        chroma_stft = librosa.feature.chroma_stft(y=self.y, sr=self.sr)
        features['chroma_stft_mean'], features['chroma_stft_var'] = self._calculate_statistics(chroma_stft)

        # RMS
        rms = librosa.feature.rms(y=self.y)
        features['rms_mean'], features['rms_var'] = self._calculate_statistics(rms)

        # Spectral Centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=self.y, sr=self.sr)
        features['spectral_centroid_mean'], features['spectral_centroid_var'] = self._calculate_statistics(spectral_centroid)

        # Spectral Bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=self.y, sr=self.sr)
        features['spectral_bandwidth_mean'], features['spectral_bandwidth_var'] = self._calculate_statistics(spectral_bandwidth)

        # Spectral Rolloff
        rolloff = librosa.feature.spectral_rolloff(y=self.y, sr=self.sr)
        features['rolloff_mean'], features['rolloff_var'] = self._calculate_statistics(rolloff)

        # Zero Crossing Rate
        zero_crossing_rate = librosa.feature.zero_crossing_rate(self.y)
        features['zero_crossing_rate_mean'], features['zero_crossing_rate_var'] = self._calculate_statistics(zero_crossing_rate)

        # Harmony
        harmony = librosa.effects.harmonic(self.y)
        features['harmony_mean'], features['harmony_var'] = self._calculate_statistics(harmony)

        # Perceptual Features
        perceptual = librosa.feature.spectral_flatness(y=self.y)
        features['perceptr_mean'], features['perceptr_var'] = self._calculate_statistics(perceptual)

        # Tempo
        onset_env = librosa.onset.onset_strength(y=self.y, sr=self.sr)
        features['tempo'], _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=self.sr)

        # MFCC
        mfcc = librosa.feature.mfcc(y=self.y, sr=self.sr, n_mfcc=20)
        for i in range(20):
            mean, var = self._calculate_statistics(mfcc[i])
            features[f'mfcc{i+1}_mean'] = mean
            features[f'mfcc{i+1}_var'] = var

        return pd.DataFrame([features])

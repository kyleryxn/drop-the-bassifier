import librosa
import numpy as np

class AudioFeatureExtractor:
    def __init__(self, duration=30):
        self.duration = duration

    def extract_features(self, file_path):
        y, sr = librosa.load(file_path, duration=self.duration)

        features = {
            "chroma_stft_mean": self._mean_var(librosa.feature.chroma_stft(y=y, sr=sr)),
            "rms_mean": self._mean_var(librosa.feature.rms(y=y)),
            "spectral_centroid_mean": self._mean_var(librosa.feature.spectral_centroid(y=y, sr=sr)),
            "tempo": self._tempo(y, sr),
        }
        return features

    @staticmethod
    def _mean_var(feature):
        return np.mean(feature), np.var(feature)

    @staticmethod
    def _tempo(y, sr):
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo, _ = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
        return tempo

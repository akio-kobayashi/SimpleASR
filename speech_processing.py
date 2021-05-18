import librosa
import numpy as np

def compute_melspec(path):
    data, sr = librosa.load(path, sr=16000)
    mel = np.log(librosa.feature.melspectrogram(y=data, sr=sr, n_mels=20, n_fft=512, win_length=400, hop_length=160))

    return np.transpose(mel)

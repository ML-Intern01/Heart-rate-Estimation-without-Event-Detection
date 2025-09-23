#Feature extraction for both Train and test folders

import os
import librosa
import numpy as np

input_root = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\preprocessing"
output_root = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\Feature extraction"
os.makedirs(output_root, exist_ok=True)

# Parameters
sr = 1000 
n_fft = sr // 2
win_length = 50                  # Each FFT window covers 50 samples (at 1000 Hz, that's 50 ms)
hop_length = 10                  # The window moves forward by 10 samples (10 ms) each step
n_mels = 128
f_min = 20
f_max = 450

def extract_and_save_mel(wav_path, save_path):
    try:
        y, _ = librosa.load(wav_path, sr=sr)

        mel_spec = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=n_fft, hop_length=hop_length,
            win_length=win_length, n_mels=n_mels, fmin=f_min,
            fmax=f_max, power=2.0
        )

        mel_db = librosa.power_to_db(mel_spec, ref=np.max)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        np.save(save_path, mel_db[:, :-1].T)  # Shape: (time, n_mels)
        print(f" Saved: {save_path}, Shape: {mel_db.shape}")

    except Exception as e:
        print(f" Error processing {wav_path}: {e}")

for root, _, files in os.walk(input_root):
    for fname in files:
        if fname.endswith(".wav"):
            wav_path = os.path.join(root, fname)

            relative_path = os.path.relpath(wav_path, input_root)
            save_path = os.path.join(output_root, relative_path).replace(".wav", ".npy")

            extract_and_save_mel(wav_path, save_path)

print(" All Mel spectrogram features extracted and saved.")



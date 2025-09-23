# -*- coding: utf-8 -*-
"""
Created on Fri Jun 13 15:02:35 2025

@author: AIML
"""

import os
import librosa
import numpy as np
import tensorflow as tf
from collections import Counter
from scipy import signal
from sklearn.preprocessing import StandardScaler


model_path = ""  
input_audio_path = ""  
target_sr = 1000  # Target sample rate after resampling
chunk_duration = 5


# Filtering
def butter_bandpass_filter_stable(data, lowcut=20, highcut=450, fs=1000, order=4):
    sos = signal.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)

# Resampling
def resample_audio(data, original_sr, target_sr=1000):
    if original_sr != target_sr:
        return librosa.resample(y=data, orig_sr=original_sr, target_sr=target_sr)
    return data

# Z-normalization
def z_normalize(y):
    scaler = StandardScaler()
    y = y.reshape(-1, 1)
    return scaler.fit_transform(y).flatten()

# Mel-spectrogram
def extract_features(chunk, original_sr):
    y = butter_bandpass_filter_stable(chunk, fs=original_sr)
    y = resample_audio(y, original_sr, target_sr=target_sr)
    y = z_normalize(y)

    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=target_sr, n_fft=target_sr // 2,
        hop_length=2, win_length=50,
        n_mels=128, fmin=20, fmax=450, power=2.0
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    mel_db_T = mel_db.T[:-1, :]  # shape: (2500, 128)
    return mel_db_T

# Chunking 
def chunks(input_file, chunk_duration=5, max_duration=30):
    y, sr = librosa.load(input_file, sr=None)
    max_samples = int(max_duration * sr)
    y = y[:max_samples]                      #Keeps only 30 sec 

    chunk_size = int(chunk_duration * sr)
    total_samples = len(y)
    num_chunks = total_samples // chunk_size

    chunks = [y[i * chunk_size: (i + 1) * chunk_size] for i in range(num_chunks)]
    return chunks, sr

# Predict chunks directly
def predict_chunks(model, chunks, original_sr):
    predictions = []
    for i, chunk in enumerate(chunks):
        try:
            features = extract_features(chunk, original_sr)
            input_tensor = np.expand_dims(features, axis=0)
            pred = model.predict(input_tensor, verbose=0)
            class_index = np.argmax(pred)
            predictions.append(class_index)
            print(f"Predicted chunk {i}: Class {class_index}")
        except Exception as e:
            print(f"Failed to process chunk {i}: {e}")
    return predictions


# Final predictions
def get_final_prediction(predictions):
    if not predictions:
        return "No prediction"
    count = Counter(predictions)
    most_common = count.most_common(1)[0][0]
    return most_common


if __name__ == "__main__":
    if not model_path or not input_audio_path:
        print("Provide a correct path.")
    else:
        model = tf.keras.models.load_model(model_path)

        print(f"Loading and chunking audio: {input_audio_path}")
        chunks, sr = chunks(input_audio_path, chunk_duration=chunk_duration)

        print(f"Processing {len(chunks)} chunks")
        predictions = predict_chunks(model, chunks, original_sr=sr)

        final_class = get_final_prediction(predictions)
        print(f"\n Final Predicted BPM Class: {final_class}")

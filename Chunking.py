# -*- coding: utf-8 -*-
"""
Created on Thu May 22 11:42:06 2025

@author: AIML
"""

#chunk with 70% overlap

import os
import librosa
import soundfile as sf
import numpy as np

def chunk_audio(input_dir, output_dir, chunk_duration=5, overlap=0.7):
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3')):  
                file_path = os.path.join(root, file)
                signal, sample_rate = librosa.load(file_path, sr=None)

                total_samples = len(signal)
                chunk_size = int(chunk_duration * sample_rate)
                step_size = int(chunk_size * (1 - overlap))

                usable_length = (total_samples // chunk_size) * chunk_size
                remaining_samples = total_samples - usable_length

                if remaining_samples >= (chunk_size // 2):  
                    padding_needed = chunk_size - remaining_samples
                    signal = np.pad(signal, (0, padding_needed)) 
                    print(f"Padding applied: {padding_needed} samples for {file}")

                relative_folder = os.path.relpath(root, input_dir)
                save_folder = os.path.join(output_dir, relative_folder)
                os.makedirs(save_folder, exist_ok=True)

                file_stem = os.path.splitext(file)[0]
                index = 0

                for start in range(0, len(signal) - chunk_size + 1, step_size):
                    end = start + chunk_size
                    segment = signal[start:end]

                    # Handle last chunk (pad if needed)
                    if len(segment) < chunk_size:
                        padding_needed = chunk_size - len(segment)
                        if padding_needed > 3 * sample_rate:  # If padding needed is greater than 3 sec
                            segment = np.pad(segment, (0, padding_needed))  # Pad with silence (zeros)
                            print(f"Padded last chunk for {file_stem}_{index} with {padding_needed} samples.")
                        else:
                            print(f"Skipping last chunk for {file_stem}_{index} as it is too short.")
                            continue

                    new_filename = f"{file_stem}_{index}.wav"
                    sf.write(os.path.join(save_folder, new_filename), segment, sample_rate)
                    print(f"Saved: {new_filename}")
                    index += 1

if __name__ == "__main__":
    input_dir = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\Split data"
    output_dir = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\chunk"

    os.makedirs(output_dir, exist_ok=True)
    chunk_audio(input_dir, output_dir, chunk_duration=5, overlap=0.7)
    
print("All file are chunked")

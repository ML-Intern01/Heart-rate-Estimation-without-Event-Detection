#Preprocessing the data
#Data is split before and then chunked and then preprocessed for Train and test folders

import os
import librosa
import scipy.signal as signal
from sklearn.preprocessing import StandardScaler
import soundfile as sf
import logging

# Set up logging
#logging.basicConfig(filename="audio_processing.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Bandpass Filter
def butter_bandpass_filter_stable(data, lowcut, highcut, fs, order=4):
    sos = signal.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)

# Resample Audio
def resample_audio(data, original_sr, target_sr=1000):
    if original_sr != target_sr:
        logging.debug(f"Resampling audio from {original_sr} Hz to {target_sr} Hz")
        return librosa.resample(y=data, orig_sr=original_sr, target_sr=target_sr)
    return data
 
# Z-Normalization
# def z_normalize(data):
#     scaler = StandardScaler()
#     return scaler.fit_transform(data.reshape(-1, 1))

def process_audio(input_path, output_path, lowcut=20, highcut=450, target_sr=1000):
    for dataset_type in ['train', 'test']:
        dataset_input = os.path.join(input_path, dataset_type)
        dataset_output = os.path.join(output_path, dataset_type)

        if not os.path.exists(dataset_input):
            logging.warning(f"Missing folder: {dataset_input}")
            continue

        # Walk through BPM folders
        for bpm_folder in os.listdir(dataset_input):
            bpm_input_path = os.path.join(dataset_input, bpm_folder)
            bpm_output_path = os.path.join(dataset_output, bpm_folder)

            if not os.path.isdir(bpm_input_path):
                continue

            os.makedirs(bpm_output_path, exist_ok=True)

            for filename in os.listdir(bpm_input_path):
                if not filename.lower().endswith(".wav"):
                    continue

                file_path = os.path.join(bpm_input_path, filename)
                output_file_path = os.path.join(bpm_output_path, filename)

                try:
                    # Load, filter, resample, normalize
                    audio, sr = librosa.load(file_path, sr=None)
                    filtered = butter_bandpass_filter_stable(audio, lowcut, highcut, sr)
                    resampled = resample_audio(filtered, sr, target_sr)
                    # normalized = z_normalize(resampled)

                    sf.write(output_file_path, resampled, target_sr)
                    print(f" Processed: {output_file_path}")
                    logging.info(f"Processed: {output_file_path}")

                except Exception as e:
                    print(f" Error processing {file_path}: {e}")
                    logging.error(f"Error processing {file_path}: {e}")


input_path = r"D:\H_R_EwED\chunk"
output_path = r"D:\H_R_EwED\Preprocessing"

process_audio(input_path, output_path)
print(" All audio files processed.")

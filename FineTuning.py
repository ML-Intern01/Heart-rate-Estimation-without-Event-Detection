# -*- coding: utf-8 -*-
"""
Created on Thu Jun 19 12:05:02 2025

@author: AIML
"""


# import numpy as np
# import pandas as pd
# import librosa
# import tensorflow as tf
# import matplotlib.pyplot as plt
# from scipy.signal import butter, sosfilt
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import Precision, Recall, F1Score


# def butter_bandpass_filter(data, lowcut=20.0, highcut=450.0, fs=1000.0, order=4):
#     sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
#     return sosfilt(sos, data)


# def preprocess(wav_path, save_path, sr=1000, n_mels=128):
#     y, orig_sr = librosa.load(wav_path, sr=None)
#     y = butter_bandpass_filter(y, fs=orig_sr)
#     y = librosa.resample(y, orig_sr, sr)
    
#     y = (y - np.mean(y)) / (np.std(y) + 1e-9)  # z-normalization
    
    
#     mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
#     mel_db = librosa.power_to_db(mel, ref=np.max)
#     np.save(save_path, mel_db.T[:2500])  # shape: (2500, 128)


# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, csv_path, input_shape=(2500, 128), batch_size=2, shuffle=True):
#         self.df = pd.read_csv(csv_path)
#         self.input_shape = input_shape
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.indices = np.arange(len(self.df))
#         self.on_epoch_end()

#     def __len__(self):
#         return int(np.ceil(len(self.df) / self.batch_size))

#     def __getitem__(self, index):
#         batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
#         X = np.empty((len(batch_indices), *self.input_shape))
#         y = []

#         for i, idx in enumerate(batch_indices):
#             row = self.df.iloc[idx]
#             x = np.load(row['filepath'])  # .npy file
#             label = np.array(eval(row['label']))
#             X[i,] = x
#             y.append(label)

#         return X, np.array(y)

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)
    

# # BiLSTM MODEL 
# def BiLSTM(input_shape, num_classes):
#     inputs = Input(shape=input_shape)
#     x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
#     x = BatchNormalization()(x)
#     x = Bidirectional(LSTM(128, return_sequences=True))(x)
#     x = GlobalAveragePooling1D()(x)
#     x = BatchNormalization()(x)
#     x = Dense(32, activation='relu')(x)
#     x = Dropout(0.3)(x)
#     outputs = Dense(num_classes, activation='softmax')(x)
#     return Model(inputs, outputs)
# # Load Pretrained Model

# model = load_model('model.h5', custom_objects={'F1Score': F1Score})

# # Compile Model

# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss='categorical_crossentropy',
#     metrics=[
#         'accuracy',
#         Precision(name='precision'),
#         Recall(name='recall'),
#         F1Score(num_classes=25, average='macro', name='f1_score')
#     ]
# )


# # Load CSV and Train

# train_gen = DataGenerator('train_verified.csv', batch_size=2)
# val_gen = DataGenerator('val_verified.csv', batch_size=2)



# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=10,
#     callbacks=[ ],
#     verbose=1
# )


# def plot_metrics(history):
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Val Accuracy')
#     plt.title("Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.title("Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.tight_layout()
#     plt.show()

# plot_metrics(history)


# # Evaluation on Test Set

# test_csv = 'test_verified.csv'  # you will provide this path
# test_gen = DataGenerator(test_csv, batch_size=2, shuffle=False)

# print("\n Evaluating on Test Data-")
# results = model.evaluate(test_gen)

# print("\nTest Metrics:")
# for name, value in zip(model.metrics_names, results):
#     print(f"{name}: {value:.4f}")


# # Save Fine-Tuned Model

# model.save('bilstm_finetuned_verified_5sec.h5')
# print(" Model saved.")





# =============================================================================
# import os
# import numpy as np
# import pandas as pd
# import librosa
# import matplotlib.pyplot as plt
# from scipy.signal import butter, sosfilt
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import Precision, Recall, F1Score
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.preprocessing import StandardScaler
# 
# output_dir = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\Feature Extraction(hop size 2ms)\Fine-Tuning\Output"
# os.makedirs(output_dir, exist_ok=True)
# 
# 
# # Preprocessing 
# 
# def butter_bandpass_filter(data, lowcut=20.0, highcut=450.0, fs=1000.0, order=4):
#     sos = butter(order, [lowcut, highcut], btype='bandpass', fs=fs, output='sos')
#     return sosfilt(sos, data)
# 
# def preprocess_and_extract_mel(wav_path, save_path, sr=1000, n_mels=128):
#     y, orig_sr = librosa.load(wav_path, sr=None)
#     y = butter_bandpass_filter(y, fs=orig_sr)
#     y = librosa.resample(y, orig_sr, sr)
# 
# def z_normalize(y):
#     scaler = StandardScaler()
#     y = y.reshape(-1, 1)
#     return scaler.fit_transform(y)
# 
# 
# #  Feature Extraction
# def feature_extraction(y, sr, save_path, n_mels=128, max_frames=2500):
#     mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
#     mel_db = librosa.power_to_db(mel, ref=np.max)
#     mel_db = mel_db.T[:max_frames]  # shape: (max_frames, n_mels)
#     np.save(save_path, mel_db)
# 
# 
# # Data Generator
# 
# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, csv_path, input_shape=(2500, 128), batch_size=2, shuffle=True):
#         self.df = pd.read_csv(csv_path)
#         self.input_shape = input_shape
#         self.batch_size = batch_size
#         self.shuffle = shuffle
#         self.indices = np.arange(len(self.df))
#         self.on_epoch_end()
# 
#     def __len__(self):
#         return int(np.ceil(len(self.df) / self.batch_size))
# 
#     def __getitem__(self, index):
#         batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
#         X = np.empty((len(batch_indices), *self.input_shape))
#         y = []
# 
#         for i, idx in enumerate(batch_indices):
#             row = self.df.iloc[idx]
#             x = np.load(row['filepath'])
#             label = np.array(eval(row['label']))
#             X[i,] = x
#             y.append(label)
# 
#         return X, np.array(y)
# 
#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)
# 
# 
# # Load Pretrained Model
# 
# model = load_model('model.h5', custom_objects={'F1Score': F1Score})
# 
# 
# # Compile Model
# 
# model.compile(
#     optimizer=Adam(learning_rate=1e-4),
#     loss='categorical_crossentropy',
#     metrics=[
#         'accuracy',
#         Precision(name='precision'),
#         Recall(name='recall'),
#         F1Score(num_classes=25, average='macro', name='f1_score')
#     ]
# )
# 
# 
# # Load Data Generators
# 
# train_gen = DataGenerator('train_verified.csv', batch_size=2)
# val_gen = DataGenerator('val_verified.csv', batch_size=2, shuffle=False)
# 
# 
# # Train Model
# 
# early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=10,
#     callbacks=[early_stop],
#     verbose=1
# )
# 
# # Plot and Save Metrics
# 
# def plot_metrics(history, output_dir):
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train Accuracy')
#     plt.plot(history.history['val_accuracy'], label='Val Accuracy')
#     plt.title("Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend()
# 
#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train Loss')
#     plt.plot(history.history['val_loss'], label='Val Loss')
#     plt.title("Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.tight_layout()
# 
#     plot_path = os.path.join(output_dir, 'training_metrics.png')
#     plt.savefig(plot_path)
#     plt.close()
#     print(f" Plot saved to: {plot_path}")
# 
# plot_metrics(history, output_dir)
# 
# 
# # Evaluate on Test Set
# 
# test_gen = DataGenerator('test_verified.csv', batch_size=2, shuffle=False)
# results = model.evaluate(test_gen)
# 
# # Save Evaluation Metrics to CSV
# metrics_dict = dict(zip(model.metrics_names, results))
# df_metrics = pd.DataFrame([metrics_dict])
# metrics_path = os.path.join(output_dir, "evaluation_metrics.csv")
# df_metrics.to_csv(metrics_path, index=False)
# print(f"Evaluation metrics saved to: {metrics_path}")
# 
# 
# # Save Fine-Tuned Model
# 
# model_path = os.path.join(output_dir, 'bilstm_finetuned_verified_5sec.h5')
# model.save(model_path)
# print(f" Model saved to: {model_path}")
# =============================================================================








# import os
# import numpy as np
# import pandas as pd
# import librosa
# from scipy import signal
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# import tensorflow as tf
# from tensorflow.keras.models import load_model
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.metrics import Precision, Recall
# from tensorflow.keras import backend as K
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix


# Input_path = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\Feature Extraction(hop size 2ms)\Fine-Tuning\Excel file\train_file.csv"
# Output_DIR = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\Feature Extraction(hop size 2ms)\Fine-Tuning\output 1"
# Pretrained_model = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\Feature Extraction(hop size 2ms)\Fine-Tuning\model\best_model_epoch_25.h5"
# Test_csv = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\Feature Extraction(hop size 2ms)\Fine-Tuning\Excel file\test_file.csv" 

# Sr = 1000
# N_Mels = 128
# hop_len = 2
# win_len = 50
# N_fft = Sr // 2
# Batch_size = 2

# os.makedirs(Output_DIR, exist_ok=True)
# NPY_DIR = os.path.join(Output_DIR, "npy")
# os.makedirs(NPY_DIR, exist_ok=True)

# class F1Score(tf.keras.metrics.Metric):
#     def __init__(self, name='f1_score', **kwargs):
#         super().__init__(name=name, **kwargs)
#         self.precision = Precision()
#         self.recall = Recall()

#     def update_state(self, y_true, y_pred, sample_weight=None):
#         self.precision.update_state(y_true, y_pred, sample_weight)
#         self.recall.update_state(y_true, y_pred, sample_weight)

#     def result(self):
#         p = self.precision.result()
#         r = self.recall.result()
#         return 2 * ((p * r) / (p + r + K.epsilon()))

#     def reset_state(self):
#         self.precision.reset_state()
#         self.recall.reset_state()

# # Preprocessing & Feature Extraction
# def butter_bandpass_filter(data, lowcut=20, highcut=450, fs=1000, order=4):
#     sos = signal.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
#     return signal.sosfiltfilt(sos, data)

# def resample_audio(data, original_sr, target_sr=1000):
#     if original_sr != target_sr:
#         return librosa.resample(y=data, orig_sr=original_sr, target_sr=target_sr)
#     return data

# def z_normalize(data):
#     scaler = StandardScaler()
#     return scaler.fit_transform(data.reshape(-1, 1)).flatten

# def extract_mel_spec(y, sr):
#     mel_spec = librosa.feature.melspectrogram(
#         y=y, sr=sr, n_fft=N_fft, hop_length=hop_len,
#         win_length=win_len, n_mels=N_Mels, fmin=20, fmax=450, power=2.0
#     )
#     mel_db = librosa.power_to_db(mel_spec, ref=np.max)
#     return mel_db.T[:2500]

# def preprocess_and_extract(df, split_name):
#     processed = []
#     for i, row in df.iterrows():
#         wav_path = row['file_path']
#         label = row['labels']
#         try:
#             y, sr = librosa.load(wav_path, sr=None)
#             y = butter_bandpass_filter(y, fs=sr)
#             y = librosa.resample(y, orig_sr=sr, target_sr=Sr)
#             y = z_normalize(y)
#             mel = extract_mel_spec(y, Sr)
#             npy_path = os.path.join(NPY_DIR, f"{split_name}_{i}.npy")
#             np.save(npy_path, mel)
#             processed.append({'file_path': npy_path, 'label': int(label)})
#         except Exception as e:
#             print(f"[ERROR] {wav_path}: {e}")
#     return pd.DataFrame(processed)

# df = pd.read_csv(Input_path)
# df['label_int'] = pd.Categorical(df['labels']).codes
# train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_int'], random_state=42)

# print(" Preprocessing and extracting training data ")
# train_processed = preprocess_and_extract(train_df, "train")
# train_csv = os.path.join(Output_DIR, "train.csv")
# train_processed.to_csv(train_csv, index=False)

# print(" Preprocessing and extracting validation data ")
# val_processed = preprocess_and_extract(val_df, "val")
# val_csv = os.path.join(Output_DIR, "val.csv")
# val_processed.to_csv(val_csv, index=False)

# # Data Generator 

# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, csv_path, batch_size=2, input_shape=(2500, 128), shuffle=True):
#         self.df = pd.read_csv(csv_path)
#         self.batch_size = batch_size
#         self.input_shape = input_shape
#         self.shuffle = shuffle
#         self.indices = np.arange(len(self.df))
#         self.on_epoch_end()

#     def __len__(self):
#         return int(np.ceil(len(self.df) / self.batch_size))

#     def __getitem__(self, index):
#         idxs = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
#         X = np.zeros((len(idxs), *self.input_shape))
#         y = []
#         for i, idx in enumerate(idxs):
#             row = self.df.iloc[idx]
#             X[i] = np.load(row['file_path'])  
#             label = int(row['label'])
#             y.append(label)
#         return X, tf.keras.utils.to_categorical(y, num_classes=25)

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)



# # Load & Compile Pretrained Model 
# model = load_model(Pretrained_model, custom_objects={'F1Score': F1Score})
# model.compile(
#     optimizer=Adam(1e-4),
#     loss='categorical_crossentropy',
#     metrics=[
#         'accuracy',
#         Precision(name='precision'),
#         Recall(name='recall'),
#         F1Score(name='f1_score')
#     ]
# )


# # Fine-Tuning 
# train_gen = DataGenerator(train_csv, batch_size=Batch_size)
# val_gen = DataGenerator(val_csv, batch_size=Batch_size, shuffle=False)

# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=10,
#     callbacks=[],
#     verbose=1
# )

# # Plotting 
# def plot_metrics(history, save_path):
#     plt.figure(figsize=(10, 4))
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train', marker='o')
#     plt.plot(history.history['val_accuracy'], label='Val')
#     plt.title("Accuracy")
#     plt.legend()

#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Train', marker='s')
#     plt.plot(history.history['val_loss'], label='Val')
#     plt.title("Loss")
#     plt.legend()

#     plt.tight_layout()
#     path = os.path.join(save_path, "training_plots.png")
#     plt.savefig(path)
#     plt.close()
#     print(f" Training plot saved: {path}")

# plot_metrics(history, Output_DIR)


# # Evaluation on the test data

# test_df = pd.read_csv(Test_csv)
# test_df['label_int'] = pd.Categorical(test_df['labels']).codes

# # Extract features and save .npy files
# test_processed = preprocess_and_extract(test_df, "test")
# test_csv_verified = os.path.join(Output_DIR, "test.csv")
# test_processed.to_csv(test_csv_verified, index=False)

# # Load .npy features and labels
# X_test, y_test = [], []
# for _, row in test_processed.iterrows():
#     try:
#         x = np.load(row['filepath'])
#         y = np.array(eval(row['label']))  
#         X_test.append(x)
#         y_test.append(y)
#     except Exception as e:
#         print(f"[ERROR] Loading failed: {row['filepath']} - {e}")

# X_test = np.array(X_test)
# y_test = np.array(y_test)
# y_true = np.argmax(y_test, axis=1)

# # Prediction
# y_pred = model.predict(X_test, batch_size=32)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # Confusion Matrix
# cm = confusion_matrix(y_true, y_pred_classes)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
# plt.title("Confusion Matrix - Test Data")
# plt.xlabel("Predicted")
# plt.ylabel("True")
# cm_test_path = os.path.join(Output_DIR, "confusion_matrix_test.png")
# plt.savefig(cm_test_path)
# plt.close()
# print(f" Test confusion matrix saved: {cm_test_path}")

# #  Evaluation Metrics
# loss, acc, precision, recall, f1 = model.evaluate(X_test, y_test, batch_size=32)
# metrics = {
#     "loss": loss,
#     "accuracy": acc,
#     "precision": precision,
#     "recall": recall,
#     "f1_score": f1
# }
# metrics_path = os.path.join(Output_DIR, "evaluation_metrics_test.csv")
# pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
# print(f" Test evaluation metrics saved: {metrics_path}")






















import os
import numpy as np
import pandas as pd
import librosa
from scipy import signal
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report


Input_path = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\Feature Extraction(hop size 2ms)\Fine-Tuning\Excel file\train_file.csv"
Output_DIR = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\Feature Extraction(hop size 2ms)\Fine-Tuning\Output(best of pretrained model)"
Pretrained_model = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\Feature Extraction(hop size 2ms)\data generator(Architecture)\Avg pooling 3\models\best_model_epoch_25.h5"
Test_csv = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\Feature Extraction(hop size 2ms)\Fine-Tuning\Excel(main)\test_file.csv"

Sr = 1000
N_Mels = 128
hop_len = 2
win_len = 50
N_fft = Sr // 2
Batch_size = 2


os.makedirs(Output_DIR, exist_ok=True)
NPY_DIR = os.path.join(Output_DIR, "npy")
os.makedirs(NPY_DIR, exist_ok=True)

class F1Score(tf.keras.metrics.Metric):
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.precision = Precision()
        self.recall = Recall()

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision.update_state(y_true, y_pred, sample_weight)
        self.recall.update_state(y_true, y_pred, sample_weight)

    def result(self):
        p = self.precision.result()
        r = self.recall.result()
        return 2 * ((p * r) / (p + r + K.epsilon()))

    def reset_state(self):
        self.precision.reset_state()
        self.recall.reset_state()

# Preprocessing & Feature Extraction
def butter_bandpass_filter(data, lowcut=20, highcut=450, fs=1000, order=4):
    sos = signal.butter(order, [lowcut, highcut], btype='band', fs=fs, output='sos')
    return signal.sosfiltfilt(sos, data)

def resample_audio(data, original_sr, target_sr=1000):
    if original_sr != target_sr:
        return librosa.resample(y=data, orig_sr=original_sr, target_sr=target_sr)
    return data

def z_normalize(data):
    scaler = StandardScaler()
    return scaler.fit_transform(data.reshape(-1, 1)).flatten()  

def extract_mel_spec(y, sr):
    mel_spec = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_fft, hop_length=hop_len,
        win_length=win_len, n_mels=N_Mels, fmin=20, fmax=450, power=2.0
    )
    mel_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_db.T[:2500]

def preprocess_and_extract(df, split_name):
    processed = []
    for i, row in df.iterrows():
        wav_path = row['file_path']
        label = row['labels']
        try:
            y, sr = librosa.load(wav_path, sr=None)
            y = butter_bandpass_filter(y, fs=sr)
            y = librosa.resample(y, orig_sr=sr, target_sr=Sr)
            y = z_normalize(y)
            mel = extract_mel_spec(y, Sr)
            npy_path = os.path.join(NPY_DIR, f"{split_name}_{i}.npy")
            np.save(npy_path, mel)
            processed.append({'file_path': npy_path, 'label': int(label)})
        except Exception as e:
            print(f"[ERROR] {wav_path}: {e}")
    return pd.DataFrame(processed)

# # Data loading and splitting
# df = pd.read_csv(Input_path)
# df['label_int'] = pd.Categorical(df['labels']).codes
# train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label_int'], random_state=42)

# print("Preprocessing and extracting training data...")
# train_processed = preprocess_and_extract(train_df, "train")
# train_csv = os.path.join(Output_DIR, "train.csv")
# train_processed.to_csv(train_csv, index=False)

# print("Preprocessing and extracting validation data...")
# val_processed = preprocess_and_extract(val_df, "val")
# val_csv = os.path.join(Output_DIR, "val.csv")
# val_processed.to_csv(val_csv, index=False)

# # Data Generator 
# class DataGenerator(tf.keras.utils.Sequence):
#     def __init__(self, csv_path, batch_size=2, input_shape=(2500, 128), shuffle=True):
#         self.df = pd.read_csv(csv_path)
#         self.batch_size = batch_size
#         self.input_shape = input_shape
#         self.shuffle = shuffle
#         self.indices = np.arange(len(self.df))
#         self.on_epoch_end()

#     def __len__(self):
#         return int(np.ceil(len(self.df) / self.batch_size))

#     def __getitem__(self, index):
#         idxs = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
#         X = np.zeros((len(idxs), *self.input_shape))
#         y = []
#         for i, idx in enumerate(idxs):
#             row = self.df.iloc[idx]
#             X[i] = np.load(row['file_path'])  
#             label = int(row['label'])
#             y.append(label)
#         return X, tf.keras.utils.to_categorical(y, num_classes=25)

#     def on_epoch_end(self):
#         if self.shuffle:
#             np.random.shuffle(self.indices)

# Load & Compile Pretrained Model 
print("Loading pretrained model...")
model = load_model(Pretrained_model, custom_objects={'F1Score': F1Score})
model.compile(
    optimizer=Adam(learning_rate=1e-4),  
    loss='categorical_crossentropy',
    metrics=[
        'accuracy',
        Precision(name='precision'),
        Recall(name='recall'),
        F1Score(name='f1_score')
    ]
)

# # Callbacks for fine-tuning
# callbacks = [
#     ModelCheckpoint(
#         filepath=os.path.join(Output_DIR, 'best_finetuned_model.h5'),
#         monitor='val_accuracy',
#         save_best_only=True,
#         mode='max',
#         verbose=1
#     )
# ]

# # Fine-Tuning 

# train_gen = DataGenerator(train_csv, batch_size=Batch_size)
# val_gen = DataGenerator(val_csv, batch_size=Batch_size, shuffle=False)

# history = model.fit(
#     train_gen,
#     validation_data=val_gen,
#     epochs=100,
#     callbacks=callbacks,
#     verbose=1
# )

# # Plotting 
# def plot_metrics(history, save_path):
#     plt.figure(figsize=(15, 10))
    
#     # Plot accuracy
#     plt.subplot(2, 2, 1)
#     plt.plot(history.history['accuracy'], label='Train', marker='o')
#     plt.plot(history.history['val_accuracy'], label='Val', marker='s')
#     plt.title("Accuracy")
#     plt.xlabel("Epoch")
#     plt.ylabel("Accuracy")
#     plt.legend()
#     plt.grid(True)

#     # Plot loss
#     plt.subplot(2, 2, 2)
#     plt.plot(history.history['loss'], label='Train', marker='o')
#     plt.plot(history.history['val_loss'], label='Val', marker='s')
#     plt.title("Loss")
#     plt.xlabel("Epoch")
#     plt.ylabel("Loss")
#     plt.legend()
#     plt.grid(True)

#     plt.tight_layout()
#     path = os.path.join(save_path, "training_plots.png")
#     plt.savefig(path, dpi=300, bbox_inches='tight')
#     plt.close()
#     print(f"Training plot saved: {path}")

# plot_metrics(history, Output_DIR)

# # Save training history
# history_df = pd.DataFrame(history.history)
# history_path = os.path.join(Output_DIR, "training_history.csv")
# history_df.to_csv(history_path, index=False)
# print(f"Training history saved: {history_path}")


# Evaluation on the test data
print("Processing test data.")
test_df = pd.read_csv(Test_csv)
test_df['label_int'] = pd.Categorical(test_df['labels']).codes

# Extract features and save .npy files
test_processed = preprocess_and_extract(test_df, "test")
test_csv_verified = os.path.join(Output_DIR, "test.csv")
test_processed.to_csv(test_csv_verified, index=False)

# Load .npy features and labels
print("Loading test features.")
X_test, y_test = [], []
for _, row in test_processed.iterrows():
    try:
        x = np.load(row['file_path'])  
        label = int(row['label'])
        X_test.append(x)
        y_test.append(label)
    except Exception as e:
        print(f"[ERROR] Loading failed: {row['file_path']} - {e}")

if len(X_test) == 0:
    print("[ERROR] No test data loaded successfully!")
    exit()

X_test = np.array(X_test)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=25)  # Fixed: Convert to categorical
y_true = np.argmax(y_test, axis=1)

# Prediction
print("Making predictions on test data...")
y_pred = model.predict(X_test, batch_size=32, verbose=1)
y_pred_classes = np.argmax(y_pred, axis=1)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred_classes)
plt.figure(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar_kws={'label': 'Count'})
plt.title("Confusion Matrix - Test Data", fontsize=16)
plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("True Label", fontsize=12)
cm_test_path = os.path.join(Output_DIR, "confusion_matrix_test.png")
plt.savefig(cm_test_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Test confusion matrix saved: {cm_test_path}")

# Classification Report
class_report = classification_report(y_true, y_pred_classes, output_dict=True)
class_report_df = pd.DataFrame(class_report).transpose()
class_report_path = os.path.join(Output_DIR, "classification_report_test.csv")
class_report_df.to_csv(class_report_path)
print(f"Classification report saved: {class_report_path}")

# Evaluation Metrics
print("Evaluating model on test data...")
test_loss, test_acc, test_precision, test_recall, test_f1 = model.evaluate(X_test, y_test, batch_size=32, verbose=1)

metrics = {
    "test_accuracy": test_acc,
    "test_precision": test_precision,
    "test_recall": test_recall,
    "test_f1_score": test_f1
}

metrics_path = os.path.join(Output_DIR, "evaluation_metrics_test.csv")
pd.DataFrame([metrics]).to_csv(metrics_path, index=False)
print(f"Test evaluation metrics saved: {metrics_path}")



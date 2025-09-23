# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 17:49:13 2025

@author: AIML
"""

import os
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Input, Bidirectional, LSTM, BatchNormalization, GlobalAveragePooling1D, Dropout, Dense,GlobalMaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam



output_dir = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\Feature Extraction(hop size 2ms)\data generator\Avg pooling"
model_dir = os.path.join(output_dir, "models")
plot_dir = os.path.join(output_dir, "plots")

for d in [model_dir, plot_dir]:
    os.makedirs(d, exist_ok=True)


# Load CSV
csv_file = r"D:\Heart rate estimation (updated)\BiLSTM(70,30 split)\5 sec duration\Feature Extraction(hop size 2ms)\Excel\train_file_labels.csv"
df = pd.read_csv(csv_file)

if 'labels' in df.columns:
    df['class_index'] = df['labels'].astype(int)
elif 'class_index' not in df.columns:
    raise ValueError("No 'labels' or 'class_index' column found!")

# Train-validation split
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['class_index'], random_state=42)

# Upsample training data
max_count = train_df['class_index'].value_counts().max()
upsampled_dfs = []
for class_idx, group in train_df.groupby('class_index'):
    if len(group) < max_count:
        group = resample(group, replace=True, n_samples=max_count, random_state=42)
    upsampled_dfs.append(group)
train_df_upsampled = pd.concat(upsampled_dfs).sample(frac=1, random_state=42).reset_index(drop=True)

print("Class distribution in upsampled training set:")
print(train_df_upsampled['class_index'].value_counts().sort_index())

print("\nClass distribution in validation set:")
print(val_df['class_index'].value_counts().sort_index())

train_output = os.path.join(output_dir, "train_split_upsampled.csv")
val_output = os.path.join(output_dir, "val_split.csv")
train_df_upsampled.to_csv(train_output, index=False)
val_df.to_csv(val_output, index=False)


class DataGenerator(Sequence):
    def __init__(self, csv_path, samples_per_class=4, batch_size=None, input_shape=(2500, 128), shuffle=True, is_training=True):
        self.csv_path = csv_path
        self.samples_per_class = samples_per_class
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.shuffle = shuffle
        self.is_training = is_training

        # Load CSV 
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['file_path'].apply(os.path.exists)].reset_index(drop=True)

        if len(self.df) == 0:
            raise ValueError(f"No valid files found in {csv_path}")

        self.class_indices = sorted(self.df['class_index'].unique())
        self.num_classes = len(self.class_indices)

        if self.is_training:
            self.batch_size = self.num_classes * self.samples_per_class

            self.class_to_indices = {
                class_idx: self.df[self.df['class_index'] == class_idx].index.tolist()
                for class_idx in self.class_indices
            }

            # Check balanced classes
            class_sizes = [len(indices) for indices in self.class_to_indices.values()]
            if len(set(class_sizes)) != 1:
                raise ValueError("All classes must have the same number of samples after upsampling.")

            class_size = class_sizes[0]
            self.batches_per_epoch = class_size // self.samples_per_class
        else:
            if self.batch_size is None:
                self.batch_size = 64
            self.batches_per_epoch = len(self.df) // self.batch_size

        self.on_epoch_end()

    def __len__(self):
        return self.batches_per_epoch

    def __getitem__(self, index):
        if self.is_training:
            batch_indices = []
            for class_idx in self.class_indices:
                class_indices = self.class_to_indices[class_idx]
                start_idx = (index * self.samples_per_class) % len(class_indices)
                selected_indices = []
                for i in range(self.samples_per_class):
                    idx = (start_idx + i) % len(class_indices)
                    selected_indices.append(class_indices[idx])
                batch_indices.extend(selected_indices)
            return self._get_batch_data(batch_indices)
        else:
            start = index * self.batch_size
            end = start + self.batch_size
            batch_df = self.df.iloc[start:end]
            return self._get_batch_data_from_df(batch_df)

    def _get_batch_data(self, batch_indices):
        X_batch = []
        y_batch = []

        for idx in batch_indices:
            row = self.df.iloc[idx]
            file_path = row['file_path']
            label = row['class_index']

            try:
                data = np.load(file_path)
                if data.shape == self.input_shape:
                    X_batch.append(data)
                    y_batch.append(label)
                else:
                    print(f"Warning: Shape mismatch for {file_path}: expected {self.input_shape}, got {data.shape}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if len(X_batch) == 0:
            print("Warning: No valid samples in batch, creating dummy batch")
            X_batch = [np.zeros(self.input_shape)]
            y_batch = [0]

        X = np.array(X_batch)
        y = tf.keras.utils.to_categorical(y_batch, num_classes=self.num_classes)

        return X, y

    def _get_batch_data_from_df(self, batch_df):
        X_batch = []
        y_batch = []

        for _, row in batch_df.iterrows():
            file_path = row['file_path']
            label = row['class_index']
            try:
                data = np.load(file_path)
                if data.shape == self.input_shape:
                    X_batch.append(data)
                    y_batch.append(label)
                else:
                    print(f"Warning: Shape mismatch for {file_path}: expected {self.input_shape}, got {data.shape}")
            except Exception as e:
                print(f"Error loading {file_path}: {e}")

        if len(X_batch) == 0:
            print("Warning: No valid samples in batch, creating dummy batch")
            X_batch = [np.zeros(self.input_shape)]
            y_batch = [0]

        X = np.array(X_batch)
        y = tf.keras.utils.to_categorical(y_batch, num_classes=self.num_classes)

        return X, y

    def on_epoch_end(self):
        if self.shuffle and self.is_training:
            for class_indices in self.class_to_indices.values():
                random.shuffle(class_indices)


# BiLSTM MODEL 
def BiLSTM(input_shape, num_classes):
    inputs = Input(shape=input_shape)
    x = Bidirectional(LSTM(128, return_sequences=True))(inputs)
    x = BatchNormalization()(x)
    x = Bidirectional(LSTM(128, return_sequences=True))(x)
    x = GlobalAveragePooling1D()(x)
    x = BatchNormalization()(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.3)(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs, outputs)


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


class BatchEpochLogger(tf.keras.callbacks.Callback):
    def __init__(self, model_dir, plot_dir):
        super().__init__()
        self.model_dir = model_dir
        self.plot_dir = plot_dir
        self.batch_losses = []
        self.batch_accuracies = []
        self.epoch_metrics = []
        self.best_val_loss = float('inf')
        self.best_epoch = -1

    def on_train_begin(self, logs=None):
        self.batch_losses = []
        self.batch_accuracies = []
        self.epoch_metrics = []

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        self.batch_losses.append(logs.get('loss'))
        self.batch_accuracies.append(logs.get('accuracy'))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        mean_batch_loss = np.mean(self.batch_losses) if self.batch_losses else None
        mean_batch_acc = np.mean(self.batch_accuracies) if self.batch_accuracies else None

        epoch_data = {
            'epoch': epoch + 1,
            'mean_batch_loss': mean_batch_loss,
            'mean_batch_accuracy': mean_batch_acc,
            'val_loss': logs.get('val_loss'),
            'val_accuracy': logs.get('val_accuracy'),
            'val_precision': logs.get('val_precision'),
            'val_recall': logs.get('val_recall'),
            'val_f1_score': logs.get('val_f1_score')
        }
        self.epoch_metrics.append(epoch_data)

        # Reset batch metrics
        self.batch_losses = []
        self.batch_accuracies = []

        self._save_plots()

        # Save every epoch model
        epoch_model_path = os.path.join(self.model_dir, f"model_epoch_{epoch + 1:02d}.h5")
        self.model.save(epoch_model_path)
        
        # Save best model
        if logs.get('val_loss') is not None and logs['val_loss'] < self.best_val_loss:
            self.best_val_loss = logs['val_loss']
            self.best_epoch = epoch + 1
            best_model_path = os.path.join(self.model_dir, f"best_model_epoch_{self.best_epoch}.h5")
            self.model.save(best_model_path)

        # Save metrics
        metrics_df = pd.DataFrame(self.epoch_metrics)
        metrics_df.to_csv(os.path.join(self.model_dir, "epoch_metrics.csv"), index=False)

    def _save_plots(self):
        if len(self.epoch_metrics) < 1:
            return
            
        epochs = [m['epoch'] for m in self.epoch_metrics]
        train_loss = [m['mean_batch_loss'] for m in self.epoch_metrics]
        val_loss = [m['val_loss'] for m in self.epoch_metrics]
        train_acc = [m['mean_batch_accuracy'] for m in self.epoch_metrics]
        val_acc = [m['val_accuracy'] for m in self.epoch_metrics]

        plt.figure(figsize=(15, 5))

        # Accuracy plot
        plt.subplot(1, 3, 2)
        if None not in train_acc:
            plt.plot(epochs, train_acc, label='Train Accuracy', marker='o')
        if None not in val_acc:
            plt.plot(epochs, val_acc, label='Val Accuracy', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy per Epoch')
        plt.legend()
        plt.grid(True)
        
        # Loss plot
        plt.subplot(1, 3, 1)
        if None not in train_loss:
            plt.plot(epochs, train_loss, label='Train Loss', marker='o')
        if None not in val_loss:
            plt.plot(epochs, val_loss, label='Val Loss', marker='s')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss per Epoch')
        plt.legend()
        plt.grid(True)

        latest_epoch = epochs[-1]
        plot_path = os.path.join(self.plot_dir, f"plot_{latest_epoch:02d}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()



train_gen = DataGenerator(csv_path=train_output, samples_per_class=4, input_shape=(2500, 128), shuffle=True)
val_gen = DataGenerator(csv_path=val_output, batch_size=64, input_shape=(2500, 128), shuffle=False, is_training=False)

print(f"Number of classes: {train_gen.num_classes}")
print(f"Training generator: {len(train_gen)} batches per epoch")
print(f"Validation generator: {len(val_gen)} batches per epoch")

optimizer = Adam(learning_rate=1e-4)

model = BiLSTM(input_shape=(2500, 128), num_classes=train_gen.num_classes)
model.compile(
    optimizer=optimizer,
    loss='categorical_crossentropy',
    metrics=['accuracy', Precision(), Recall(), F1Score()]
)
print(model.summary())

batch_epoch_logger = BatchEpochLogger(model_dir=model_dir, plot_dir=plot_dir)
callbacks = [
    batch_epoch_logger,
    tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(model_dir, 'checkpoint_epoch_{epoch:02d}.h5'),
        save_freq='epoch',
        verbose=1
    )
]

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=50,
    callbacks=callbacks,
    verbose=1
)
    
# Save final model
final_model_path = os.path.join(model_dir, "bilstm_model_final.h5")
model.save(final_model_path)
print(f"Final model saved to: {final_model_path}")
    
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
from imblearn.over_sampling import SMOTE
from scipy import ndimage, fft


class LightFluxProcessor:
    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True):
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize

    def fourier_transform(self, X):
        return np.abs(fft.fft(X, n=X.size))

    def process(self, df_train_x, df_dev_x):
        # Fourier Transform
        if self.fourier:
            print("Applying Fourier Transform...")
            df_train_x = df_train_x.apply(self.fourier_transform, axis=1)
            df_dev_x = df_dev_x.apply(self.fourier_transform, axis=1)
            df_train_x = np.stack(df_train_x)
            df_dev_x = np.stack(df_dev_x)

        # Normalize
        if self.normalize:
            print("Normalizing...")
            df_train_x = normalize(df_train_x)
            df_dev_x = normalize(df_dev_x)

        # Gaussian Filter
        if self.gaussian:
            print("Applying Gaussian Filter...")
            df_train_x = ndimage.gaussian_filter(df_train_x, sigma=10)
            df_dev_x = ndimage.gaussian_filter(df_dev_x, sigma=10)

        # Standardize
        if self.standardize:
            print("Standardizing...")
            scaler = StandardScaler()
            df_train_x = scaler.fit_transform(df_train_x)
            df_dev_x = scaler.transform(df_dev_x)

        print("Finished Processing!")
        return df_train_x, df_dev_x


class NeuralNetworkDetector:
    def __init__(self, train_dataset_path, eval_dataset_path, layers, dropouts, epochs, batch_size, learning_rate):
        self.train_dataset_path = train_dataset_path
        self.eval_dataset_path = eval_dataset_path
        self.layers = layers
        self.dropouts = dropouts
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

    def load_data(self):
        print(f"Loading training dataset from {self.train_dataset_path}...")
        df_train = pd.read_csv(self.train_dataset_path)
        print(f"Loading evaluation dataset from {self.eval_dataset_path}...")
        df_eval = pd.read_csv(self.eval_dataset_path)

        # Separate features and labels
        df_train_x = df_train.drop("LABEL", axis=1)
        df_train_y = (df_train["LABEL"] == 2).astype(int)
        df_dev_x = df_eval.drop("LABEL", axis=1)
        df_dev_y = (df_eval["LABEL"] == 2).astype(int)

        return df_train_x, df_train_y, df_dev_x, df_dev_y

    def preprocess_data(self, df_train_x, df_train_y, df_dev_x, df_dev_y):
        print("Applying SMOTE to balance classes...")
        sm = SMOTE()
        df_train_x, df_train_y = sm.fit_resample(df_train_x, df_train_y)

        print("Preprocessing data with Fourier Transform, Normalization, etc...")
        processor = LightFluxProcessor()
        df_train_x, df_dev_x = processor.process(df_train_x, df_dev_x)

        return df_train_x, df_train_y, df_dev_x, df_dev_y

    def build_network(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Input(shape=input_shape))

        # Add layers and dropouts dynamically
        for i, units in enumerate(self.layers):
            model.add(tf.keras.layers.Dense(units, activation="relu"))
            if i < len(self.dropouts):
                model.add(tf.keras.layers.Dropout(self.dropouts[i]))

        # Output layer
        model.add(tf.keras.layers.Dense(1, activation="sigmoid"))
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        return model

    def train_and_evaluate(self):
        # Load and preprocess data
        df_train_x, df_train_y, df_dev_x, df_dev_y = self.load_data()
        df_train_x, df_train_y, df_dev_x, df_dev_y = self.preprocess_data(df_train_x, df_train_y, df_dev_x, df_dev_y)

        # Build and train the network
        model = self.build_network(input_shape=df_train_x.shape[1:])
        print("Training Neural Network...")
        history = model.fit(
            df_train_x,
            df_train_y,
            validation_data=(df_dev_x, df_dev_y),
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=1  # Shows progress for each epoch
        )
        
        # Plot accuracy and loss evolution
        self.plot_metrics(history)

        print("Evaluating the model...")
        self.evaluate_model(model, df_train_x, df_train_y, df_dev_x, df_dev_y)

        # Save the model weights
        model.save_weights("nn_model_weights.h5")
        print("Model weights saved to nn_model_weights.h5")

    def evaluate_model(self, model, X_train, Y_train, X_eval, Y_eval):
        # Predictions
        train_preds = (model.predict(X_train) > 0.5).astype(int)
        eval_preds = (model.predict(X_eval) > 0.5).astype(int)

        # Metrics
        self.report_metrics(Y_train, train_preds, "Train")
        self.report_metrics(Y_eval, eval_preds, "Eval")

    def report_metrics(self, y_true, y_pred, dataset_name):
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)

        print(f"Metrics for {dataset_name} Set:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Confusion Matrix:\n{conf_matrix}\n")

      

    def plot_metrics(self, history):
        """Plots accuracy and loss evolution"""
        
        # Plot accuracy
        if "accuracy" in history.history and "val_accuracy" in history.history:
            plt.figure()
            plt.plot(history.history["accuracy"], marker="o", markersize=4, label="Train Accuracy", linestyle='-', color='b')
            plt.plot(history.history["val_accuracy"], marker="x", markersize=6, label="Validation Accuracy", linestyle='--', color='g')
            plt.title("Model Accuracy Evolution")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.grid(True)
            plt.savefig("accuracy_evolution.png")
            plt.show()
        else:
            print("Warning: Accuracy metrics are missing from training history!")

        # Plot loss
        if "loss" in history.history and "val_loss" in history.history:
            plt.figure()
            plt.plot(history.history["loss"], marker="o", markersize=4, label="Train Loss", linestyle='-', color='b')
            plt.plot(history.history["val_loss"], marker="x", markersize=6, label="Validation Loss", linestyle='--', color='r')
            plt.title("Model Loss Evolution")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig("loss_evolution.png")
            plt.show()
        else:
            print("Warning: Loss metrics are missing from training history!")





import os
import yaml
import numpy as np 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.preprocessing import normalize, StandardScaler
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
        if self.fourier:
            print("Applying Fourier Transform...")
            df_train_x = df_train_x.apply(self.fourier_transform, axis=1)
            df_dev_x = df_dev_x.apply(self.fourier_transform, axis=1)

            # Rebuild datasets
            df_train_x = np.stack(df_train_x)
            df_dev_x = np.stack(df_dev_x)

        if self.normalize:
            print("Normalizing...")
            df_train_x = normalize(df_train_x)
            df_dev_x = normalize(df_dev_x)

        if self.gaussian:
            print("Applying Gaussian Filter...")
            df_train_x = ndimage.gaussian_filter(df_train_x, sigma=10)
            df_dev_x = ndimage.gaussian_filter(df_dev_x, sigma=10)

        if self.standardize:
            print("Standardizing...")
            scaler = StandardScaler()
            df_train_x = scaler.fit_transform(df_train_x)
            df_dev_x = scaler.transform(df_dev_x)

        print("Finished Preprocessing!")
        return df_train_x, df_dev_x


class SVMExoplanetDetector:
    def __init__(self, kernel="linear", dataset_path=""):
        self.kernel = kernel
        self.dataset_path = dataset_path

    def load_data(self):
        print(f"Loading dataset from {self.dataset_path}...")
        data = pd.read_csv(self.dataset_path)
        data = shuffle(data)
        X = data.drop(columns=["LABEL"])
        Y = (data["LABEL"] == 2).astype(int)  # Binary classification
        return train_test_split(X, Y, test_size=0.3, random_state=42)

    def train_and_evaluate(self):
        # Load and preprocess data
        X_train, X_dev, Y_train, Y_dev = self.load_data()
        print(f"Training SVM with kernel: {self.kernel}")

        # Preprocess the data
        processor = LightFluxProcessor()
        X_train, X_dev = processor.process(X_train, X_dev)

        # Train SVM
        model = SVC(kernel=self.kernel)
        model.fit(X_train, Y_train)

        # Evaluate the model
        Y_train_pred = model.predict(X_train)
        Y_dev_pred = model.predict(X_dev)

        self.report_metrics(Y_train, Y_train_pred, "Train")
        self.report_metrics(Y_dev, Y_dev_pred, "Dev")

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

        # Plot confusion matrix
        plt.figure()
        plt.matshow(conf_matrix, cmap="Blues")
        plt.colorbar()
        plt.title(f"{dataset_name} Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.savefig(f"{dataset_name.lower()}_conf_matrix.png")
        plt.show()

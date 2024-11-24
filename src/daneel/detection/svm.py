import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
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
        # Apply Fourier Transform
        if self.fourier:
            print("Applying Fourier Transform...")
            df_train_x = df_train_x.apply(self.fourier_transform, axis=1)
            df_dev_x = df_dev_x.apply(self.fourier_transform, axis=1)

            # Rebuild datasets
            df_train_x = np.stack(df_train_x)
            df_dev_x = np.stack(df_dev_x)

        # Normalize data
        if self.normalize:
            print("Normalizing...")
            df_train_x = normalize(df_train_x)
            df_dev_x = normalize(df_dev_x)

        # Apply Gaussian Filter
        if self.gaussian:
            print("Applying Gaussian Filter...")
            df_train_x = ndimage.gaussian_filter(df_train_x, sigma=10)
            df_dev_x = ndimage.gaussian_filter(df_dev_x, sigma=10)

        # Standardize data
        if self.standardize:
            print("Standardizing...")
            scaler = StandardScaler()
            df_train_x = scaler.fit_transform(df_train_x)
            df_dev_x = scaler.transform(df_dev_x)

        print("Finished Preprocessing!")
        return df_train_x, df_dev_x


class SVMExoplanetDetector:
    def __init__(self, kernel="linear", train_dataset_path="", eval_dataset_path=""):
        self.kernel = kernel
        self.train_dataset_path = train_dataset_path
        self.eval_dataset_path = eval_dataset_path

    def load_data(self):
        # Load the training and evaluation datasets from CSV
        print(f"Loading training dataset from {self.train_dataset_path}...")
        df_train = pd.read_csv(self.train_dataset_path)
        print(f"Loading evaluation dataset from {self.eval_dataset_path}...")
        df_dev = pd.read_csv(self.eval_dataset_path)

        # Separate features and labels for training
        df_train_x = df_train.drop('LABEL', axis=1)
        df_dev_x = df_dev.drop('LABEL', axis=1)
        df_train_y = df_train.LABEL
        df_dev_y = df_dev.LABEL

        # Shuffle data for randomness
        df_train = shuffle(df_train)
        df_dev = shuffle(df_dev)

        # Preprocess datasets using LightFluxProcessor
        processor = LightFluxProcessor(
            fourier=True,
            normalize=True,
            gaussian=True,
            standardize=True
        )
        df_train_x, df_dev_x = processor.process(df_train_x, df_dev_x)

        # Return the processed features and labels
        return df_train_x, df_train_y, df_dev_x, df_dev_y

    def train_and_evaluate(self):
        # Load and preprocess the data
        X_train, Y_train, X_dev, Y_dev = self.load_data()

        print(f"Training SVM with kernel: {self.kernel}")

        # Train the SVM model
        model = SVC(kernel=self.kernel, probability=True)
        model.fit(X_train, Y_train)

        # Predict and evaluate the model
        self.evaluate_model(model, X_train, Y_train, X_dev, Y_dev)

    def evaluate_model(self, model, X_train, Y_train, X_dev, Y_dev):
        # Make predictions for train and dev sets
        Y_train_pred = model.predict(X_train)
        Y_dev_pred = model.predict(X_dev)

        # Print metrics
        self.report_metrics(Y_train, Y_train_pred, "Train")
        self.report_metrics(Y_dev, Y_dev_pred, "Dev")

    def report_metrics(self, y_true, y_pred, dataset_name):
        # Calculate accuracy, precision, recall, and confusion matrix
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        conf_matrix = confusion_matrix(y_true, y_pred)

        # Print evaluation metrics
        print(f"Metrics for {dataset_name} Set:")
        print(f"Accuracy: {accuracy:.2f}")
        print(f"Precision: {precision:.2f}")
        print(f"Recall: {recall:.2f}")
        print(f"Confusion Matrix:\n{conf_matrix}\n")

        # Print detailed classification report
        #print(f"Classification Report - {dataset_name} Set:")
        #print(classification_report(y_true, y_pred))



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
        #sm = SMOTE()
        #df_train_x, df_train_y = sm.fit_resample(df_train_x, df_train_y)
        

        print("Preprocessing data with Fourier Transform, Normalization, etc...")
        processor = LightFluxProcessor()   #applying preprocess to the data already divided between data and labels
        df_train_x, df_dev_x= processor.process(df_train_x, df_dev_x)
        print(np.shape(df_train_x))

        return df_train_x, df_train_y, df_dev_x, df_dev_y  #the function just returns the values after the preprocessing


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
        df_train_x, df_train_y, df_dev_x, df_dev_y = self.load_data() #data loaded and divided

        
        df_train_x, df_train_y, df_dev_x, df_dev_y = self.preprocess_data(df_train_x, df_train_y, df_dev_x, df_dev_y) #divided data are preprocessed

        # Build and train the network
        model = self.build_network(input_shape=df_train_x.shape[1:])  #now building the model for the specific input of data, returns the model with all of its characteristics
        print("Training Neural Network...")
        history = model.fit(df_train_x,  df_train_y, epochs=self.epochs,  batch_size=self.batch_size) # Shows progress for each epoch
        
        
        # Plot accuracy and loss evolution
        self.plot_metrics(history)

        print("Evaluating the model...")
        self.evaluate_model(model, df_train_x, df_train_y, df_dev_x, df_dev_y)

        # Save the model weights
        #model.save_weights("nn_model_weights.h5")
        #print("Model weights saved to nn_model_weights.h5")

    def evaluate_model(self, model, X_train, Y_train, X_dev, Y_dev):
        # Predictions
        train_outputs = (model.predict(X_train, batch_size=self.batch_size) > 0.5).astype(int)
        dev_outputs = (model.predict(X_dev, batch_size=self.batch_size) > 0.5).astype(int)

        # Final Metrics
        accuracy_train = accuracy_score(Y_train, train_outputs)
        accuracy_dev = accuracy_score(Y_dev, dev_outputs)
        precision_train = precision_score(Y_train, train_outputs)
        precision_dev = precision_score(Y_dev, dev_outputs)
        recall_train = recall_score(Y_train, train_outputs)
        recall_dev = recall_score(Y_dev, dev_outputs)
        confusion_matrix_train = confusion_matrix(Y_train, train_outputs)
        confusion_matrix_dev = confusion_matrix(Y_dev, dev_outputs)

        print("Metrics:")
        print(f"Train Accuracy: {accuracy_train:.2f}, Dev Accuracy: {accuracy_dev:.2f}")
        print(f"Train Precision: {precision_train:.2f}, Dev Precision: {precision_dev:.2f}")
        print(f"Train Recall: {recall_train:.2f}, Dev Recall: {recall_dev:.2f}")
        print(f"Train Confusion Matrix:\n{confusion_matrix_train}")
        print(f"Dev Confusion Matrix:\n{confusion_matrix_dev}")
      

    def plot_metrics(self, history):
        """Plots accuracy and loss evolution"""
        
        # Plot accuracy
        if "accuracy" in history.history :
            plt.figure()
            plt.plot(history.history["accuracy"], marker="o", markersize=4, label="Train Accuracy", linestyle='-', color='b')
            
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
        if "loss" in history.history :
            plt.figure()
            plt.plot(history.history["loss"], marker="o", markersize=4, label="Train Loss", linestyle='-', color='b')
            
            plt.title("Model Loss Evolution")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.grid(True)
            plt.savefig("loss_evolution.png")
            plt.show()
        else:
            print("Warning: Loss metrics are missing from training history!")





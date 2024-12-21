import pandas as pd
import numpy as np
from scipy import ndimage, fft
from sklearn.preprocessing import normalize, StandardScaler
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

class LightFluxProcessor:

    def __init__(self, fourier=True, normalize=True, gaussian=True, standardize=True):
        self.fourier = fourier
        self.normalize = normalize
        self.gaussian = gaussian
        self.standardize = standardize

    def fourier_transform(self, X):
        return np.abs(fft.fft(X, n=X.size))

    def process(self, df_train_x, df_dev_x):
        # Apply fourier transform
        if self.fourier:
            print("Applying Fourier...")
            shape_train = df_train_x.shape
            shape_dev = df_dev_x.shape
            df_train_x = df_train_x.apply(self.fourier_transform, axis=1)
            df_dev_x = df_dev_x.apply(self.fourier_transform, axis=1)

            df_train_x_build = np.zeros(shape_train)
            df_dev_x_build = np.zeros(shape_dev)

            for ii, x in enumerate(df_train_x):
                df_train_x_build[ii] = x

            for ii, x in enumerate(df_dev_x):
                df_dev_x_build[ii] = x

            df_train_x = pd.DataFrame(df_train_x_build)
            df_dev_x = pd.DataFrame(df_dev_x_build)

            # Keep first half of data as it is symmetrical after previous steps
            df_train_x = df_train_x.iloc[:, : (df_train_x.shape[1] // 2)].values
            df_dev_x = df_dev_x.iloc[:, : (df_dev_x.shape[1] // 2)].values

        # Normalize
        if self.normalize:
            print("Normalizing...")
            df_train_x = pd.DataFrame(normalize(df_train_x))
            df_dev_x = pd.DataFrame(normalize(df_dev_x))

            # df_train_x = df_train_x.div(df_train_x.sum(axis=1), axis=0)
            # df_dev_x = df_dev_x.div(df_dev_x.sum(axis=1), axis=0)

        # Gaussian filter to smooth out data
        if self.gaussian:
            print("Applying Gaussian Filter...")
            df_train_x = ndimage.filters.gaussian_filter(df_train_x, sigma=1)
            df_dev_x = ndimage.filters.gaussian_filter(df_dev_x, sigma=1)
            
        if self.standardize:
            # Standardize X data
            print("Standardizing...")
            std_scaler = StandardScaler()
            df_train_x = std_scaler.fit_transform(df_train_x)
            df_dev_x = std_scaler.fit_transform(df_dev_x)

        print("Finished Processing!")
        return df_train_x, df_dev_x
    
class CNN:
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
        df_train = pd.read_csv(self.train_dataset_path, header = 0)
        print(f"Loading evaluation dataset from {self.eval_dataset_path}...")
        df_eval = pd.read_csv(self.eval_dataset_path, header = 0)

        # Separate features and labels x = data, y = labels
        labels_train = df_train.LABEL.values
        labels_dev = df_eval.LABEL.values
        df_flux_train_raw = df_train.drop('LABEL', axis=1)
        df_flux_dev_raw = df_eval.drop('LABEL', axis=1)

        return labels_train, labels_dev, df_flux_train_raw, df_flux_dev_raw
    
    def preprocess_data(self, labels_train, labels_dev, df_flux_train_raw, df_flux_dev_raw):
        print("Applying SMOTE to balance classes...")
        #sm = SMOTE()
        #df_train_x, df_train_y = sm.fit_resample(df_train_x, df_train_y)
        
        print("Preprocessing data with Fourier Transform, Normalization, etc...")
        processor = LightFluxProcessor()   #applying preprocess to the data already divided between data and labels
        df_flux_train, df_flux_dev= processor.process(df_flux_train_raw, df_flux_dev_raw)
        
        labels_train = labels_train[:, :3136]
        labels_dev = labels_dev[:, :3136]
        flux_train = df_flux_train[:, :3136]
        flux_dev = df_flux_dev[:, :3136]

        return labels_train, labels_dev, flux_train, flux_dev  #the function just returns the values after the preprocessing
    
# Custom Dataset for the datacube
class DataCubeDataset(Dataset):
    def __init__(self, datacube, labels, transform=None):
        """
        Args:
            datacube (numpy.ndarray): 3D array of shape (5000, 56, 56)
            labels (numpy.ndarray): 1D array of shape (5000,) containing labels
        """
        self.datacube = datacube
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.datacube)

    def __getitem__(self, idx):
        image = self.datacube[idx]  # Shape: (56, 56)
        label = self.labels[idx]   # Scalar
        
        # Add channel dimension for grayscale images
        image = np.expand_dims(image, axis=0)  # Shape: (1, 56, 56)
        
        if self.transform:
            image = self.transform(image)
        
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, kernel_size):
        super(SimpleCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=kernel_size, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 14 * 14, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x

class CNNPipeline:
    def __init__(self, train_path, eval_path, batch_size=32, kernel_size = 3, learning_rate=0.001, epochs=20):
        self.train_path = train_path
        self.eval_path = eval_path
        self.batch_size = batch_size
        self.kernel_size = kernel_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_data(self):
        print("Loading datasets...")
        df_train = pd.read_csv(self.train_path)
        df_eval = pd.read_csv(self.eval_path)
        labels_train = df_train.LABEL.values - 1
        labels_eval = df_eval.LABEL.values - 1
        df_flux_train = df_train.drop(columns=['LABEL'])
        df_flux_eval = df_eval.drop(columns=['LABEL'])
        return labels_train, labels_eval, df_flux_train, df_flux_eval

    def preprocess_data(self, labels_train, labels_eval, df_flux_train, df_flux_eval):
        processor = LightFluxProcessor(fourier = False)
        df_flux_train, df_flux_eval = processor.process(df_flux_train, df_flux_eval)
        flux_train = df_flux_train[:, :3136]
        flux_eval = df_flux_eval[:, :3136]
        cube_train = flux_train.reshape(len(flux_train), 56, 56)
        cube_eval = flux_eval.reshape(len(flux_eval), 56, 56)
        return labels_train, labels_eval, cube_train, cube_eval

    def train_model(self, train_loader, val_loader):
        model = SimpleCNN(num_classes=2, kernel_size=self.kernel_size).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()

            print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {running_loss / len(train_loader):.4f}")

        return model

    def evaluate_model(self, model, train_loader, val_loader):
        model.eval()
        train_outputs, train_labels = [], []
        val_outputs, val_labels = [], []

        with torch.no_grad():
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                train_outputs.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                train_labels.extend(labels.cpu().numpy())

            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = model(inputs)
                val_outputs.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                val_labels.extend(labels.cpu().numpy())

        train_accuracy = accuracy_score(train_labels, train_outputs)
        val_accuracy = accuracy_score(val_labels, val_outputs)
        print(f"Training Accuracy: {train_accuracy * 100:.2f}%")
        print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    def run(self):
        labels_train, labels_eval, df_flux_train, df_flux_eval = self.load_data()
        labels_train, labels_eval, cube_train, cube_eval = self.preprocess_data(
            labels_train, labels_eval, df_flux_train, df_flux_eval
        )

        train_dataset = DataCubeDataset(cube_train, labels_train)
        eval_dataset = DataCubeDataset(cube_eval, labels_eval)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=False)

        model = self.train_model(train_loader, val_loader)
        self.evaluate_model(model, train_loader, val_loader)

    

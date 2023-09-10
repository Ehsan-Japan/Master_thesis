# -*- coding: utf-8 -*-
"""
Created on Tue Sep  5 01:59:36 2023

@author: ehsan
"""

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch.optim as optim
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torch.nn.functional as F
import pickle
import pandas as pd
import os

import sys
sys.path.append(r'C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\PlotNeuralNet-master')
from pycore.tikzeng import *




class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        if isinstance(data, torch.utils.data.dataset.Subset):
            self.dataframe = data.dataset.iloc[data.indices].reset_index(drop=True)
        else:
            self.dataframe = data
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        image_array = self.dataframe.iloc[idx]['image_filename']
        image = Image.fromarray(image_array).convert('L')  # Convert image to grayscale
    
        label = torch.tensor(self.dataframe.iloc[idx]['coordinates']).float()
    
        if self.transform:
            image = self.transform(image)
    
        return image, label
    

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16*256*256, 128)  # Assuming input image size is 512x512
        self.fc2 = nn.Linear(128, 4)  # Predicting 4 values (4 positons!)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = x.view(-1, 16*256*256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x    


model = SimpleCNN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])

# Split data into training and validation
# Define the path to the pickle file
directory_data=r"C:\Users\ehsan\OneDrive\Desktop\All2\Seminar Fujita\machine learning_QDs\Noisy Double Dot\slanted_H_training_data"
pickle_filepath = os.path.join(directory_data, "data.pkl")

# Load the DataFrame from the pickle file
with open(pickle_filepath, 'rb') as f:
    df = pickle.load(f)

train_size = int(0.8 * len(df))
val_size = len(df) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(df, [train_size, val_size])
train_loader = DataLoader(CustomDataset(train_dataset, transform=transform), batch_size=32, shuffle=True)
val_loader = DataLoader(CustomDataset(val_dataset, transform=transform), batch_size=32, shuffle=False)
num_epochs = 10
train_losses = []
val_losses = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    train_loss = running_loss / len(train_loader)
    train_losses.append(train_loss)

    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
    val_loss = running_loss / len(val_loader)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

import matplotlib.pyplot as plt


plt.figure(figsize=(10,5))
plt.title("Training and Validation Loss")
plt.plot(train_losses, label="train")
plt.plot(val_losses, label="val")
plt.xlabel("epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

    
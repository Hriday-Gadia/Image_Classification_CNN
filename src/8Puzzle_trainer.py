import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import importlib.util
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score,precision_score,recall_score
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as T
import json
from torch.utils.data import DataLoader, TensorDataset, random_split

module_path = "../src/8Puzzle_model.py"
module_name = "8Puzzle_model"

spec = importlib.util.spec_from_file_location(module_name, module_path)
puzzle_module = importlib.util.module_from_spec(spec)
sys.modules[module_name] = puzzle_module
spec.loader.exec_module(puzzle_module)

with open("../src/config8puz.json", "r") as f:
    config = json.load(f)

model_params = config["model_params"]
training_params = config["training_params"]
print(f" Model Parameters : {model_params}, Training Parameters : {training_params}")

# Load the saved data
bal_data , bal_label  = torch.load("../src/Dataset/8Puzzle/8Puzzle_balanced.pt",weights_only=True)
imbal_data , imbal_label = torch.load("../src/Dataset/8Puzzle/8Puzzle_imbalanced.pt",weights_only=True)

model1 = puzzle_module.ModularMLP(model_params["input_size"],model_params["activation"],model_params["hidden_layers"],model_params["output_size"])
model2 = puzzle_module.ModularMLP(model_params["input_size"],model_params["activation"],model_params["hidden_layers"],model_params["output_size"])

def train_model(model, images, labels, learningrate, epochs, device="cpu"):
    # Flatten images if needed
    if len(images.shape) > 2:
        images = images.view(images.size(0), -1)

    dataset = TensorDataset(images, labels)
    train, valid = random_split(dataset,[16000,4000])
    loadertrain = DataLoader(train, batch_size=1, shuffle=True)
    loaderval =  DataLoader(valid, batch_size=1, shuffle=True)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learningrate)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": []
    }

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        y_true_train, y_pred_train = [], []

        for batch_imgs, batch_labels in loadertrain:
            sample = batch_imgs[0].numpy().reshape(84,84)
            # Define the grid size (3x3)
            grid_size = 3
            height, width = sample.shape[0], sample.shape[1]

            partition_height = height // grid_size
            partition_width = width // grid_size
            partitioned_images = []
            for i in range(grid_size):
                for j in range(grid_size):
                # Slice the tensor into smaller parts
                    partition = sample[i*partition_height:(i+1)*partition_height, j*partition_width:(j+1)*partition_width]
                    partitioned_images.append(partition)
            for i in range(9):
                image = torch.tensor(partitioned_images[i], dtype=torch.float32).view(-1).unsqueeze(0).to(device)
                label = batch_labels[0][i].long().unsqueeze(0).to(device)
                outputs = model(image)
                loss = criterion(outputs, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                y_true_train.append(label.item())
                y_pred_train.append(outputs.argmax(dim=1).item())

        history["train_loss"].append(train_loss / (len(loadertrain)*9))
        history["train_acc"].append(accuracy_score(y_true_train, y_pred_train))
        history["train_precision"].append(precision_score(y_true_train, y_pred_train, average='macro', zero_division=0))
        history["train_recall"].append(recall_score(y_true_train, y_pred_train, average='macro', zero_division=0))

        model.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []

        for batch_imgs, batch_labels in loaderval:
            sample = batch_imgs[0].numpy().reshape(84,84)

            partitioned_images = []
            grid_size = 3
            partition_height = sample.shape[0] // grid_size
            partition_width = sample.shape[1] // grid_size

            for i in range(grid_size):
                for j in range(grid_size):
                    part = sample[i*partition_height:(i+1)*partition_height,
                                  j*partition_width:(j+1)*partition_width]
                    partitioned_images.append(part)

            for i in range(9):
                image = torch.tensor(partitioned_images[i], dtype=torch.float32).view(-1).unsqueeze(0).to(device)
                label = batch_labels[0][i].long().unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(image)
                    loss = criterion(outputs, label)

                val_loss += loss.item()
                y_true_val.append(label.item())
                y_pred_val.append(outputs.argmax(dim=1).item())

        history["val_loss"].append(val_loss / (len(loaderval)*9))
        history["val_acc"].append(accuracy_score(y_true_val, y_pred_val))
        history["val_precision"].append(precision_score(y_true_val, y_pred_val, average='macro', zero_division=0))
        history["val_recall"].append(recall_score(y_true_val, y_pred_val, average='macro', zero_division=0))

        print(f"Epoch {epoch+1}/{epochs} "
              f"Train Loss: {history['train_loss'][-1]:.4f} "
              f"Val Loss: {history['val_loss'][-1]:.4f} "
              f"Train Acc: {history['train_acc'][-1]:.4f} "
              f"Val Acc: {history['val_acc'][-1]:.4f}")
    return history

def test_model(model, images, labels, epochs, device="cpu"):
    # Flatten images if needed
    if len(images.shape) > 2:
        images = images.view(images.size(0), -1)

    dataset = TensorDataset(images, labels)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    model.to(device)
    criterion = nn.CrossEntropyLoss()


    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [],
        "train_precision": [], "val_precision": [],
        "train_recall": [], "val_recall": []
    }

    for epoch in range(epochs):
        model.eval()
        val_loss = 0.0
        y_true_val, y_pred_val = [], []

        for batch_imgs, batch_labels in loader:
            sample = batch_imgs[0].numpy().reshape(84,84)
            label_list = batch_labels[0]

            partitioned_images = []
            grid_size = 3
            partition_height = sample.shape[0] // grid_size
            partition_width = sample.shape[1] // grid_size

            for i in range(grid_size):
                for j in range(grid_size):
                    part = sample[i*partition_height:(i+1)*partition_height,
                                  j*partition_width:(j+1)*partition_width]
                    partitioned_images.append(part)

            for i in range(9):
                image = torch.tensor(partitioned_images[i], dtype=torch.float32).view(-1).unsqueeze(0).to(device)
                label = batch_labels[0][i].long().unsqueeze(0).to(device)

                with torch.no_grad():
                    outputs = model(image)
                    loss = criterion(outputs, label)

                val_loss += loss.item()
                y_true_val.append(label.item())
                y_pred_val.append(outputs.argmax(dim=1).item())

        history["val_loss"].append(val_loss / (len(loader)*9))
        history["val_acc"].append(accuracy_score(y_true_val, y_pred_val))
        history["val_precision"].append(precision_score(y_true_val, y_pred_val, average='macro', zero_division=0))
        history["val_recall"].append(recall_score(y_true_val, y_pred_val, average='macro', zero_division=0))

        print(f"Epoch {epoch+1}/{epochs} "
              f"Train Loss: {history['train_loss'][-1]:.4f} "
              f"Val Loss: {history['val_loss'][-1]:.4f} "
              f"Train Acc: {history['train_acc'][-1]:.4f} "
              f"Val Acc: {history['val_acc'][-1]:.4f}")
    return history


def plot_metrics(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    def plot_metric(metric_name):
        plt.figure()
        plt.plot(epochs, history[f"train_{metric_name}"], label="Train")
        plt.plot(epochs, history[f"val_{metric_name}"], label="Validation")
        plt.title(metric_name.capitalize()+ " Training with Balanced, Testing with Imbalanced")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

    for metric in ["loss", "acc", "precision", "recall"]:
        plot_metric(metric)

def plot_test(history):
    epochs = range(1, len(history["val_loss"]) + 1)

    def plot_metric(metric_name):
        plt.figure()
        plt.plot(epochs, history[f"test_{metric_name}"], label="Testing")
        plt.title(metric_name.capitalize() + " Training with Balanced, Testing with Imbalanced")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

    for metric in ["loss", "acc", "precision", "recall"]:
        plot_metric(metric)

def plot_metrics1(history):
    epochs = range(1, len(history["train_loss"]) + 1)

    def plot_metric(metric_name):
        plt.figure()
        plt.plot(epochs, history[f"train_{metric_name}"], label="Train")
        plt.plot(epochs, history[f"val_{metric_name}"], label="Validation")
        plt.title(metric_name.capitalize() + " Training with Imbalanced, Testing with Balanced")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

    for metric in ["loss", "acc", "precision", "recall"]:
        plot_metric(metric)

def plot_test1(history):
    epochs = range(1, len(history["val_loss"]) + 1)

    def plot_metric(metric_name):
        plt.figure()
        plt.plot(epochs, history[f"test_{metric_name}"], label="Testing")
        plt.title(metric_name.capitalize() + " Training with Imbalanced, Testing with Balanced")
        plt.xlabel("Epoch")
        plt.ylabel(metric_name.capitalize())
        plt.legend()
        plt.grid(True)
        plt.show()

    for metric in ["loss", "acc", "precision", "recall"]:
        plot_metric(metric)

#training model 1
history1 = train_model(model1,bal_data,bal_label,training_params["lr"],training_params["epochs"])
plot_metrics(history1)
history2 = test_model(model1,imbal_data,imbal_label,training_params["epochs"])
plot_test(history2)
#training model 2
history3 = train_model(model2,imbal_data,imbal_label,training_params["lr"],training_params["epochs"])
plot_metrics1(history3)
history4 = test_model(model2,bal_data,bal_label,training_params["epochs"])
plot_test1(history3)
torch.save(model1, "../src/modular.pth")

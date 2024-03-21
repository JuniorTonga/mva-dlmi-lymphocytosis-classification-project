import enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from PIL import Image
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset


class Resnet18Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), x.size(1))
        return x


class Resnet50Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), x.size(1))
        return x


class VGG16Encoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = models.vgg16()
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), 512 * 7 * 7)
        return x


class PredictionAggregator(enum.Enum):
    MIN  = 0
    MAX  = 1
    MEAN = 2


class ImageLabelDataset(Dataset):

    def __init__(self, images, labels, transform_function=None):
        self.images = images
        self.labels = labels
        self.transform_function = transform_function

    def __getitem__(self, index):
        image_path = self.images[index]
        image = image_to_numpy(image_path)
        if self.transform_function is not None:
            image = self.transform_function(image)
        return image, self.labels[index]

    def __len__(self):
        return len(self.labels)


class ImageDataset(Dataset):

    def __init__(self, images, transform_function=None):
        self.images = images
        self.transform_function = transform_function

    def __getitem__(self, index):
        image_path = self.images[index]
        image = image_to_numpy(image_path)
        if self.transform_function is not None:
            image = self.transform_function(image)
        return image

    def __len__(self):
        return len(self.images)


def image_to_numpy(image_path):
    image_pil = Image.open(image_path)
    image_np = np.array(image_pil, dtype=np.float32) / 255.0
    return image_np


def load_images(split_dir):
    images = {}

    for p in os.listdir(split_dir):
        p_dir = os.path.join(split_dir, p)
        if not os.path.isdir(p_dir):
            continue

        p_images = []
        for image_path in os.listdir(p_dir):
            p_images.append(os.path.join(p_dir, image_path))

        images[p] = p_images
    return images


def assign_labels(images_dict, labels_dict):
    images = []
    labels = []
    for p, p_images in images_dict.items():
        images.extend(p_images)
        labels.extend([labels_dict[p]] * len(p_images))
    return images, labels


def predict(
        model, 
        image_encoder, 
        p_dataloader, 
        prediction_aggregator, 
        device
    ):
    y_pred = []
    model.eval()
    with torch.no_grad():
        for X in p_dataloader:
            X = torch.FloatTensor(X).to(device)
            X = image_encoder(X)
            logits = model(X)
            pred = F.softmax(logits, dim=-1).argmax(dim=-1)
            y_pred.extend(pred.cpu().detach().numpy().tolist())
    if prediction_aggregator is PredictionAggregator.MIN:
        return np.min(y_pred)
    if prediction_aggregator is PredictionAggregator.MAX:
        return np.max(y_pred)
    return int(np.mean(y_pred) > .5) # MEAN


def predict_all(
        model, 
        image_encoder, 
        images_dict, 
        transform_function, 
        batch_size, 
        prediction_aggregator, 
        device
    ):
    y_pred = []
    for p, p_images in images_dict.items():
        p_dataset = ImageDataset(p_images, transform_function)
        p_dataloader = DataLoader(p_dataset, batch_size=batch_size, shuffle=False)
        y = predict(model, image_encoder, p_dataloader, prediction_aggregator, device)
        y_pred.append(y)
    return np.array(y_pred)


def test(model, image_encoder, loss_fn, dataloader, device):
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for X, y in dataloader:
            X = torch.FloatTensor(X).to(device)
            y = torch.LongTensor(y).to(device)
            X = image_encoder(X)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            #accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
            y_true.extend(y.cpu().detach().numpy())
            y_pred.extend(pred.argmax(1).cpu().detach().numpy())
    test_loss /= num_batches
    acc = metrics.accuracy_score(y_true, y_pred)
    bacc = metrics.balanced_accuracy_score(y_true, y_pred)
    return test_loss, acc, bacc


def train(model, image_encoder, optimizer, loss_fn, dataloader, device):
    num_batches = len(dataloader)
    model.train()
    train_loss = 0
    y_true, y_pred = [], []
    for X, y in dataloader:
        X = torch.FloatTensor(X).to(device)
        y = torch.LongTensor(y).to(device)
        with torch.no_grad():
            X = image_encoder(X)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        #accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
        y_true.extend(y.cpu().detach().numpy())
        y_pred.extend(pred.argmax(1).cpu().detach().numpy())
    train_loss /= num_batches
    acc = metrics.accuracy_score(y_true, y_pred)
    bacc = metrics.balanced_accuracy_score(y_true, y_pred)
    return train_loss, acc, bacc


def trainer(
        model, 
        image_encoder,
        optimizer, 
        loss_fn, 
        train_dataloader, 
        test_dataloader, 
        n_epochs, 
        device=None,
        verbose=True,
        verbose_every=10,
        plot=True
    ):
    train_losses, train_accuracies, train_bal_accuracies = [], [], []
    test_losses, test_accuracies, test_bal_accuracies = [], [], []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc, train_bacc = train(model, image_encoder, optimizer, loss_fn, train_dataloader, device)
        test_loss, test_acc, test_bacc = test(model, image_encoder, loss_fn, test_dataloader, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        train_bal_accuracies.append(train_bacc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)
        test_bal_accuracies.append(test_bacc)

        if verbose and epoch % verbose_every == 0:
            print(f"[Epoch {epoch} / {n_epochs}]",
                f"train loss = {train_loss:.5f} acc = {train_acc:.5f} bacc = {train_bacc:.5f}",
                f"test loss = {test_loss:.5f} acc = {test_acc:.5f} bacc = {test_bacc:.5f}",
                sep="\n\t")

    if plot:
        plt.figure()
        plt.plot(train_bal_accuracies, label="Train balanced accuracy")
        plt.plot(test_bal_accuracies, label="Test balanced accuracy")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(train_accuracies, label="Train accuracy")
        plt.plot(test_accuracies, label="Test accuracy")
        plt.legend()
        plt.show()

        plt.figure()
        plt.plot(train_losses, label="Train loss")
        plt.plot(test_losses, label="Test loss")
        plt.legend()
        plt.show()

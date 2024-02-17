import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn import metrics, model_selection
from torch.utils.data import DataLoader, Dataset


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


def predict(model, p_dataloader, device):
    y_pred = []
    model.eval()
    with torch.no_grad():
        for X in p_dataloader:
            X = torch.FloatTensor(X).to(device)
            logits = model(X)
            pred = F.softmax(logits, dim=-1).argmax(dim=-1)
            y_pred.extend(pred.cpu().detach().numpy().tolist())
    return int(np.mean(y_pred) > .5)

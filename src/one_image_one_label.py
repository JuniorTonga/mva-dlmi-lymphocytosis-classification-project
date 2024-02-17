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

    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image_path = self.images[index]
        image = image_to_numpy(image_path)
        return image, self.labels[index]

    def __len__(self):
        return len(self.labels)


class ImageDataset(Dataset):

    def __init__(self, images):
        self.images = images

    def __getitem__(self, index):
        image_path = self.images[index]
        image = image_to_numpy(image_path)
        return image

    def __len__(self):
        return len(self.images)


def image_to_numpy(image_path):
    image_pil = Image.open(image_path)
    image_np = np.array(image_pil)
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

    for p, p_images in images_dict:
        images.extend(p_images)
        labels.extend([labels_dict[p]] * len(p_images))

    return images, labels


def predict(model, p_dataloader):
    y_pred = []
    for X in p_dataloader:
        pass
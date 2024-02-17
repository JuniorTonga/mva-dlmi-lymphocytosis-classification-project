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
        return self.images[index], self.labels[index]

    def __len__(self):
        return len(self.labels)


class ImageDataset(Dataset):

    def __init__(self, images):
        self.images = images

    def __getitem__(self, index):
        return self.images[index]

    def __len__(self):
        return len(self.images)


def load_images(split_dir):
    images = {}

    for p in os.listdir(split_dir):
        p_dir = os.path.join(split_dir, p)
        if not os.path.isdir(p_dir):
            continue

        p_images = []
        for image_path in os.listdir(p_dir):
            image_pil = Image.open(os.path.join(p_dir, image_path))
            image_array = np.array(image_pil).tolist()
            p_images.append(image_array)

        images[p] = p_images
    return images


def assign_labels(images_dict, labels_dict):
    images = []
    labels = []

    for p, p_images in images_dict:
        images.extend(p_images)
        labels.extend([labels_dict[p]] * len(p_images))

    return images, labels


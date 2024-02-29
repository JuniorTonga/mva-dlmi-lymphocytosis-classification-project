import enum
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from sklearn import metrics, model_selection
from torch.utils.data import DataLoader, Dataset


def image_to_numpy(image_path):
    image_pil = Image.open(image_path)
    image_np = np.array(image_pil, dtype=np.float32) / 255.0
    return image_np


class PatientImageDataset(Dataset):

    def __init__(self, patient_path, transform_function=None):
        super().__init__()
        self.patient_path = patient_path
        self.transform_function = transform_function

        images_path = []
        for image_path in os.listdir(patient_path):
            images_path.append(os.path.join(patient_path, image_path))

        self.images_path = images_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        image_path = self.images_path[index]
        image = image_to_numpy(image_path)
        if self.transform_function is not None:
            image = self.transform_function(image)
        return image


class PatientDatasetItem(object):

    def __init__(self, 
                patient_id, 
                age, 
                sex, 
                lymph_count, 
                label, 
                images_dataloader) -> None:
        self.patient_id = patient_id
        self.age = age
        self.sex = sex
        self.lymph_count = lymph_count
        self.label = label
        self.images_dataloader = images_dataloader


class PatientDataset(Dataset):

    def __init__(self, patients_items):
        super().__init__()
        self.patients_items = patients_items

    def __len__(self):
        return len(self.patients_items)

    def __getitem__(self, index):
        return self.patients_items[index]


class AttributesModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 2)
        )

    def forward(self, x):
        return self.model(x)


class AggregationOperator(enum.Enum):
    MAX = 0
    MIN = 1
    AVG = 2
    SUM = 3



        
        

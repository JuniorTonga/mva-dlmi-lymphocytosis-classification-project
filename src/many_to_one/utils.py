import enum
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models
from PIL import Image
from sklearn import metrics
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


class FeatureAggregator(enum.Enum):
    MIN = 0
    MAX = 1
    MEAN = 2
    ATTENTION = 3


class FeatureExtractor(enum.Enum):
    CNN = 0
    RESNET18 = 1
    RESNET50 = 2
    VGG16 = 3


class PretrainFeatureExtractor(nn.Module):

    def __init__(self, feature_extractor) -> None:
        super().__init__()
        
        if feature_extractor is FeatureExtractor.RESNET18:
            self.encoder = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        elif feature_extractor is FeatureExtractor.RESNET50:
            self.encoder = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        elif feature_extractor is FeatureExtractor.VGG16:
            self.encoder = models.vgg16()
        else:
            raise ValueError("Unknown pre-train feature extractor")
        
        self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        return x


class BagImageModel(nn.Module):

    def __init__(
        self,
        feature_extractor=FeatureExtractor.CNN,
        feature_aggregator=FeatureAggregator.MIN,
        hidden_dim=128,
        num_classes=2,
        attn_num_heads=4,
        image_size=224
    ):
        super().__init__()

        if feature_extractor is FeatureExtractor.CNN:
            self._pretrain_feature_extractor = False
            self.extractor = nn.Sequential(
                nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(4),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(8),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1, stride=1),
                nn.BatchNorm2d(16),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Flatten(),
                nn.Dropout(0.5),
            )
        else:
            self._pretrain_feature_extractor = True
            self.extractor = PretrainFeatureExtractor(feature_extractor)

        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.d_model = -1
        with torch.no_grad():
            x = torch.rand(1, 3, image_size, image_size)
            x = self.extractor(x)
            self.d_model = x.size(1)
        self.project = nn.Linear(self.d_model, hidden_dim)

        if feature_aggregator is FeatureAggregator.MIN:
            self.aggregator = lambda x: torch.min(x, dim=-2).values
        elif feature_aggregator is FeatureAggregator.MAX:
            self.aggregator = lambda x: torch.max(x, dim=-2).values
        elif feature_aggregator is FeatureAggregator.MEAN:
            self.aggregator = lambda x: torch.mean(x, dim=-2)
        elif feature_aggregator is FeatureAggregator.ATTENTION:
            self.aggregator = nn.MultiheadAttention(
                embed_dim=hidden_dim, batch_first=True, num_heads=attn_num_heads
            )
        else:
            raise ValueError("Unknown feature aggregator")
        self.feature_aggregator = feature_aggregator

        self.batch_norm = nn.BatchNorm1d(self.hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, num_classes)
        )

    def forward(self, x: torch.Tensor):
        batch_size, seq_len, in_chan, im_width, im_height = x.size()
        x = x.view(batch_size * seq_len, in_chan, im_width, im_height)

        if self._pretrain_feature_extractor:
            with torch.no_grad():
                x = self.extractor(x) # (BxS, D)
        else:
            x = self.extractor(x) # (BxS, D)

        x = x.view(batch_size * seq_len, self.d_model) # (BxS, D)
        x = self.project(x) # (BxS, H)

        if seq_len > 1:
            x = self.batch_norm(x.view(seq_len, self.hidden_dim, batch_size)) # (S, H, B)
        x = x.view(batch_size, seq_len, self.hidden_dim) # (B, S, H)

        if self.feature_aggregator is FeatureAggregator.ATTENTION:
            attn, _ = self.aggregator(x, x, x, need_weights=False) # (B, S, H)
            x = torch.sum(attn, dim=1) # (B, H)
        else:
            x = self.aggregator(x) # (B, H)

        x = self.head(x) # (B, C)
        return x


class PredictionAggregator(enum.Enum):
    MIN  = 0
    MAX  = 1
    MEAN = 2


class PatientImageDataset(Dataset):

    def __init__(self, 
        patient_path, 
        transform=None,
    ):
        super().__init__()
        self.patient_path = patient_path
        self.transform = transform

        images_path = []
        for image_path in os.listdir(patient_path):
            images_path.append(os.path.join(patient_path, image_path))

        self.images_path = images_path

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, index):
        image_path = self.images_path[index]
        image = image_to_numpy(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image


class PatientsDataset(Dataset):

    def __init__(
        self, 
        split_dir, 
        patients_ids,
        patients_labels
    ):
        super().__init__()
        self.split_dir = split_dir
        self.patients_ids = patients_ids
        self.patients_labels = patients_labels

    def __len__(self):
        return len(self.patients_ids)

    def __getitem__(self, index):
        patient_id = self.patients_ids[index]
        patient_label = self.patients_labels[index]
        patient_path = os.path.join(self.split_dir, patient_id)
        return {
            "id": patient_id, 
            "label": patient_label, 
            "patient_path": patient_path
        }


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
    dataloader,
    seq_len,
    transform,
    prediction_aggregator,
    device
):
    all_ids = []
    all_preds = []

    with torch.no_grad():
        for data in tqdm(dataloader, "Predict"):

            patient_id = data["id"][0]
            batch_preds = []

            patient_path = data["patient_path"][0]
            patient_dataset = PatientImageDataset(
                patient_path=patient_path,
                transform=transform,
            )
            patient_dataloader = DataLoader(patient_dataset, batch_size=seq_len, shuffle=False)

            for batch_images in patient_dataloader:
                x = batch_images.unsqueeze(0).to(device)
                logits = model(x)
                preds = logits.argmax(dim=-1).cpu().detach().numpy()
                batch_preds.extend(preds)

            if prediction_aggregator is PredictionAggregator.MIN:
                patient_pred = np.min(batch_preds)
            elif prediction_aggregator is PredictionAggregator.MAX:
                patient_pred = np.max(batch_preds)
            else:
                patient_pred = int(np.mean(batch_preds) > 0.5)

            all_preds.append(patient_pred)
            all_ids.append(patient_id)

    return all_ids, all_preds


def test(
    model,
    criterion,
    dataloader,
    seq_len,
    transform,
    prediction_aggregator,
    device
):
    total_loss = 0
    all_preds = []
    all_true = []

    with torch.no_grad():
        for data in tqdm(dataloader, "Test"):

            patient_label_item = data["label"].item()
            patient_label = data["label"].to(device)
            batch_preds = []
            patient_loss = 0

            patient_path = data["patient_path"][0]
            patient_dataset = PatientImageDataset(
                patient_path=patient_path,
                transform=transform,
            )
            patient_dataloader = DataLoader(patient_dataset, batch_size=seq_len, shuffle=False)

            for batch_images in patient_dataloader:
                x = batch_images.unsqueeze(0).to(device)
                logits = model(x)
                loss = criterion(logits, patient_label)

                patient_loss += loss.item()
                preds = logits.argmax(dim=-1).cpu().detach().numpy()
                batch_preds.extend(preds)

            patient_loss /= len(patient_dataset)
            total_loss += patient_loss
            
            if prediction_aggregator is PredictionAggregator.MIN:
                patient_pred = np.min(batch_preds)
            elif prediction_aggregator is PredictionAggregator.MAX:
                patient_pred = np.max(batch_preds)
            else:
                patient_pred = int(np.mean(batch_preds) > 0.5)

            all_preds.append(patient_pred)
            all_true.append(patient_label_item)

    total_loss /= len(dataloader.dataset)
    acc = metrics.accuracy_score(all_true, all_preds)
    bacc = metrics.balanced_accuracy_score(all_true, all_preds)

    return total_loss, acc, bacc


def train(
    model,
    optimizer,
    criterion,
    dataloader,
    seq_len,
    transform,
    prediction_aggregator,
    device
):
    total_loss = 0
    all_preds = []
    all_true = []

    for data in tqdm(dataloader, "Train"):

        patient_label_item = data["label"].item()
        patient_label = data["label"].to(device)
        batch_preds = []
        patient_loss = 0

        patient_path = data["patient_path"][0]
        patient_dataset = PatientImageDataset(
            patient_path=patient_path,
            transform=transform,
        )
        patient_dataloader = DataLoader(patient_dataset, batch_size=seq_len, shuffle=True)

        for batch_images in patient_dataloader:
            x = batch_images.unsqueeze(0).to(device)
            logits = model(x)
            loss = criterion(logits, patient_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            patient_loss += loss.item()
            preds = logits.argmax(dim=-1).cpu().detach().numpy()
            batch_preds.extend(preds)

        patient_loss /= len(patient_dataset)
        total_loss += patient_loss

        if prediction_aggregator is PredictionAggregator.MIN:
            patient_pred = np.min(batch_preds)
        elif prediction_aggregator is PredictionAggregator.MAX:
            patient_pred = np.max(batch_preds)
        else:
            patient_pred = int(np.mean(batch_preds) > 0.5)

        all_preds.append(patient_pred)
        all_true.append(patient_label_item)

    total_loss /= len(dataloader.dataset)
    acc = metrics.accuracy_score(all_true, all_preds)
    bacc = metrics.balanced_accuracy_score(all_true, all_preds)

    return total_loss, acc, bacc


def trainer(
    model,
    optimizer, 
    criterion, 
    train_dataloader, 
    test_dataloader, 
    n_epochs,
    seq_len,
    train_transform,
    test_transform,
    prediction_aggregator,
    device,
    model_name,
    verbose=True,
    verbose_every=1,
    save=True,
    save_every=1,
    save_dir=".",
    plot=True
):
    train_accs      = []
    train_baccs     = []
    train_losses    = []
    test_accs       = []
    test_baccs      = []
    test_losses     = []

    best_bacc = 0

    for epoch in range(1, n_epochs + 1):
        train_loss, train_acc, train_bacc = train(
            model, optimizer, criterion, train_dataloader,
            seq_len, train_transform, prediction_aggregator, device
        )
        test_loss, test_acc, test_bacc = test(
            model, criterion, test_dataloader,
            seq_len, test_transform, prediction_aggregator, device
        )

        train_accs.append(train_acc)
        train_baccs.append(train_bacc)
        train_losses.append(train_loss)

        test_accs.append(test_acc)
        test_baccs.append(test_bacc)
        test_losses.append(test_loss)

        if verbose and epoch % verbose_every == 0:
            print(
                f"[Epoch {epoch} / {n_epochs}]",
                f"\ttrain loss = {train_loss:.4f} acc = {train_acc:.4f} bacc = {train_bacc:.4f}",
                f"\ttest loss = {test_loss:.4f} acc = {test_acc:.4f} bacc = {test_bacc:.4f}", 
                sep="\n"
            )

        if save and epoch % save_every == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 
                os.path.join(save_dir, f"{model_name}.pt")
            )

        if test_bacc > best_bacc:
            best_bacc = test_bacc
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, 
                os.path.join(save_dir, f"{model_name}_best.pt")
            )

    if plot:
        plt.figure()
        plt.plot(train_accs, label="Train")
        plt.plot(test_accs, label="Test")
        plt.title("Accuracy")
        plt.show()

        plt.figure()
        plt.plot(train_baccs, label="Train")
        plt.plot(test_baccs, label="Test")
        plt.title("Balanced accuracy")
        plt.show()

        plt.figure()
        plt.plot(train_losses, label="Train")
        plt.plot(test_losses, label="Test")
        plt.title("Loss")
        plt.show()
    

def load_checkpoint(checkpoint_path, model_args={}, optimizer_args={}):
    model = BagImageModel(**model_args)
    optimizer = optim.Adam(model.parameters(), **optimizer_args)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch

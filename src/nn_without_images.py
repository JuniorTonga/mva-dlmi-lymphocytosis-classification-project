import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics, model_selection
from torch.utils.data import DataLoader, Dataset


class LDataset(Dataset):

    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def __getitem__(self, index):
        return self.X[index], self.Y[index]

    def __len__(self):
        return len(self.Y)


class XDataset(Dataset):

    def __init__(self, X):
        self.X = X

    def __getitem__(self, index):
        return self.X[index]

    def __len__(self):
        return len(self.X)


def test(model, loss_fn, dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, accuracy = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X = torch.FloatTensor(X).to(device)
            y = torch.LongTensor(y).to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy /= size
    return test_loss, accuracy


def train(model, optimizer, loss_fn, dataloader):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_loss, accuracy = 0, 0
    for X, y in dataloader:
        X = torch.FloatTensor(X).to(device)
        y = torch.LongTensor(y).to(device)
        pred = model(X)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        train_loss += loss.item()
        accuracy += (pred.argmax(1) == y).type(torch.float).sum().item()
    train_loss /= num_batches
    accuracy /= size
    return train_loss, accuracy


def trainer(model, optimizer, loss_fn, train_dataloader, test_dataloader, n_epochs):
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = train(model, optimizer, loss_fn, train_dataloader)
        test_loss, test_accuracy = test(model, loss_fn, test_dataloader)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if epoch % 10 == 0:
            print(f"[Epoch {epoch} / {n_epochs}] train loss = {train_loss:.2f} acc = {train_accuracy:.2f} ",
                f"test loss = {test_loss:.2f} acc = {test_accuracy:.2f}")

    return (train_losses, train_accuracies), (test_losses, test_accuracies)


def nn_predict(model, dataloader):
    y = []
    model.eval()
    with torch.no_grad():
        for X in dataloader:
            X = torch.FloatTensor(X).to(device)
            logits = model(X)
            pred = F.softmax(logits, dim=-1).argmax(dim=-1)
            y.extend(pred.cpu().detach().numpy().tolist())
    return np.array(y)


if __name__ == "__main__":
    data_dir = "../data/dlmi-lymphocytosis-classification/"
    compute_age = lambda x: 2024 - int(x.replace("-", "/").split("/")[-1])
    gender_to_int = lambda x: 1 if x.lower() == "m" else 0

    trainset_data_df = pd.read_csv(data_dir + "trainset/trainset_true.csv")
    trainset_data_df["AGE"] = trainset_data_df["DOB"].apply(compute_age)
    trainset_data_df["GENDER"] = trainset_data_df["GENDER"].apply(gender_to_int)

    testset_data_df = pd.read_csv(data_dir + "testset/testset_data.csv")
    testset_data_df["AGE"] = testset_data_df["DOB"].apply(compute_age)
    testset_data_df["GENDER"] = testset_data_df["GENDER"].apply(gender_to_int)

    selected_columns = ["GENDER", "LYMPH_COUNT", "AGE"]
    X_train_val = trainset_data_df[selected_columns].to_numpy(dtype=np.float32)
    X_test = testset_data_df[selected_columns].to_numpy(dtype=np.float32)
    y_train_val = trainset_data_df["LABEL"].to_numpy(dtype=int)

    X_train, X_val, y_train, y_val = model_selection.train_test_split(X_train_val, y_train_val)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    batch_size = 8
    train_dataset = LDataset(X_train, y_train)
    test_dataset = LDataset(X_val, y_val)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    X_train_val_dataset = XDataset(X_train_val)
    X_val_dataset = XDataset(X_val)
    X_test_dataset = XDataset(X_test)
    X_train_val_dataloader = DataLoader(X_train_val_dataset, batch_size=batch_size, shuffle=False)
    X_val_dataloader = DataLoader(X_val_dataset, batch_size=batch_size, shuffle=False)
    X_test_dataloader = DataLoader(X_test_dataset, batch_size=batch_size, shuffle=False)

    train_test_dataset = LDataset(X_train_val, y_train_val)
    train_test_dataloader = DataLoader(train_test_dataset, batch_size=batch_size, shuffle=False)

    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2)
    ).to(device)
    lr = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    loss_fn = nn.CrossEntropyLoss()

    n_epochs = 100
    (train_losses, train_accuracies), (test_losses, test_accuracies) = trainer(
        model, optimizer, loss_fn, train_dataloader, test_dataloader, n_epochs
    )

    y_pred = nn_predict(model, X_val_dataloader)
    acc = metrics.accuracy_score(y_val, y_pred)
    balanced_acc = metrics.balanced_accuracy_score(y_val, y_pred)
    print(f"Val acc = {acc:.2f} bal. acc = {balanced_acc:.2f}")

    model = nn.Sequential(
        nn.Linear(3, 8),
        nn.ReLU(),
        nn.Linear(8, 4),
        nn.ReLU(),
        nn.Linear(4, 2)
    ).to(device)
    lr = 1e-2
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) 
    loss_fn = nn.CrossEntropyLoss()

    n_epochs = 100
    (train_losses, train_accuracies), (test_losses, test_accuracies) = trainer(
        model, optimizer, loss_fn, train_test_dataloader, train_test_dataloader, n_epochs
    )

    y_test_pred = nn_predict(model, X_test_dataloader)
    submission_df = testset_data_df[["ID"]]
    submission_df["LABEL"] = y_test_pred
    submission_df = submission_df.rename({"ID": "Id", "LABEL": "Predicted"}, axis=1)
    submission_df.to_csv("../submissions/nn_baseline.csv", index=False)
    
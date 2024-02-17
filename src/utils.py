import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def test(model, loss_fn, dataloader, device):
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


def train(model, optimizer, loss_fn, dataloader, device):
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


def trainer(model, optimizer, loss_fn, train_dataloader, test_dataloader, n_epochs, device=None):
    train_losses, train_accuracies = [], []
    test_losses, test_accuracies = [], []

    for epoch in range(1, n_epochs + 1):
        train_loss, train_accuracy = train(model, optimizer, loss_fn, train_dataloader, device)
        test_loss, test_accuracy = test(model, loss_fn, test_dataloader, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

        if epoch % 10 == 0:
            print(f"[Epoch {epoch} / {n_epochs}] train loss = {train_loss:.2f} acc = {train_accuracy:.2f} ",
                f"test loss = {test_loss:.2f} acc = {test_accuracy:.2f}")

    return (train_losses, train_accuracies), (test_losses, test_accuracies)


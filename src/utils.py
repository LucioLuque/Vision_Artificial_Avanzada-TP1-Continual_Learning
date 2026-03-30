import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

def deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def _to_local_labels(y, task_number, num_classes):
    class_offset = task_number * num_classes
    y_local = y - class_offset
    if y_local.min().item() < 0 or y_local.max().item() >= num_classes:
        raise ValueError(
            f"Labels out of range for task {task_number}. "
            f"Expected labels in [{class_offset}, {class_offset + num_classes - 1}], "
            f"got min={y.min().item()} max={y.max().item()}."
        )
    return y_local

def train(model, dataloader, optimizer, criterion, title, epochs, task_number, save=True):
    device = next(model.parameters()).device
    writer = SummaryWriter(log_dir=f"../runs/{title}")
    epoch_bar = tqdm(range(epochs), desc = "Epochs", unit = "epoch")

    for epoch in epoch_bar:
        batch_bar = tqdm(dataloader, desc = f"Epoch {epoch+1}/{epochs}", leave=False, unit="batch")
        for i, (x, y) in enumerate(batch_bar):
            x_all, y_all = x.to(device), y.to(device) # ver randaugment

            optimizer.zero_grad()
            pred = model(x_all, task_number)
            y_local = _to_local_labels(y_all, task_number, pred.size(1))
            loss = criterion(pred, y_local)
            writer.add_scalar("Loss/Train", loss.item(), epoch * len(dataloader) + i)
            # print(loss.item())

            loss.backward()
            optimizer.step()

            batch_bar.set_postfix({"loss": loss.item()})
        epoch_bar.set_postfix({"loss": loss.item()})


    writer.close()
    if save == True:
        model.save(f"../models/weights/{title}.pth")


def backbone_train(model, dataloader, optimizer, criterion, title, tsne, epochs=5):
    device = next(model.parameters()).device
    writer = SummaryWriter(log_dir=f"../runs/{title}")

    epoch_bar = tqdm(range(epochs), desc = "Epochs", unit = "epoch")
    for epoch in epoch_bar:
        batch_bar = tqdm(dataloader, desc = f"Epoch {epoch+1}/{epochs}", leave=False, unit="batch")
        for i, (x, y) in enumerate(batch_bar):
            x_all, y_all = x.to(device), y.to(device) # ver randaugment

            optimizer.zero_grad()
            embeddings = model(x_all)
            loss = criterion(embeddings, y_all)

            if (epoch == 0 or epoch == epochs // 2 or epoch == epochs - 1) and i == 0:
                emb_2d = tsne.fit_transform(embeddings.detach().cpu().numpy())
                labels = y_all.detach().cpu().numpy()
                binary_labels = (labels == labels[0]).astype(int)
                
                plt.figure(figsize=(8, 6))
                scatter = plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=binary_labels, cmap='bwr', vmin=0, vmax=1)
                plt.colorbar(scatter, ticks=[0, 1], label='Class')
                plt.title(f"Epoch {epoch}")
                os.makedirs("../images", exist_ok=True)
                plt.savefig(f"../images/tsne_epoch_{epoch}.png", dpi=150, facecolor='white')
                plt.show()
                plt.close()
                

            writer.add_scalar("Loss/Train", loss.item(), epoch * len(dataloader) + i)

            loss.backward()
            optimizer.step()

            batch_bar.set_postfix({"loss": loss.item()})
        epoch_bar.set_postfix({"loss": loss.item()})

    writer.close()
    model.save(f"../models/weights/{title}.pth")

def accuracy(model, val_data, task_number):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in tqdm(val_data, desc="Evaluating", unit="batch"):
            x, y = x.to(device), y.to(device)
            pred = model(x, task_number)

            y_local = _to_local_labels(y, task_number, pred.size(1))
            _, predicted = torch.max(pred.data, 1)
            total += y_local.size(0)
            correct += (predicted == y_local).sum().item()
        return correct / total
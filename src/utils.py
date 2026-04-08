import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter
import os

from backbone import BackBone
from multiheadmodel import MultiHeadModel
from dataset import get_data_loaders


def deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def _to_local_labels(y, task_number, num_classes, global_labels=False):
    if global_labels:
        # CIL: las labels ya son globales, no hay offset
        if y.min().item() < 0 or y.max().item() >= num_classes:
            raise ValueError(
                f"Labels out of range. Expected [0, {num_classes - 1}], "
                f"got min={y.min().item()} max={y.max().item()}."
            )
        return y
    # TIL: offset por task
    class_offset = task_number * num_classes
    y_local = y - class_offset
    if y_local.min().item() < 0 or y_local.max().item() >= num_classes:
        raise ValueError(
            f"Labels out of range for task {task_number}. "
            f"Expected labels in [{class_offset}, {class_offset + num_classes - 1}], "
            f"got min={y.min().item()} max={y.max().item()}."
        )
    return y_local


def _maybe_update_criterion(criterion, model, dataloader, task_number):
    update_fn = getattr(criterion, "update", None)
    if callable(update_fn):
        update_fn(model=model, dataloader=dataloader, task_number=task_number)

def train(model, dataloader, optimizer, criterion, title, epochs, task_number, save=True, global_labels=False):
    #se puede unificar title con save, siempore que no escribimo la loss no guardamos el modelo
    #global_labels is f
    device = next(model.parameters()).device
    if title is not None:
        writer = SummaryWriter(log_dir=f"../runs/{title}")
    epoch_bar = tqdm(range(epochs), desc = "Epochs", unit = "epoch")

    for epoch in epoch_bar:
        batch_bar = tqdm(dataloader, desc = f"Epoch {epoch+1}/{epochs}", leave=False, unit="batch")
        for i, (x, y) in enumerate(batch_bar):
            x_all, y_all = x.to(device), y.to(device) # ver randaugment

            if hasattr(criterion, "x_cache"):
                criterion.x_cache = x_all
            
            optimizer.zero_grad()
            pred = model(x_all, task_number)
            y_local = _to_local_labels(y_all, task_number, pred.size(1), global_labels)
            loss = criterion(pred, y_local)
            if title is not None:
                writer.add_scalar("Loss/Train", loss.item(), epoch * len(dataloader) + i)
            # print(loss.item())

            loss.backward()
            optimizer.step()

            batch_bar.set_postfix({"loss": loss.item()})
        epoch_bar.set_postfix({"loss": loss.item()})

    if title is not None:
        writer.close()
    if save == True:
        model.save(f"../models/weights/{title}.pth")

def new_model_from_backbone(device, path = "../models/weights/backbone.pth"):
    backbone = BackBone()
    backbone.load_state_dict(torch.load(path))
    model = MultiHeadModel(backbone)
    model.to(device)
    return model

def run_til_experiment(criterion, title, device, batch_size=512, epochs = 20, seed=42):
    deterministic(seed)
    model = new_model_from_backbone(device)
    dataloaders = get_data_loaders(batch_size=batch_size)

    for task in range(len(dataloaders)):
        model.add_head(task)
        model.to(device)
        train_data = dataloaders[task][0]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train(model, train_data, optimizer, criterion, None, epochs, task_number=task, save=False)
        _maybe_update_criterion(criterion, model, dataloaders[task][0], task_number=task)
        
        # Accuracy después de cada entrenamiento
        eval_data = dataloaders[task][1]
        acc = accuracy(model, eval_data, task_number=task)
        print(f"Task {task} Accuracy: {acc:.4f}")

    # Accuracy después de terminar
    for i in range(len(dataloaders)):
        eval_data = dataloaders[i][1]
        acc = accuracy(model, eval_data, task_number=i)
        print(f"Task {i} Accuracy: {acc:.4f}")

    model.save(f"../models/weights/{title}_TIL.pth")

def run_cil_experiment(criterion, title, device, batch_size=512, epochs = 20, seed=42):
    deterministic(seed)
    model = new_model_from_backbone(device)
    dataloaders = get_data_loaders(batch_size=batch_size)

    for task in range(len(dataloaders)):
        print(f"Training task {task}")
        if task == 0:
            model.add_head(task)
        else:
            model.expand_head()
        model.to(device)
        train_data = dataloaders[task][0]
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

        train(model, train_data, optimizer, criterion, None, epochs, task_number=0, save=False, global_labels=True)
        _maybe_update_criterion(criterion, model, dataloaders[task][0], task_number=0)
        
        for trained_task in range(task + 1):
            eval_data = dataloaders[trained_task][1]
            acc = accuracy(model, eval_data, task_number=0, global_labels=True)
            print(f"Task {trained_task} Accuracy: {acc:.4f}")

    model.save(f"../models/weights/{title}_CIL.pth")

def backbone_train(model, dataloader, optimizer, criterion, title, tsne, epochs=5):
    model.train()
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

def accuracy(model, val_data, task_number, global_labels=False):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in tqdm(val_data, desc="Evaluating", unit="batch"):
            x, y = x.to(device), y.to(device)
            pred = model(x, task_number)

            y_local = _to_local_labels(y, task_number, pred.size(1), global_labels)
            _, predicted = torch.max(pred.data, 1)
            total += y_local.size(0)
            correct += (predicted == y_local).sum().item()
        return correct / total



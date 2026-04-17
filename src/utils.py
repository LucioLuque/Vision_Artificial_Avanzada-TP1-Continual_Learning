import random
import os

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset

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


def _maybe_update_criterion(criterion, dataloader, task_number):
    update_fn = getattr(criterion, "update", None)
    if callable(update_fn):
        update_fn(dataloader=dataloader, task_number=task_number)


def _freeze_backbone(model):
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    model.backbone.eval()


def extract_features_from_loader(backbone, loader, device):
    feat_chunks, y_chunks = [], []
    backbone.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            feat = backbone(x)
            feat_chunks.append(feat.cpu())
            y_chunks.append(y.cpu())

    if len(feat_chunks) == 0:
        raise ValueError("Feature extraction received an empty loader.")

    return torch.cat(feat_chunks, dim=0), torch.cat(y_chunks, dim=0)


def _build_feature_loader(feat_cpu, y_cpu, batch_size, num_workers):
    return DataLoader(
        TensorDataset(feat_cpu, y_cpu),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )


def _train_on_features(
    model,
    feature_loader,
    optimizer,
    criterion,
    epochs,
    device,
    head_key,
    label_task_number,
    global_labels,
):
    model.train()
    model.backbone.eval()
    epoch_bar = tqdm(range(epochs), desc="Epochs", unit="epoch", miniters=20)

    for _ in epoch_bar:
        batch_bar = tqdm(feature_loader, leave=False, unit="batch", disable=True)
        for feat_cpu, y_global_cpu in batch_bar:
            feat = feat_cpu.to(device, non_blocking=True)
            y_global = y_global_cpu.to(device, non_blocking=True)

            if hasattr(criterion, "x_cache"):
                criterion.x_cache = feat

            optimizer.zero_grad(set_to_none=True)
            pred = model.heads[str(head_key)](feat)
            y_local = _to_local_labels(y_global, label_task_number, pred.size(1), global_labels=global_labels)
            loss = criterion(pred, y_local)

            loss.backward()
            optimizer.step()

            batch_bar.set_postfix({"loss": float(loss.item())})

        epoch_bar.set_postfix({"loss": float(loss.item())})


def _prepare_task_mode(model, mode, task):
    if mode == "til":
        model.add_head(task)
        train_task_number = task
    elif mode == "cil":
        if task == 0:
            model.add_head(task)
        else:
            model.expand_head()
        train_task_number = 0
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Expected 'til' or 'cil'.")

    return train_task_number


def avg_accuracy_on_all_tasks(model, dataloaders, task_number, cil=False, test=False):
    data_idx = 2 if test else 1
    accs = []

    for i in range(task_number + 1):
        eval_data = dataloaders[i][data_idx]
        eval_task_number = 0 if cil else i
        acc = accuracy(model, eval_data, task_number=eval_task_number, global_labels=cil)
        accs.append(acc)
        print(f"|{i}: {acc:.4f}|", end="")

    avg_acc = sum(accs) / len(accs)
    print(f"\nAvg accuracy: {avg_acc:.4f}")


def _run_feature_experiment(
    model,
    criterion,
    title,
    mode,
    batch_size=512,
    epochs=20,
    seed=42,
    lr=1e-4,
    num_workers=0,
):
    deterministic(seed)
    _freeze_backbone(model)
    device = next(model.parameters()).device

    dataloaders = get_data_loaders(batch_size=batch_size, num_workers=num_workers)

    for task in range(len(dataloaders)):
        print(f"Training task {task}")
        train_task_number = _prepare_task_mode(model, mode=mode, task=task)
        model.to(device)
        train_data = dataloaders[task][0]

        feat_train_cpu, y_train_global_cpu = extract_features_from_loader(
            backbone=model.backbone,
            loader=train_data,
            device=device,
        )
        feature_loader = _build_feature_loader(
            feat_cpu=feat_train_cpu,
            y_cpu=y_train_global_cpu,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        optimizer = torch.optim.Adam(model.heads.parameters(), lr=lr)
        _train_on_features(
            model=model,
            feature_loader=feature_loader,
            optimizer=optimizer,
            criterion=criterion,
            epochs=epochs,
            device=device,
            head_key=train_task_number,
            label_task_number=train_task_number,
            global_labels=(mode == "cil"),
        )

        _maybe_update_criterion(criterion, dataloaders[task][0], task_number=train_task_number)

        avg_accuracy_on_all_tasks(model, dataloaders, task, cil=(mode == "cil"))

    avg_accuracy_on_all_tasks(model, dataloaders, task, cil=(mode == "cil"), test=True)

    model.save(f"../models/weights/{title}_{mode.upper()}.pth")


def train(model, dataloader, optimizer, criterion, title, epochs, task_number, save=True, global_labels=False):
    device = next(model.parameters()).device

    if title is not None:
        writer = SummaryWriter(log_dir=f"../runs/{title}")

    epoch_bar = tqdm(range(epochs), desc="Epochs", unit="epoch", miniters=50)

    for epoch in epoch_bar:
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, unit="batch", disable=True)
        for i, (x, y) in enumerate(batch_bar):
            x_all, y_all = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if hasattr(criterion, "x_cache"):
                criterion.x_cache = x_all

            optimizer.zero_grad()
            pred = model(x_all, task_number)
            y_local = _to_local_labels(y_all, task_number, pred.size(1), global_labels)
            loss = criterion(pred, y_local)

            if title is not None:
                writer.add_scalar("Loss/Train", loss.item(), epoch * len(dataloader) + i)

            loss.backward()
            optimizer.step()

            batch_bar.set_postfix({"loss": loss.item()})

        epoch_bar.set_postfix({"loss": loss.item()})

    if title is not None:
        writer.close()

    if save:
        model.save(f"../models/weights/{title}.pth")


def new_model_from_backbone(device, path="../models/weights/backbone.pth"):
    backbone = BackBone()
    backbone.load_state_dict(torch.load(path, weights_only=True))
    model = MultiHeadModel(backbone)
    model.to(device)
    return model


def run_til_experiment(model, criterion, title, batch_size=512, epochs=20, seed=42, lr=1e-4, num_workers=0):
    _run_feature_experiment(
        model=model,
        criterion=criterion,
        title=title,
        mode="til",
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        lr=lr,
        num_workers=num_workers,
    )


def run_cil_experiment(model, criterion, title, batch_size=512, epochs=20, seed=42, lr=1e-4, num_workers=0):
    _run_feature_experiment(
        model=model,
        criterion=criterion,
        title=title,
        mode="cil",
        batch_size=batch_size,
        epochs=epochs,
        seed=seed,
        lr=lr,
        num_workers=num_workers,
    )

def backbone_train(model, dataloader, optimizer, criterion, title, tsne, epochs=5):
    model.train()
    device = next(model.parameters()).device
    writer = SummaryWriter(log_dir=f"../runs/{title}")

    epoch_bar = tqdm(range(epochs), desc="Epochs", unit="epoch", miniters=50)
    for epoch in epoch_bar:
        batch_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", leave=False, unit="batch", disable=True)
        for i, (x, y) in enumerate(batch_bar):
            x_all, y_all = x.to(device, non_blocking=True), y.to(device, non_blocking=True)

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
        # for x, y in tqdm(val_data, desc="Evaluating", unit="batch"):
        for x, y in val_data:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(x, task_number)

            y_local = _to_local_labels(y, task_number, pred.size(1), global_labels)
            _, predicted = torch.max(pred.data, 1)
            total += y_local.size(0)
            correct += (predicted == y_local).sum().item()

        model.train()
        return correct / total



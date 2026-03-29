import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from torch.utils.tensorboard import SummaryWriter

def deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True

def train(model, dataloader, optimizer, criterion, title, epochs, task_number):
    device = next(model.parameters()).device
    writer = SummaryWriter(log_dir=f"../runs/{title}")
    for epoch in tqdm(range(epochs), desc = "Epochs", unit = "epoch"):
        batch_bar = tqdm(dataloader, desc = f"Epoch {epoch+1}/{epochs}", leave=False, unit="batch")
        for i, (x, y) in enumerate(batch_bar):
            x_all, y_all = x.to(device), y.to(device) # ver randaugment

            optimizer.zero_grad()
            pred = model(x_all, task_number)
            loss = criterion(pred, y_all)
            writer.add_scalar("Loss/Train", loss.item(), epoch * len(dataloader) + i)
            print(loss.item())

            loss.backward()
            optimizer.step()

            batch_bar.set_postfix({"loss": loss.item()})

    writer.close()
    model.save(f"../models/weights/{title}.pth")


def backbone_train(model, dataloader, optimizer, criterion, title, tsne, epochs=5):
    device = next(model.parameters()).device
    writer = SummaryWriter(log_dir=f"../runs/{title}")

    for epoch in tqdm(range(epochs), desc = "Epochs", unit = "epoch"):
        batch_bar = tqdm(dataloader, desc = f"Epoch {epoch+1}/{epochs}", leave=False, unit="batch")
        for i, (x, y) in enumerate(batch_bar):
            x_all, y_all = x.to(device), y.to(device) # ver randaugment

            optimizer.zero_grad()
            embeddings = model(x_all)
            loss = criterion(embeddings, y_all)

            if epoch==0 or epoch==epochs//2 or epoch==epochs-1:
                emb_2d = tsne.fit_transform(embeddings.detach().cpu().numpy())
                plt.scatter(emb_2d[:, 0], emb_2d[:, 1], c=y_all.detach().cpu().numpy(), cmap='tab10')
                plt.colorbar()
                plt.title(f"Epoch {epoch+1}")
                plt.show()

            writer.add_scalar("Loss/Train", loss.item(), epoch * len(dataloader) + i)
            print(loss.item())

            loss.backward()
            optimizer.step()

            batch_bar.set_postfix({"loss": loss.item()})

    writer.close()
    model.save(f"../models/weights/{title}.pth")

def accuracy(model, dataloader, task_number):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for x, y in tqdm(task0_val, desc="Evaluating", unit="batch"):
            x, y = x.to(device), y.to(device)
            pred = model(x, task_number)
            _, predicted = torch.max(pred.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        return correct / total
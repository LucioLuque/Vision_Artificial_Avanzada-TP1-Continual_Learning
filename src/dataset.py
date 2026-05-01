import numpy as np
import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_task_data(dataset, task_classes):
    targets = np.array(dataset.targets)
    indices = np.where(np.isin(targets, task_classes))[0]
    return Subset(dataset, indices)

def _resolve_loader_runtime(num_workers=None, pin_memory=None, persistent_workers=None):
    if num_workers is None:
        cpu_count = os.cpu_count() or 2
        if os.name == "nt":
            num_workers = min(2, max(1, cpu_count // 4))
        else:
            num_workers = min(4, max(2, cpu_count // 2))

    if os.name == "nt":
        num_workers = min(num_workers, 4)

    num_workers = max(0, int(num_workers))

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    if persistent_workers is None:
        persistent_workers = (num_workers > 0) and (os.name != "nt")

    return num_workers, pin_memory, persistent_workers


def get_split_loaders(subset_train, subset_test, val_size, batch_size, num_workers=None, pin_memory=None, persistent_workers=None):
    n = len(subset_train)
    indices = np.random.permutation(n)
    split = int(np.floor(val_size * n))

    num_workers, pin_memory, persistent_workers = _resolve_loader_runtime(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers

    train_loader = DataLoader(Subset(subset_train, indices[split:]), shuffle=True, **loader_kwargs)
    val_loader = DataLoader(Subset(subset_train, indices[:split]), shuffle=False, **loader_kwargs)
    test_loader = DataLoader(subset_test, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader

def get_task_data_loaders(tasks, train_dataset, test_dataset, val_size, batch_size, num_workers=None, pin_memory=None, persistent_workers=None):
    dataloaders = []
    for task_classes in tasks:
        train_subset = get_task_data(train_dataset, task_classes)
        test_subset  = get_task_data(test_dataset,  task_classes)
        loaders = get_split_loaders(
            train_subset,
            test_subset,
            val_size,
            batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )
        dataloaders.append(loaders)
        train_l, val_l, test_l = loaders
        print(f"Task {task_classes}: Train={len(train_l.dataset)}, Val={len(val_l.dataset)}, Test={len(test_l.dataset)}")
    return dataloaders

def get_data_loaders(val_size=0.1, batch_size=64, num_workers=0, pin_memory=None, persistent_workers=None):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    TASKS = [[0,1], [2,3], [4,5], [6,7], [8,9]]

    train_dataset = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return get_task_data_loaders(
        TASKS,
        train_dataset,
        test_dataset,
        val_size,
        batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
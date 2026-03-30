import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

def get_task_data(dataset, task_classes):
    targets = np.array(dataset.targets)
    indices = np.where(np.isin(targets, task_classes))[0]
    return Subset(dataset, indices)

def get_split_loaders(subset_train, subset_test, val_size, batch_size):
    n = len(subset_train)
    indices = np.random.permutation(n)  # cleaner than shuffle in-place
    split = int(np.floor(val_size * n))

    train_loader = DataLoader(Subset(subset_train, indices[split:]), batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True)
    val_loader   = DataLoader(Subset(subset_train, indices[:split]),  batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader  = DataLoader(subset_test,                            batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    return train_loader, val_loader, test_loader

def get_task_data_loaders(tasks, train_dataset, test_dataset, val_size, batch_size):
    dataloaders = []
    for task_classes in tasks:
        train_subset = get_task_data(train_dataset, task_classes)
        test_subset  = get_task_data(test_dataset,  task_classes)
        loaders = get_split_loaders(train_subset, test_subset, val_size, batch_size)
        dataloaders.append(loaders)
        train_l, val_l, test_l = loaders
        print(f"Task {task_classes}: Train={len(train_l.dataset)}, Val={len(val_l.dataset)}, Test={len(test_l.dataset)}")
    return dataloaders

def get_data_loaders(val_size=0.1, batch_size=64):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010))
    ])

    TASKS = [[0,1], [2,3], [4,5], [6,7], [8,9]]

    train_dataset = datasets.CIFAR10(root='./data', train=True,  download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    return get_task_data_loaders(TASKS, train_dataset, test_dataset, val_size, batch_size)
import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def get_task_data(dataset, task_classes):
    targets = np.array(dataset.targets)
    mask = np.isin(targets, task_classes)
    indices = np.where(mask)[0]
    return Subset(dataset, indices)

def get_split_loaders(subset_train, subset_test, val_size, batch_size):
    num_samples = len(subset_train)
    indices = list(range(num_samples))
    np.random.shuffle(indices)
    
    split = int(np.floor(val_size * num_samples))
    train_indices, val_indices = indices[split:], indices[:split]
    
    train_subset = Subset(subset_train, train_indices)
    val_subset = Subset(subset_train, val_indices)

    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(subset_test, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

def get_task_data_loaders(tasks, train_dataset, test_dataset, val_size, batch_size):
    dataloaders = []
    for task_classes in tasks:
        train_subset = get_task_data(train_dataset, task_classes)
        test_subset  = get_task_data(test_dataset, task_classes)
        
        train_loader, val_loader, test_loader = get_split_loaders(train_subset, test_subset, val_size, batch_size)
        
        dataloaders.append((train_loader, val_loader, test_loader))
        print(f"Task {task_classes}: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}, Test={len(test_loader.dataset)}")
    return dataloaders


def get_data_loaders(val_size=0.1, batch_size=64):
    transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), 
                         (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = datasets.CIFAR10(root='./data', train=True, 
                                    download=True, transform=transform)
    test_dataset  = datasets.CIFAR10(root='./data', train=False, 
                                    download=True, transform=transform)
    
    TASKS = [
        [0, 1],   # airplane, automobile
        [2, 3],   # bird, cat
        [4, 5],   # deer, dog
        [6, 7],   # frog, horse
        [8, 9],   # ship, truck
    ]

    return get_task_data_loaders(TASKS, train_dataset, test_dataset, val_size, batch_size)
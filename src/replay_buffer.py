import torch
from torch.utils.data import DataLoader, TensorDataset, Subset
import numpy as np

class ReplayBuffer:
    def __init__(self, size):
        self.size = size
        self.buffer = []
        self.seen = 0

    def update(self, dataloader):
        for x_batch, y_batch in dataloader:
            for x, y in zip(x_batch, y_batch):
                self.seen += 1
                if len(self.buffer) < self.size:
                    self.buffer.append((x, y))
                else:
                    idx = np.random.randint(0, self.seen)
                    if idx < self.size:
                        self.buffer[idx]= (x,y)

    def get_dataloader(self, batch_size):
        xs, ys = zip(*self.buffer)
        dataset = TensorDataset(torch.stack(xs), torch.stack(ys))
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
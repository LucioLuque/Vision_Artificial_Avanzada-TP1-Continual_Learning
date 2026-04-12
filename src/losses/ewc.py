import torch

class EWCCriterion:
    def __init__(self, criterion, lambda_=1000, global_labels=False, model=None):
        self.criterion = criterion
        self.lambda_ = lambda_
        self.fishers = []
        self.thetas = []
        self.global_labels = global_labels
        self.model = model

    def __call__(self, pred, y):
        loss = self.criterion(pred, y)
        current_parameters = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        for fisher, theta_star in zip(self.fishers, self.thetas):
            for n, p in current_parameters.items():
                if n not in fisher:
                    continue

                p_curr = p
                p_old = theta_star[n].to(p.device, non_blocking=True)
                f = fisher[n].to(p.device, non_blocking=True)

                if p_curr.shape != p_old.shape:
                    n_old = p_old.shape[0]
                    p_curr = p_curr[:n_old]   # weight: (new_classes, feat) → (old_classes, feat)
                                              # bias:   (new_classes,)      → (old_classes,)

                penalty = self.lambda_ * (f * (p_curr - p_old) ** 2).sum()
                loss += penalty

        return loss

    def update(self, dataloader, task_number):
        device = next(self.model.parameters()).device
        current_parameters = {n: p for n, p in self.model.named_parameters() if p.requires_grad}

        theta_star = {n: p.clone().detach() for n, p in current_parameters.items()}
        fisher     = {n: torch.zeros_like(p) for n, p in current_parameters.items()}

        self.model.eval()
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            self.model.zero_grad()
            pred = self.model(x, task_number)
            y_local = _to_local_labels(y, task_number, pred.size(1), self.global_labels)
            loss = self.criterion(pred, y_local)
            loss.backward()
            for n, p in current_parameters.items():
                if p.grad is not None:
                    fisher[n] += p.grad ** 2 / len(dataloader)

        self.fishers.append(fisher)
        self.thetas.append(theta_star)
        self.model.train()

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
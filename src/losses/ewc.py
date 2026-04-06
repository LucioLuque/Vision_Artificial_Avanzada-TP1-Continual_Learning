import torch

class EWCCriterion:
    def __init__(self, criterion, lambda_=1000, global_labels=False):
        self.criterion = criterion
        self.lambda_ = lambda_
        self.fishers = []
        self.thetas = []
        self.head_sizes = []  # <-- nuevo: tamaño de cabeza al consolidar
        self.latest_parameters = {}  # <-- vacío, el loop EWC no itera
        self.global_labels = global_labels

    def __call__(self, pred, y):
        loss = self.criterion(pred, y)
        base_loss = loss.item()
        ewc_penalty = 0.0
        current_params = {n: p for n, p in self.latest_parameters.items() if p.requires_grad}

        for fisher, theta_star, head_size in zip(self.fishers, self.thetas, self.head_sizes):
            for n, p in current_params.items():
                if n not in fisher:
                    continue

                p_curr = p
                p_old  = theta_star[n]
                f      = fisher[n]

                # Si es un parámetro de la cabeza, recortamos a las clases que existían
                if p_curr.shape != p_old.shape:
                    n_old = p_old.shape[0]
                    p_curr = p_curr[:n_old]   # weight: (new_classes, feat) → (old_classes, feat)
                                              # bias:   (new_classes,)      → (old_classes,)

                penalty = self.lambda_ * (f * (p_curr - p_old) ** 2).sum()
                ewc_penalty += penalty.item()
                loss += penalty

        return loss

    def update(self, model, dataloader, task_number):
        device = next(model.parameters()).device
        self.latest_parameters = {n: p for n, p in model.named_parameters() if p.requires_grad}

        theta_star = {n: p.clone().detach() for n, p in self.latest_parameters.items()}
        fisher     = {n: torch.zeros_like(p) for n, p in self.latest_parameters.items()}

        # Guardamos el tamaño actual de la cabeza
        head_out = model.heads["0"].out_features
        self.head_sizes.append(head_out)

        model.eval()
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            model.zero_grad()
            pred = model(x, task_number)
            y_local = _to_local_labels(y, task_number, pred.size(1), self.global_labels)
            loss = self.criterion(pred, y_local)
            loss.backward()
            for n, p in self.latest_parameters.items():
                if p.grad is not None:
                    fisher[n] += p.grad ** 2 / len(dataloader)

        self.fishers.append(fisher)
        self.thetas.append(theta_star)
        model.train()

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
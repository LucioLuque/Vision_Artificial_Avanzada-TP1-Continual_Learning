import torch
import torch.nn as nn
import torch.nn.functional as F

class LwFCriterion:
    def __init__(self, criterion, lambda_=1.0, temperature=2.0):
        self.criterion = criterion       # CrossEntropyLoss
        self.lambda_ = lambda_
        self.temperature = temperature
        self.teacher = None              # copia del modelo anterior
        self.n_old_classes = 0           # cuántas clases tenía la cabeza antes

    def update_teacher(self, model):
        """Llamar ANTES de expand_head() y entrenar la nueva task"""
        import copy
        self.teacher = copy.deepcopy(model)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False
        self.n_old_classes = model.heads["0"].out_features

    def __call__(self, pred, y, x=None):
        # loss tarea nueva (todas las clases)
        loss = self.criterion(pred, y)

        # si no hay teacher todavía (Task 0), no hay distilación
        if self.teacher is None or x is None:
            return loss

        # predicciones del profesor sobre las clases viejas
        device = pred.device
        with torch.no_grad():
            teacher_logits = self.teacher(x, 0)  # shape: (batch, n_old_classes)

        # logits del modelo nuevo, solo las clases viejas
        student_logits = pred[:, :self.n_old_classes]  # shape: (batch, n_old_classes)

        # KL divergence con temperatura
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits  / self.temperature, dim=1)
        distillation_loss = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        distillation_loss *= self.temperature ** 2  # reescalar por T²

        loss += self.lambda_ * distillation_loss
        return loss
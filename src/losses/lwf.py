import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class LwFCriterion:
    def __init__(self, criterion, model, lambda_=1.0, temperature=2.0, cil_binary_distill=True):
        self.criterion = criterion
        self.lambda_ = lambda_
        self.temperature = temperature
        self.cil_binary_distill = cil_binary_distill
        self.student = model # Se guarda una referencia que se actualiza sola en vez de una copia
        self.teacher = None
        self.x_cache = None

    def _is_feature_batch(self, x):
        return (
            x is not None
            and torch.is_tensor(x)
            and x.ndim == 2
            and x.size(1) == self.student.num_features
        )

    def _forward_logits(self, model, x, task_number):
        if self._is_feature_batch(x):
            return model.heads[str(task_number)](x)
        return model(x, task_number)

    def update(self, dataloader=None, task_number=None):
        #solo para pder generalizar con ewc, pero no se usa model, dataloader ni task_number
        self.teacher = copy.deepcopy(self.student) # Teacher es el modelo anterior que se usa de referencia
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def __call__(self, pred, y):
        loss = self.criterion(pred, y)

        if self.teacher is None or self.x_cache is None: # Primera llamada
            return loss

        is_cil = (len(self.student.heads) == 1) # Si solo hay una cabeza para la segunda llamada, es CIL
        kd_loss = 0.0
        if is_cil:
            with torch.no_grad():
                teacher_logits = self._forward_logits(self.teacher, self.x_cache, 0)
            student_logits = pred[:, :teacher_logits.size(1)]  # old class columns

            if self.cil_binary_distill:
                kd_loss += self._calc_binary_distill(student_logits, teacher_logits)
            else:
                kd_loss += self._calc_kl(student_logits, teacher_logits)
        else:
            for old_task_id in range(len(self.teacher.heads)):
                with torch.no_grad():
                    teacher_logits = self._forward_logits(self.teacher, self.x_cache, old_task_id)
                student_logits = self._forward_logits(self.student, self.x_cache, old_task_id)

                kd_loss += self._calc_kl(student_logits, teacher_logits)

        return loss + kd_loss

    def _calc_kl(self, student_logits, teacher_logits):
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distill = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        distill *= self.temperature ** 2
        return self.lambda_ * distill

    def _calc_binary_distill(self, student_logits, teacher_logits):
        teacher_probs = torch.sigmoid(teacher_logits / self.temperature)
        distill = F.binary_cross_entropy_with_logits(
            student_logits / self.temperature,
            teacher_probs,
            reduction="mean",
        )
        distill *= self.temperature ** 2
        return self.lambda_ * distill

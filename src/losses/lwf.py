import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class LwFCriterion:
    def __init__(self, criterion, model, lambda_=1.0, temperature=2.0):
        self.criterion = criterion
        self.lambda_ = lambda_
        self.temperature = temperature
        self.student = model # Se guarda una referencia que se actualiza sola en vez de una copia
        self.teacher = None
        self.x_cache = None

    def update(self, model=None, dataloader=None, task_number=None):
        #solo para pder generalizar con ewc, pero no se usa model, dataloader ni task_number
        self.teacher = copy.deepcopy(self.student)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def __call__(self, pred, y):
        loss = self.criterion(pred, y)

        if self.teacher is None or self.x_cache is None: # Primera llamada
            return loss

        is_cil = (len(self.student.heads) == 1) # Si solo hay una cabeza para la segunda llamada, es CIL

        if is_cil:
            with torch.no_grad():
                teacher_logits = self.teacher(self.x_cache, 0)
            student_logits = pred[:, :teacher_logits.size(1)]  # old class columns

            loss += self._calc_kl(student_logits, teacher_logits)
        else:
            for old_task_id in range(len(self.teacher.heads)):
                with torch.no_grad():
                    teacher_logits = self.teacher(self.x_cache, old_task_id)
                student_logits = self.student(self.x_cache, old_task_id)

                loss += self._calc_kl(student_logits, teacher_logits)

        return loss

    def _calc_kl(self, student_logits, teacher_logits):
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        distill = F.kl_div(student_soft, teacher_soft, reduction="batchmean")
        distill *= self.temperature ** 2
        return self.lambda_ * distill

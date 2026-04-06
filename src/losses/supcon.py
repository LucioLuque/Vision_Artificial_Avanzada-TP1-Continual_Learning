import torch
import torch.nn.functional as F

class SupConLoss(torch.nn.Module):
    def __init__(self, tau=0.07):
        super(SupConLoss, self).__init__()
        self.tau = tau

    def forward(self, embeddings, labels):
        device = embeddings.device
    
        embeddings = F.normalize(embeddings, dim=1)
        N = embeddings.shape[0]
        
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.tau  # Todas las multiplicaciones vectoriales de una
        self_mask = torch.eye(N, dtype=torch.bool).to(device)
        sim_matrix = sim_matrix.masked_fill_(self_mask, -1e9) # Al hacer exp se descartan los términos donde i==j

        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T) & ~self_mask # Queda True si i,j comparten label

        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
        
        P_i = pos_mask.sum(dim=1)
        if (P_i == 0).any():
            print("Hay samples sin positivos")
        valid = P_i > 0
        loss = -(log_prob * pos_mask).sum(dim=1)[valid] / P_i[valid]
        
        return loss.mean()

    def update(self, model=None, dataloader=None, task_number=None): #seguro se saca!
        pass # No hay nada que actualizar en SupConLoss, pero se define el método para compatibilidad con el loop de entrenamiento

        

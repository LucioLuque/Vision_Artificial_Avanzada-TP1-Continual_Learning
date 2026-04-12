import torch
import torch.nn.functional as F

class SupConLoss(torch.nn.Module):
    def __init__(self, tau=0.07):
        super(SupConLoss, self).__init__()
        self.tau = tau

    def forward(self, embeddings, labels, anchor_mask=None):
        device = embeddings.device
    
        embeddings = F.normalize(embeddings, dim=1)
        N = embeddings.shape[0]
        
        sim_matrix = torch.matmul(embeddings, embeddings.T) / self.tau
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim_matrix = sim_matrix.masked_fill(self_mask, -1e9)

        labels = labels.unsqueeze(1)
        pos_mask = (labels == labels.T) & ~self_mask

        log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)

        if anchor_mask is None:
            anchor_mask = torch.ones(N, dtype=torch.bool, device=device)

        P_i = pos_mask.sum(dim=1)
        valid = anchor_mask & (P_i > 0)

        loss = -(log_prob * pos_mask).sum(dim=1)[valid] / P_i[valid]
        
        return loss.mean()
import torch
import torch.nn.functional as F


# def l_supcon(embeddings, labels, tau, device):

#     # count number of samples per class
#     unique_labels, counts = torch.unique(labels, return_counts=True)
#     class_counts = dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))

#     idx = torch.arange(len(labels)).to(device)

#     labels_masks = {label.item(): (labels == label).to(device) for label in unique_labels}

#     loss = 0.0
#     for (i, sample) in enumerate(embeddings):
#         suma = 0.0
#         pos_label = labels[i].item()
#         neg_labels = [label for label in unique_labels if label.item() != pos_label]
#         for idx in labels_masks[pos_label]:
#             if idx != i:
#                 numerador = torch.cosine_similarity(sample.unsqueeze(0), batch['samples'][idx].unsqueeze(0))
#                 numerador = torch.exp(numerador / tau)
#                 denominador = 0.0
#                 for neg_label in neg_labels:
#                     for idx_neg in labels_masks[neg_label]:
#                         denominador += torch.exp(torch.cosine_similarity(sample.unsqueeze(0), batch['samples'][idx_neg].unsqueeze(0)) / tau)
#                 suma += torch.log(numerador / denominador)
#         loss -= suma / class_counts[pos_label]

#     return loss

def l_supcon(embeddings, labels, tau, device): # capaz el device se debería definir dentro de la función?
    embeddings = F.normalize(embeddings, dim=1)
    N = embeddings.shape[0]
    
    sim_matrix = torch.matmul(embeddings, embeddings.T) / tau  # Todas las multiplicaciones vectoriales de una
    self_mask = torch.eye(N, dtype=torch.bool).to(device)
    sim_matrix.masked_fill_(self_mask, float('-inf')) # Al hacer exp se descartan los términos donde i==j

    labels = labels.unsqueeze(1)
    pos_mask = (labels == labels.T) & ~self_mask # Queda True si i,j comparten label

    log_prob = sim_matrix - torch.logsumexp(sim_matrix, dim=1, keepdim=True)
    
    P_i = pos_mask.sum(dim=1)
    loss = -(log_prob * pos_mask).sum(dim=1) / P_i
    
    return loss.mean() # Según la ecuación debería ser sum(), pero mean() es mejor para que no dependa del batch size

        

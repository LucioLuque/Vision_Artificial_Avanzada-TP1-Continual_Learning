import torch

def l_supcon(embeddings, labels, tau, device):

    # count number of samples per class
    unique_labels, counts = torch.unique(labels, return_counts=True)
    class_counts = dict(zip(unique_labels.cpu().numpy(), counts.cpu().numpy()))

    idx = torch.arange(len(labels)).to(device)

    labels_masks = {label.item(): (labels == label).to(device) for label in unique_labels}

    loss = 0.0
    for (i, sample) in enumerate(embeddings):
        suma = 0.0
        pos_label = labels[i].item()
        neg_labels = [label for label in unique_labels if label.item() != pos_label]
        for idx in labels_masks[pos_label]:
            if idx != i:
                numerador = torch.cosine_similarity(sample.unsqueeze(0), batch['samples'][idx].unsqueeze(0))
                numerador = torch.exp(numerador / tau)
                denominador = 0.0
                for neg_label in neg_labels:
                    for idx_neg in labels_masks[neg_label]:
                        denominador += torch.exp(torch.cosine_similarity(sample.unsqueeze(0), batch['samples'][idx_neg].unsqueeze(0)) / tau)
                suma += torch.log(numerador / denominador)
        loss -= suma / class_counts[pos_label]

    return loss

                


        

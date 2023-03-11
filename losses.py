import torch
    
def implicit_score_matching(positions, scores, batch_size):
    dims = (batch_size, -1)
    loss1 = torch.norm(scores.view(dims), dim=-1)**2 / 2.
    loss2 = torch.zeros(batch_size, device=positions.device)

    for i in range(positions.view(dims).shape[-1]):
        d_scores = torch.autograd.grad(scores.view(dims)[:,i].sum(),
                                        positions,
                                        create_graph=True,
                                        )[0].view(dims)[:,i]
        loss2 += d_scores

    loss = loss1 + loss2
    return loss.mean(), loss1.mean(), loss2.mean()

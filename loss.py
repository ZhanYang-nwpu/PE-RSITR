import torch
import torch.nn as nn


def pdist(x1, x2):
    """
        compute euclidean distance between two tensors
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise euclidean distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_square = torch.sum(x1*x1, 1).view(-1, 1)
    x2_square = torch.sum(x2*x2, 1).view(1, -1)
    return torch.sqrt(x1_square - 2 * torch.mm(x1, x2.transpose(0, 1)) + x2_square + 1e-4)


def pdist_cos(x1, x2):
    """
        compute cosine similarity between two tensors
        x1: Tensor of shape (h1, w)
        x2: Tensor of shape (h2, w)
        Return pairwise cosine distance for each row vector in x1, x2 as
        a Tensor of shape (h1, h2)
    """
    x1_norm = x1 / x1.norm(dim=1)[:, None]
    x2_norm = x2 / x2.norm(dim=1)[:, None]
    score = torch.mm(x1_norm, x2_norm.transpose(0, 1))
    mask = torch.isnan(score)
    score[mask] = 0
    return score


class ContrastiveLoss(nn.Module):
    """
    Compute contrastive loss
    """
    def __init__(self, opt, margin = 0, max_violation = False):
        super(ContrastiveLoss, self).__init__()
        self.opt = opt
        self.margin = margin
        self.max_violation = max_violation

        self.sim = pdist_cos
        self.alpha = 0.5

    def forward(self, im, s):
        scores = self.sim(im, s)

        diagonal = scores.diag().view(im.size(0), 1)
        d1 = diagonal.expand_as(scores)
        d2 = diagonal.t().expand_as(scores)

        # compare every diagonal score to scores in its column
        # caption retrieval
        cost_s = (self.margin + scores - d1).clamp(min=0)
        # compare every diagonal score to scores in its row
        # image retrieval
        cost_im = (self.margin + scores - d2).clamp(min=0)

        # clear diagonals
        I = torch.eye(scores.size(0)) > .5
        if torch.cuda.is_available():
            I = I.cuda()
        cost_s = cost_s.masked_fill_(I, 0)
        cost_im = cost_im.masked_fill_(I, 0)

        # keep the maximum violating negative for each query
        if self.max_violation:
            cost_s = cost_s.max(1)[0]
            cost_im = cost_im.max(0)[0]

        return cost_s.sum() + cost_im.sum()

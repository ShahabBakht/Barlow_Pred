import torch
import torch.nn as nn

class BarlowTwinsLoss(nn.Module):
    def __init__(self, lambd = 5e-3):
        super(BarlowTwinsLoss, self).__init__()
        
        self.lambd = lambd
        
        
    def forward(self, pred, gt):
        
        B = pred.shape[0]
        
        pred = pred.transpose(0,1)
        c = torch.matmul(pred, gt) / B
        
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.lambd * off_diag
        
        
        return loss, on_diag, off_diag

def off_diagonal(x):
# return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
import torch 
import torch.nn as nn

class ResidualMSELoss(nn.Module):
    """MSE loss for DnCNN residual learning."""
    def __init__(self):
        super(ResidualMSELoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred_residual, noisy, clean):
        true_residual = noisy - clean
        return 0.5 * self.mse(pred_residual, true_residual)
import torch
import torch.nn as nn

class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, fake_logits):
        fake_targets = torch.ones_like(fake_logits)
        gen_loss = self.criterion(fake_logits, fake_targets)
        return gen_loss.item()
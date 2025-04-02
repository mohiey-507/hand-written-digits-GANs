import torch
import torch.nn as nn

class GenLoss(nn.Module):
    def __init__(self):
        super(GenLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, fake_logits):
        fake_targets = torch.ones_like(fake_logits)
        gen_loss = self.criterion(fake_logits, fake_targets)
        return gen_loss

class DiscLoss(nn.Module):
    def __init__(self):
        super(DiscLoss, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss()

    def forward(self, real_logits, fake_logits):
        fake_targets = torch.zeros_like(fake_logits)
        real_targets = torch.ones_like(real_logits)

        fake_loss = self.criterion(fake_logits, fake_targets)
        real_loss = self.criterion(real_logits, real_targets)

        disc_loss = (real_loss + fake_loss) / 2
        return disc_loss
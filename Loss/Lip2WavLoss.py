import torch
from torch import nn


class Tacotron2Loss(nn.Module):
    def __init__(self, hp):
        super(Tacotron2Loss, self).__init__()
        self.hp = hp

    def forward(self, model_output, targets, iteration):
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        slice = torch.arange(0, gate_target.size(1), self.hp.n_frames_per_step)
        gate_target = gate_target[:, slice].view(-1, 1)

        mel_out, mel_out_postnet, gate_out, _ = model_output
        gate_out = gate_out.view(-1, 1)
        p = self.hp.p
        mel_loss = nn.MSELoss()(p * mel_out, p * mel_target)
        mel_loss_post = nn.MSELoss()(p * mel_out_postnet, p * mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        # added
        l1_loss = nn.L1Loss()(mel_target, mel_out)
        return mel_loss, mel_loss_post, l1_loss, gate_loss  # , ((mel_loss+mel_loss_post)/(p**2)+gate_loss+l1_loss).item()


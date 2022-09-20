import torch
from torch import nn


class BaseLossFunction(nn.Module):
    def __init__(self, hp, logger):
        super(BaseLossFunction, self).__init__()
        self.hp = hp
        self.logger = logger
        self.items = []
        self.last_loss = 0

    def forward(self, model_outputs, targets):
        pass

    def log_loss(self, epoch_idx):
        spec_loss, spec_loss_post, l1_loss, mask_loss = self.items

        self.logger.info(
            "spec_loss, spec_loss_post, l1_loss = {:.3f}, {:.3f}, {:.3f}".format(spec_loss, spec_loss_post, l1_loss))
        self.logger.info("mask_loss = {:.3f}".format(mask_loss))
        self.logger.add_scalars("train.MSELoss", {"specLoss": spec_loss,
                                                  "postLoss": spec_loss_post}, epoch_idx)
        self.logger.add_scalar("train.Loss", l1_loss, epoch_idx)
        self.logger.add_scalar("train.mask_loss", mask_loss, epoch_idx)

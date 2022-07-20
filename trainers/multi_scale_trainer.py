import os
import math
import importlib
from tqdm import tqdm

import torch
from torch import autograd

from loss.bce_loss  import BCELoss
from util.trainer import accumulate
from trainers.base import BaseTrainer

class Trainer(BaseTrainer):
    def __init__(self, opt, net_G, opt_G, sch_G,
                 train_data_loader, val_data_loader=None):
        super(Trainer, self).__init__(opt, net_G, opt_G,
                                      sch_G,
                                      train_data_loader, val_data_loader)
        self.accum = 0.5 ** (32 / (10 * 1000))
        print(self.net_G)

    def _init_loss(self, opt):
        self._assign_criteria(
            'bce',
            BCELoss(),
            opt.trainer.loss_weight.weight_bce)

    def _assign_criteria(self, name, criterion, weight):
        self.criteria[name] = criterion
        self.weights[name] = weight

    def optimize_parameters(self, data):
        self.gen_losses = {}

        gt_point_cloud, input_point_cloud = data['gt_point_cloud'], data['input_point_cloud']

        predict = self.net_G(input_point_cloud)
        self.gen_losses["bce"] = self.criteria['bce'](predict, gt_point_cloud)

        total_loss = 0
        for key in self.gen_losses:
            self.gen_losses[key] = self.gen_losses[key] * self.weights[key]
            total_loss += self.gen_losses[key]

        self.gen_losses['total_loss'] = total_loss

        self.net_G.zero_grad()
        total_loss.backward()
        self.opt_G.step()

        accumulate(self.net_G_ema, self.net_G_module, self.accum)

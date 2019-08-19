import torch
import torch.optim as optim
from model import *
import math

class Scheduler:
    def __init__(self, scheduler, step_criterion=None, step_log=None):
        self.scheduler = scheduler
        self.step_criterion = step_criterion
        self.step_log = step_log

    def step(self, train_log, val_log):
        if self.step_log == "val":
            self.scheduler.step(val_log[self.step_criterion])
        elif self.step_log == "train":
            self.scheduler.step(train_log[self.step_criterion])
        elif self.step_log == None:
            self.scheduler.step()

    def state_dict(self):
        return {"state_dict": self.scheduler.state_dict(),
                "step_criterion": self.step_criterion,
                "step_log": self.step_log}

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict["state_dict"])
        self.step_criterion = state_dict["step_criterion"]
        self.step_log = state_dict["step_log"]
        return self


class SWAScheduler(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, const_epochs, const_lr, freq, max_lr, min_lr, last_epoch=-1):
        self.const_epochs = const_epochs
        self.const_lr = const_lr
        self.freq = freq
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(SWAScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.const_epochs:
            self.last_lr = self.const_lr
            return [self.const_lr]*len(self.base_lrs)
        else:
            delta_epochs = (self.last_epoch - self.const_epochs) % self.freq
            delta = (self.max_lr - self.min_lr)/(self.freq - 1)
            res = self.min_lr + (self.max_lr - self.min_lr)*(1 + math.cos(math.pi*delta_epochs/(self.freq-1)))/2.
            return [res]*len(self.base_lrs)

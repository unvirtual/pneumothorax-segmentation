import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import data_loader as dl
import checkpoint as cp
import augmentations as augmentations
import pandas as pd

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn


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

class DataLoaders:
    def __init__(self, train_ds, val_ds, train_bs, val_bs, train_workers=12, val_workers=4):
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.train_bs = train_bs
        self.val_bs = val_bs
        self.train_workers = train_workers
        self.val_workers = val_workers

    def loaders(self):
        train_loader = DataLoader(dataset=self.train_ds,
                                  batch_size=self.train_bs,
                                  num_workers=self.train_workers,
                                  shuffle=True)
        val_loader = DataLoader(dataset=self.val_ds,
                                batch_size=self.val_bs,
                                num_workers=self.val_workers,
                                shuffle=False)
        return {"train": train_loader, "val": val_loader}

    def datasets(self):
        return {"train": self.train_ds, "val": self.val_ds}

    def dataframes(self):
        return {"train": self.train_ds.get_dataframe(),
                "val": self.val_ds.get_dataframe()}

    def get_img_size(self):
        assert(self.train_ds.get_img_size() == self.val_ds.get_img_size())
        return self.train_ds.get_img_size()

class Trainer:
    BASE_DIR="runs"

    @staticmethod
    def checkpoint_exists(name):
        return os.path.exists(Trainer.BASE_DIR + "/last_checkpoint.pth") or os.path.exists(Trainer.BASE_DIR + "/best_checkpoint.pth")

    @staticmethod
    def last_checkpoint_path(name):
        return Trainer.BASE_DIR + "/last_checkpoint.pth"

    @staticmethod
    def best_checkpoint_path(name):
        return Trainer.BASE_DIR + "/best_checkpoint.pth"

    def __init__(self, name, model, optimizer, epochs, loaders, scheduler=None, freeze_encoder_epochs=[], device="cpu", new_run_dir=True):
        self.name = name
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.epochs = epochs
        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.start_epoch = 0
        self.device = device
        self.max_score = 0
        self.loaders = loaders
        self.train_loader = self.loaders.loaders()["train"]
        self.val_loader = self.loaders.loaders()["val"]
        self._setup()

    def _setup(self):
        self.loss = smp.utils.losses.BCEDiceLoss(eps=1.)
        self.metrics = [
                    smp.utils.metrics.IoUMetric(eps=1.),
                    smp.utils.metrics.FscoreMetric(eps=1.),
        ]
        self.history = cp.History(metrics = ["bce_dice_loss", "iou", "f-score"],
                                  aux = ["lr", "input_size", "train_mode"])


        self.train_epoch = smp.utils.train.TrainEpoch(
            model=self.model,
            loss=self.loss,
            metrics=self.metrics,
            optimizer=self.optimizer,
            device=self.device,
            verbose=True,
            )

        self.valid_epoch = smp.utils.train.ValidEpoch(
            model=self.model,
            loss=self.loss,
            metrics=self.metrics,
            device=self.device,
            verbose=True,
            )

        self.checkpoint_best_filename = Trainer.best_checkpoint_path(self.name)
        self.checkpoint_last_filename = Trainer.last_checkpoint_path(self.name)
        self.encoder_frozen = False

    def _freeze_encoder(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = False
        self.encoder_frozen = True

    def _thaw_encoder(self):
        for p in self.model.encoder.parameters():
            p.requires_grad = True
        self.encoder_frozen = False

    def _save_checkpoints(self, epoch, train_logs, valid_logs):
        model_improved = False
        if len(self.freeze_encoder_epochs) > 0:
            aux = {"freeze_encoder_epochs": self.freeze_encoder_epochs}
        else:
            aux = None

        dataframes = self.loaders.dataframes()

        if self.max_score < valid_logs['f-score']:
            self.max_score = valid_logs['f-score']
            checkpoint_best = cp.CheckPoint("Best", self.epochs)
            checkpoint_best.save(self.checkpoint_best_filename, epoch, self.model,
                                 self.optimizer, self.history, scheduler=self.scheduler,
                                 train_df=dataframes["train"], val_df=dataframes["val"],
                                 aux=aux)
            model_improved = True

        checkpoint_last = cp.CheckPoint("Last", self.epochs)
        checkpoint_last.save(self.checkpoint_last_filename, epoch, self.model,
                             self.optimizer, self.history, scheduler=self.scheduler,
                             train_df=dataframes["train"], val_df=dataframes["val"],
                             aux=aux)

        return model_improved

    def _train_epoch(self):
        train_logs = self.train_epoch.run(self.train_loader)
        valid_logs = self.valid_epoch.run(self.val_loader)

        if self.scheduler is not None:
            self.scheduler.step(train_logs, valid_logs)

        return train_logs, valid_logs

    def _update_history(self, train_logs, valid_logs):
        aux_hist = {"lr": [p['lr'] for p in self.optimizer.param_groups],
                    "input_size": self.loaders.get_img_size()}

        if self.encoder_frozen:
            aux_hist["train_mode"] = "encoder_frozen"
        else:
            aux_hist["train_mode"] = "full"

        self.history.append(train_logs, valid_logs, aux_hist)

    def _frozen_encoder_epoch(self, epoch):
        return len(self.freeze_encoder_epochs) > 0 and epoch in self.freeze_encoder_epochs

    def train(self):
        if not self.encoder_frozen and self._frozen_encoder_epoch(self.start_epoch):
            print("Freezing encoder weights")
            self._freeze_encoder()

        for i in range(self.start_epoch, self.epochs):
            print('\nEpoch: %i/%i' % (i+1, self.epochs))
            print("Current Learning Rates:")
            for param_group in self.optimizer.param_groups:
                print(param_group['lr'])

            if self.encoder_frozen and not self._frozen_encoder_epoch(i):
                print("Thawing encoder weights")
                self._thaw_encoder()

            train_logs, valid_logs = self._train_epoch()
            self._update_history(train_logs, valid_logs)
            improved = self._save_checkpoints(i, train_logs, valid_logs)
            if improved:
                print('Model improved, checkpoint saved!')

    @classmethod
    def from_checkpoint(cls, name, checkpoint, loaders, device, newname=None):
        start_epoch = checkpoint.last_epoch + 1
        epochs = checkpoint.total_epochs
        freeze_encoder_epochs = checkpoint.aux["freeze_encoder_epochs"]

        instance = cls(name, checkpoint.model, checkpoint.optimizer, 
                epochs, loaders, scheduler=checkpoint.scheduler,
                freeze_encoder_epochs=freeze_encoder_epochs, device=device,
                new_run_dir=False)

        instance.start_epoch = start_epoch
        instance.history = checkpoint.history

        return instance

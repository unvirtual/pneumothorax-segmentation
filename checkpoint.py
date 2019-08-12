import torch
import pickle
import os

class History:
    def __init__(self, metrics, aux, first_epoch=0):
        self.first_epoch = first_epoch
        self.train_log = {}
        self.val_log = {}
        self.aux_log = {}
        self.metrics = metrics
        self.aux = aux
        self.epochs = []
        self._current_epoch = first_epoch
        for m in metrics:
            self.train_log[m] = []
            self.val_log[m] = []
        self.aux_log = {}
        for a in aux:
            self.aux_log[a] = []

    def append(self, train_log, val_log, aux_logs):
        for m in self.metrics:
            self.train_log[m].append(train_log[m])
            self.val_log[m].append(val_log[m])
        for a in self.aux:
            self.aux_log[a].append(aux_logs[a])
        self.epochs.append(self._current_epoch)
        self._current_epoch += 1

    def logs(self):
        return {'epochs':self.epochs, 'train': self.train_log, 'val': self.val_log, 'aux': self.aux_log}


class CheckPoint:
    def __init__(self, name, total_epochs):
        self.name = name
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.history = None
        self.aux = None
        self.train_df = None
        self.val_df = None
        self.total_epochs = total_epochs
        self.last_epoch = None

    def save(self, filename, epoch, model, optimizer, history, train_df=None, val_df=None, scheduler=None, aux=None):
        state = {'name': self.name, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "total_epochs": self.total_epochs, "epoch": epoch}
        state["history"] = pickle.dumps(history)
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()
        if aux is not None:
            state["aux"] = aux
        if train_df is not None:
            state["train_df"] = pickle.dumps(train_df)
        if val_df is not None:
            state["val_df"] = pickle.dumps(val_df)
        torch.save(state, filename)

    @classmethod
    def load(cls, filename, model, optimizer=None, scheduler=None, device='cuda'):
        if not os.path.isfile(filename):
            print("no checkpoint found at %s" % filename)
        fc = torch.load(filename, map_location=device)
        checkpoint = cls(fc["name"], fc["total_epochs"])
        model.load_state_dict(fc["model"])
        checkpoint.model = model
        if optimizer is not None:
            print("loading optimizer")
            optimizer.load_state_dict(fc["optimizer"])
            checkpoint.optimizer = optimizer
        if scheduler is not None:
            checkpoint.scheduler = scheduler.load_state_dict(fc["scheduler"])
        checkpoint.history = pickle.loads(fc["history"])
        if "aux" in fc:
            checkpoint.aux = fc["aux"]
        if "train_df" in fc:
            checkpoint.train_df = pickle.loads(fc["train_df"])
        if "val_df" in fc:
            checkpoint.val_df = pickle.loads(fc["val_df"])

        checkpoint.last_epoch = fc["epoch"]
        return checkpoint

    @staticmethod
    def load_history(filename, device="cuda"):
        if not os.path.isfile(filename):
            print("no checkpoint found at %s" % filename)
        fc = torch.load(filename, map_location=device)
        return pickle.loads(fc["history"])

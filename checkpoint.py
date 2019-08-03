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

    def get_logs(self):
        return {'epochs':self.epochs, 'train': self.train_log, 'val': self.val_log, 'aux': self.aux_log}

    def get_epochs(self):
        return self.epochs

    def get_train_logs(self):
        return self.train_log

    def get_val_logs(self):
        return self.val_log

    def get_metrics(self):
        return self.metrics

class CheckPoint:
    def __init__(self, name, desc):
        self.name = name
        self.desc = desc
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.history = None

    def save(self, filename, model, optimizer, history, scheduler=None):
        state = {'name': self.name, 'desc': self.desc,
                 'model': model.state_dict(),
                 'optimizer': optimizer.state_dict()}
        state["history"] = pickle.dumps(history)
        if scheduler is not None:
            state["scheduler"] = scheduler.state_dict()

        torch.save(state, filename)

    @classmethod
    def load(cls, filename, model, optimizer=None, scheduler=None, device='cuda'):
        if not os.path.isfile(filename):
            print("no checkpoint found at %s" % filename)
        fc = torch.load(filename, map_location=device)
        checkpoint = cls(fc["name"], fc["desc"])
        model.load_state_dict(fc["model"])
        checkpoint.model = model
        if optimizer is not None:
            optimizer.load_state_dict(fc["optimizer"])
            checkpoint.optimizer = optimizer
        if scheduler is not None:
            scheduler.load_state_dict(fc["scheduler"])
            checkpoint.scheduler = scheduler
        checkpoint.history = pickle.loads(fc["history"])
        return checkpoint

    @staticmethod
    def load_history(filename, device="cuda"):
        if not os.path.isfile(filename):
            print("no checkpoint found at %s" % filename)
        fc = torch.load(filename, map_location=device)
        return pickle.loads(fc["history"])

    def get_model(self):
        return self.model

    def get_optimizer(self):
        return self.optimizer

    def get_scheduler(self):
        return self.scheduler

    def get_history(self):
        return self.history

    def get_name_and_description(self):
        return self.name, self.desc

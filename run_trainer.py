import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import data_loader as dl
import checkpoint as cp
import augmentations as augmentations
import pandas as pd
import argparse
from trainer import Trainer, DataLoaders, Scheduler
from validator import Validator
from model import *

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

EPOCHS = 40
FREEZE_ENCODER_EPOCHS = []
TRAIN_BS = 32
VAL_BS = 32

IMGSIZE = 256
IN_CHANNELS = 3
VAL_SPLIT = 0.2
SAMPLE_FRAC=1
EVAL_VAL = True
EVAL_TRAIN = False
SETUP_DIR="setup_checkpoint"

preprocess_input = ResNetModel.input_preprocess_function("resnet34", pretrained="imagenet")
#preprocess_input = None


def main(name=None):
    global TRAIN_BS, VAL_BS
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: %s" % device)
    n_devices = torch.cuda.device_count()
    if n_devices > 1:
        print("found multiple GPUs:", n_devices)
        TRAIN_BS = n_devices*TRAIN_BS
        VAL_BS = n_devices*VAL_BS
        print("scaling batch sizes:", TRAIN_BS, VAL_BS)

    siim_df_filename = "full_metadata_df.pkl"
    if os.path.exists(siim_df_filename):
        print("Using existing metadata file")
        df = dl.SIIMDataFrame(pd.read_pickle(siim_df_filename))
    else:
        print("Creating metadata file")
        df = dl.SIIMDataFrame.from_dirs("data-original", "data-original/train-rle.csv")
        df.to_pickle(siim_df_filename)

    print("loading done")

    train, val = df.train_val_split(VAL_SPLIT, stratify=False,
                                    sample_frac=SAMPLE_FRAC)

    train_transforms = augmentations.get_augmentations()

    model = ResUNetPlusPlus("resnet34", pretrained="imagenet", interpolate=None, dropout=0.5)

    optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    #optimizer = optim.Adam(model.parameters(), lr=5e-3)
    #torch_scheduler = optim.lr_scheduler.CyclicLR(optimizer, 5e-4, 1e-2, step_size_up=10, step_size_down=20)
    torch_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.3, patience=3, mode="max")

    scheduler = Scheduler(torch_scheduler, step_criterion="f-score", step_log="val")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    from_checkpoint = Trainer.checkpoint_exists(name)
    from_setup = os.path.exists(SETUP_DIR+ "/last_checkpoint.pth")

    if from_checkpoint:
        print("RESUMING training from checkpoint")
        checkpoint_filename = Trainer.last_checkpoint_path(name)
        checkpoint = cp.CheckPoint.load(checkpoint_filename, model, optimizer=optimizer, scheduler=scheduler, device=device)
    elif from_setup:
        print("INITIALIZING NEW training from setup checkpoint")
        checkpoint_filename = SETUP_DIR + "/last_checkpoint.pth"
        checkpoint = cp.CheckPoint.load(checkpoint_filename, model, optimizer=None, scheduler=None, device=device)
        checkpoint.scheduler = scheduler
        checkpoint.optimizer = optimizer
    else:
        print("STARTING NEW training")

    if from_checkpoint or from_setup:
        train = checkpoint.train_df
        val = checkpoint.val_df

    train_ds = dl.SIIMDataSet(train, IMGSIZE, IN_CHANNELS, preproc=preprocess_input,
                              augmentations=train_transforms)
    val_ds = dl.SIIMDataSet(val, IMGSIZE, IN_CHANNELS, preproc=preprocess_input)

    print("input image shape: ", train_ds.__getitem__(0)[0].shape)
    print("mask shape:        ", train_ds.__getitem__(0)[1].shape)
    print()
    print("train set:")
    print("   images   : ", train.shape[0])
    print("   positives: ", train["HasMask"].sum())
    print("   negatives: ", (~train["HasMask"]).sum())
    print("val set:")
    print("   images   : ", val.shape[0])
    print("   positives: ", val["HasMask"].sum())
    print("   negatives: ", (~val["HasMask"]).sum())
    print()

    loaders = DataLoaders(train_ds, val_ds, TRAIN_BS, VAL_BS)

    if from_checkpoint:
        print("Resuming Trainer ...")
        trainer = Trainer.from_checkpoint(name, checkpoint, loaders, device=device)
    elif from_setup:
        print("Initializing Trainer from setup checkpoint...")
        trainer = Trainer.from_checkpoint(name, checkpoint, loaders, device=device)
        trainer.epochs = EPOCHS + trainer.start_epoch
        trainer.freeze_encoder_epochs = FREEZE_ENCODER_EPOCHS
    else:
        print("Starting Trainer ...")
        trainer = Trainer(name, model, optimizer, EPOCHS, loaders, scheduler=scheduler, device=device, freeze_encoder_epochs=FREEZE_ENCODER_EPOCHS)

    trainer.train()

    if EVAL_VAL:
        print("Evaluating validation set")
        validator = Validator(model, optimizer, loaders.loaders()["val"], loaders.get_img_size())
        validator.run(50, device=device)
        validator.write_to_file("runs/val_evaluation.pkl")

    if EVAL_TRAIN:
        print("Evaluating train set")
        validator = Validator(model, optimizer, loaders.loaders()["train"], loaders.get_img_size())
        validator.run(50, device=device)
        validator.write_to_file("runs/train_evaluation.pkl")

    with open("runs/FINISHED", "w") as file:
        file.write("finished")


if __name__== "__main__":
    main()

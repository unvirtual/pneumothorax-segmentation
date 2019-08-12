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
FREEZE_ENCODER_EPOCHS = range(10)
TRAIN_BS = 32
VAL_BS = 32
IMGSIZE = 256
IN_CHANNELS = 3
VAL_SPLIT = 0.2
SAMPLE_FRAC=1
EVAL_VAL = True
EVAL_TRAIN = False

preprocess_input = ResNetModel.input_preprocess_function("resnet34", pretrained="imagenet")

def main(name=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: %s" % device)

    siim_df_filename = "full_metadata_df.pkl"
    df = dl.SIIMDataFrame(pd.read_pickle(siim_df_filename))
    
    print("loading done")

    train, val = df.train_val_split(VAL_SPLIT, stratify=False, 
                                    sample_frac=SAMPLE_FRAC)

    print("train set:")
    print("   images   : ", train.shape[0])
    print("   positives: ", train["HasMask"].sum())
    print("   negatives: ", (~train["HasMask"]).sum())
    print("val set:")
    print("   images   : ", val.shape[0])
    print("   positives: ", val["HasMask"].sum())
    print("   negatives: ", (~val["HasMask"]).sum())

 
    train_transforms = augmentations.get_augmentations()

    model = ResUNet("resnet34", pretrained="imagenet")
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=0.9)
    #torch_scheduler = optim.lr_scheduler.CyclicLR(optimizer, 5e-4, 5e-3, step_size_up=25, step_size_down=15)
    torch_scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.2, patience=8)

    scheduler = Scheduler(torch_scheduler)
    
    model = model.to(device)

   
    from_checkpoint = Trainer.checkpoint_exists(name)

    if from_checkpoint:
        checkpoint_filename = Trainer.last_checkpoint_path(name)
        checkpoint = cp.CheckPoint.load(checkpoint_filename, model, optimizer=optimizer, scheduler=scheduler, device=device)
        train = checkpoint.train_df
        val = checkpoint.val_df

    train_ds = dl.SIIMDataSet(train, IMGSIZE, IN_CHANNELS, preproc=preprocess_input, 
                              augmentations=train_transforms)
    val_ds = dl.SIIMDataSet(val, IMGSIZE, IN_CHANNELS, preproc=preprocess_input)

    print("input image shape: ", train_ds.__getitem__(0)[0].shape)
    print("mask shape:        ", train_ds.__getitem__(0)[1].shape)
    
    loaders = DataLoaders(train_ds, val_ds, TRAIN_BS, VAL_BS)
    
    if from_checkpoint:
        trainer = Trainer.from_checkpoint(name, checkpoint, loaders, device=device)
    else: 
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

if __name__== "__main__":
    main()

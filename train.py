import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import data_loader as dl
import checkpoint as cp
import augmentations as augmentations
import pandas as pd

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn

parser = argparse.ArgumentParser()
parser.add_argument('train_bs', type=int)
parser.add_argument('val_bs', type=int)
parser.add_argument('epochs', type=int)
parser.add_argument('model_file', type=str)
parser.add_argument('--inchannels', type=int, default=3)
parser.add_argument('--network', type=str, default="unet")
parser.add_argument('--freeze_encoder_epochs', type=int, default=0)
parser.add_argument('--encoder', type=str, default="resnet34")
parser.add_argument('--train_val_split', type=float, default=0.2)
parser.add_argument('--data_frac', type=float, default=1.0)
parser.add_argument('--imgsize', type=int, default=256)
parser.add_argument('--from_checkpoint', type=str, default="")

env = parser.parse_args()

encoder_name = env.encoder

start_from_checkpoint = False
if env.from_checkpoint != "":
    checkpoint = env.from_checkpoint
    start_from_checkpoint = True

train_batch_size = env.train_bs
val_batch_size = env.val_bs

epochs = env.epochs
freeze_encoder_epochs = env.freeze_encoder_epochs

train_val_split = env.train_val_split
data_frac = env.data_frac

imgsize = env.imgsize
inchannels = env.inchannels

#Set always to guarantee preprocessing
pretrained = "imagenet"

network = env.network

def print_parameters():
    print("Starting training")
    print("=================")
    if start_from_checkpoint:
        print("Starting from checkpoint: ", checkpoint)
    print("Network %s with encoder %s" % (network, encoder_name))
    print("Pretrained on %s:" % pretrained)
    print("Batch sizes: train=%i, val=%i" % (train_batch_size, val_batch_size))
    print("Freezing encoder for %s epochs" % freeze_encoder_epochs)
    print("Epochs: %i" % epochs)
    print("Training validation split: %f" % train_val_split)
    if data_frac < 1:
        print("Drawing sample from full data set: %f" % data_frac)
    else:
        print("using full dataset")

def main():
    print_parameters()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: %s" % device)

    siim_df_filename = "full_metadata_df.pkl"
    df = dl.SIIMDataFrame(pd.read_pickle(siim_df_filename))
    #df = SIIMDataFrame(pd.read_pickle(siim_df_filename), only_labeled=True)

    if pretrained is not None:
        preprocess_input = get_preprocessing_fn(encoder_name, pretrained=pretrained)
    else:
        preprocess_input = None

    train, val = df.train_val_split(
            train_val_split,
            stratify=False,
            sample_frac=data_frac)

    print("train set:")
    print("   images   : ", train.shape[0])
    print("   positives: ", train["HasMask"].sum())
    print("   negatives: ", (~train["HasMask"]).sum())
    print("val set:")
    print("   images   : ", val.shape[0])
    print("   positives: ", val["HasMask"].sum())
    print("   negatives: ", (~val["HasMask"]).sum())

    train_transforms = augmentations.get_augmentations()
    train_ds = dl.SIIMDataSet(train, imgsize, inchannels, preproc=preprocess_input, augmentations=train_transforms)
    val_ds = dl.SIIMDataSet(val, imgsize, inchannels, preproc=preprocess_input)

    print("input image shape: ", train_ds.__getitem__(0)[0].shape)
    print("mask shape:        ", train_ds.__getitem__(0)[1].shape)

    train_loader = DataLoader(
            dataset=train_ds,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=12)
    val_loader = DataLoader(
            dataset=val_ds,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=4)

    model = smp.Unet(encoder_name, classes=1, encoder_weights=pretrained, activation="sigmoid", decoder_use_batchnorm=True).to(device)

    optimizer = torch.optim.Adam([
            {'params': model.decoder.parameters(), 'lr': 1e-4},
            {'params': model.encoder.parameters(), 'lr': 1e-4}
            ])

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)

    loss = smp.utils.losses.BCEDiceLoss(eps=1.)
    #loss = MyBCEWithLogitsLoss()

    metrics = [
                smp.utils.metrics.IoUMetric(eps=1.),
                smp.utils.metrics.FscoreMetric(eps=1.),
    ]

    history = cp.History(
        metrics = ["bce_dice_loss", "iou", "f-score"],
        aux = ["lr", "input_size", "train_mode"]
    )

    start_epoch = 0

#    if start_from_checkpoint:
#        cp = CheckPoint.load(checkpoint, model, optimizer, scheduler)
#        print("Using Checkpoint:")
#        print(cp.get_name_and_description())
#
#        model = cp.get_model()
#        optimizer = cp.get_optimizer()
#        scheduler = cp.get_scheduler()
#        history = cp.get_history()
#        start_epoch = history.get_epochs()[-1] + 1

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=4)

    train_epoch = smp.utils.train.TrainEpoch(
            model,
            loss=loss,
            metrics=metrics,
            optimizer=optimizer,
            device=device,
            verbose=True,
            )

    valid_epoch = smp.utils.train.ValidEpoch(
            model,
            loss=loss,
            metrics=metrics,
            device=device,
            verbose=True,
            )

    max_score = 0
    if freeze_encoder_epochs > 0:
        print("Freezing encoder weights")
        for p in model.encoder.parameters():
            p.requires_grad = False

    for i in range(epochs):
        print('\nEpoch: %i/%i' % (i+start_epoch+1, start_epoch + epochs))
        print("Current Learning Rates:")
        for param_group in optimizer.param_groups:
            print(param_group['lr'])

        if i == freeze_encoder_epochs:
            print("Thawing encoder weights")
            for p in model.encoder.parameters():
                p.requires_grad = True

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        if scheduler is not None:
            scheduler.step(valid_logs['bce_dice_loss'])

        aux_hist = {"lr": [p['lr'] for p in optimizer.param_groups], "input_size":imgsize}
        if i < freeze_encoder_epochs:
            aux_hist["train_mode"] = "encoder_frozen"
        else:
            aux_hist["train_mode"] = "full"

        history.append(train_logs, valid_logs, aux_hist)

        if max_score < valid_logs['f-score']:
            max_score = valid_logs['f-score']
            checkpoint_best = cp.CheckPoint("Best", "Bla")
            checkpoint_best.save("best.pth", model, optimizer, history, scheduler)
            print('Model improved, checkpoint saved!')


        checkpoint_last = cp.CheckPoint("Last", "Bla")
        checkpoint_last.save("last.pth", model, optimizer, history, scheduler)

if __name__=="__main__":
    main()

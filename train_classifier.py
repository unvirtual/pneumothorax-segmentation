import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader
from data_loader import *
from checkpoint import *
import torchvision
import torchvision.transforms as transforms
from collections import OrderedDict

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import albumentations as albu

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

checkpoint_last_filename = env.model_file + "_last_cp.pth"
checkpoint_best_filename = env.model_file + "_best_cp.pth"

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

train_transform = [
#    albu.HorizontalFlip(p=0.5),

    albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0.2, shift_limit=0.1, p=1, border_mode=0),

    #albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
    #albu.RandomCrop(height=320, width=320, always_apply=True),

    #albu.IAAAdditiveGaussianNoise(p=0.2),
    #albu.IAAPerspective(p=0.2),

#    albu.OneOf(
#        [
#            albu.CLAHE(p=1),
#            albu.RandomBrightness(p=1),
#            albu.RandomGamma(p=1),
#        ],
#        p=0.5,
#    ),
#
#    #albu.OneOf(
#    #    [
#    #        albu.IAASharpen(p=1),
#    #        albu.Blur(blur_limit=1, p=1),
#    #    ],
#    #    p=0.2,
#    #),
#
#    albu.OneOf(
#        [
#            albu.RandomContrast(p=1),
#            albu.HueSaturationValue(p=1),
#        ],
#        p=0.5,
#    )
]
train_transforms = albu.Compose(train_transform)

class BCELoss(nn.Module):
    __name__ = 'bce_loss'

    def __init__(self, thr=None):
        super(BCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        #self.sigmoid = nn.Sigmoid()
        self.thr = thr

    def forward(self, y_pr, y_gt):
        #y_pr = self.sigmoid(y_pr)
        #if self.thr is not None:
        #    y_pr = (y_pr > self.thr).float()
        bce = self.bce(y_pr, y_gt)
        return bce

def main():
    print_parameters()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: %s" % device)

    siim_df_filename = "full_metadata_df.pkl"
    df = SIIMDataFrame(pd.read_pickle(siim_df_filename))

    preprocess_input = get_preprocessing_fn("resnet34", pretrained="imagenet")

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


    train_ds = SIIMDataSet(train, imgsize, inchannels, preproc=preprocess_input, augmentations=train_transforms, image_label=True)
    val_ds = SIIMDataSet(val, imgsize, inchannels, preproc=preprocess_input, image_label=True)

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

    model = torchvision.models.resnet50(pretrained=True)

    n_inputs = model.fc.in_features

    # add more layers as required
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(n_inputs, 1))
    ]))
    
    model.fc = classifier

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)

    loss = BCELoss(thr=None)

    last_state_checkpoint = CheckPoint("LAST", "Classifier for mask present.")
    best_checkpoint = CheckPoint("BEST", "Classifier for mask present")

    history = History(
        metrics = ["bce_loss", "f-score"],
        aux = ["lr", "input_size"]
    )

    metrics = [
                smp.utils.metrics.FscoreMetric(eps=1.),
    ]

    start_epoch = 0

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.2, patience=3)

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
#    if freeze_encoder_epochs > 0:
#        print("Freezing encoder weights")
#        for p in model.encoder.parameters():
#            p.requires_grad = False

    for p in model.parameters():
        p.requires_grad = False
 
    for p in model.fc.parameters():
        p.requires_grad = True


    for i in range(epochs):
        print('\nEpoch: %i/%i' % (i+start_epoch+1, start_epoch + epochs))
        print("Current Learning Rates:")
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        
#        if i == freeze_encoder_epochs:
#            print("Thawing encoder weights")
#            for p in model.encoder.parameters():
#                p.requires_grad = True

        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)

        if scheduler is not None:
            scheduler.step(valid_logs['bce_loss'])

        aux_hist = {"lr": [p['lr'] for p in optimizer.param_groups], "input_size":imgsize}
        if i < freeze_encoder_epochs:
            aux_hist["train_mode"] = "encoder_frozen"
        else:
            aux_hist["train_mode"] = "full"

        history.append(train_logs, valid_logs, aux_hist)

        if max_score < valid_logs['f-score']:
            max_score = valid_logs['f-score']
            best_checkpoint.save(checkpoint_best_filename, model, optimizer, history, scheduler)
            print('Model improved, checkpoint saved!')

        last_state_checkpoint.save(checkpoint_last_filename, model, optimizer, history, scheduler)

if __name__=="__main__":
    main()

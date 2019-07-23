import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pickle
from torch.utils.data import DataLoader
from data_loader import *

import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import albumentations as albu

parser = argparse.ArgumentParser()
parser.add_argument('train_bs', type=int) 
parser.add_argument('val_bs', type=int)
parser.add_argument('init_epochs', type=int)
parser.add_argument('finetune_epochs', type=int)
parser.add_argument('model_file', type=str)
parser.add_argument('--inchannels', type=int, default=3)
parser.add_argument('--network', type=str, default="unet")
parser.add_argument('--pretrained', type=str, default="imagenet")
parser.add_argument('--encoder', type=str, default="resnet34")
parser.add_argument('--train_val_split', type=float, default=0.2)
parser.add_argument('--data_frac', type=float, default=1.0)
parser.add_argument('--imgsize', type=int, default=256)

env = parser.parse_args()

encoder_name = env.encoder
model_file_name = env.model_file

train_batch_size = env.train_bs
val_batch_size = env.val_bs

pretrain_epochs = env.init_epochs
finetune_epochs = env.finetune_epochs

train_val_split = env.train_val_split
data_frac = env.data_frac

imgsize = env.imgsize
inchannels = env.inchannels

if env.pretrained == "none":
    pretrained = None
else:
    pretrained = env.pretrained

network = env.network

def print_parameters():
    print("Starting training")
    print("=================")
    print("Network %s with encoder %s" % (network, encoder_name))
    print("Pretrained on %s:" % pretrained)
    print("Batch sizes: train=%i, val=%i" % (train_batch_size, val_batch_size))
    print("Epochs:      pretrain=%i, finetune=%i" % (pretrain_epochs, finetune_epochs))
    print("Training validation split: %f" % train_val_split)
    if data_frac < 1:
        print("Drawing sample from full data set: %f" % data_frac)
    else:
        print("using full dataset")

train_transform = [
#    albu.HorizontalFlip(p=0.5),

    albu.ShiftScaleRotate(scale_limit=0.2, rotate_limit=0.1, shift_limit=0.1, p=1, border_mode=0),

    #albu.PadIfNeeded(min_height=320, min_width=320, always_apply=True, border_mode=0),
    #albu.RandomCrop(height=320, width=320, always_apply=True),

    #albu.IAAAdditiveGaussianNoise(p=0.2),
    #albu.IAAPerspective(p=0.2),

    albu.OneOf(
        [
            albu.CLAHE(p=1),
            albu.RandomBrightness(p=1),
            albu.RandomGamma(p=1),
        ],
        p=0.5,
    ),
#
#    #albu.OneOf(
#    #    [
#    #        albu.IAASharpen(p=1),
#    #        albu.Blur(blur_limit=1, p=1),
#    #    ],
#    #    p=0.2,
#    #),
#
    albu.OneOf(
        [
            albu.RandomContrast(p=1),
            albu.HueSaturationValue(p=1),
        ],
        p=0.5,
    )
]
train_transforms = albu.Compose(train_transform)
train_transforms = None

class FocalLoss(nn.Module):
    __name__ = "focal_loss"

    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True, activation="sigmoid"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        if activation is None or activation == "none":
            self.activation_fn = lambda x: x
        elif activation == "sigmoid":
            self.activation_fn = torch.nn.Sigmoid()
        elif activation == "softmax2d":
            self.activation_fn = torch.nn.Softmax2d()
        else:
            raise NotImplementedError(
                "Activation implemented for sigmoid and softmax2d"
            )
    

    def forward(self, inputs, targets):
        inputs = self.activation_fn(inputs)
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def main():
    print_parameters()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("Using device: %s" % device)

    siim_df_filename = "full_metadata_df.pkl"
    df = SIIMDataFrame(pd.read_pickle(siim_df_filename))

    if network == "unet":
        model = smp.Unet(encoder_name, classes=1, encoder_weights=pretrained, activation="sigmoid").to(device)
    elif network == "fpn":
        model = smp.FPN(encoder_name, classes=1, encoder_weights=pretrained, activation="sigmoid").to(device)
    elif network == "pspnet":
        model = smp.PSPNet(encoder_name, classes=1, encoder_weights=pretrained, activation="sigmoid").to(device)

    if pretrained is not None:
        preprocess_input = get_preprocessing_fn(encoder_name, pretrained=pretrained)
    else:
        preprocess_input = None

    print(model)

#    optimizer = torch.optim.Adam([
#            {'params': model.decoder.parameters(), 'lr': 1e-4}, 
#                
#            # decrease lr for encoder in order not to permute 
#            # pre-trained weights with large gradients on training
#            # start
#            {'params': model.encoder.parameters(), 'lr': 1e-4}
#            ])

    optimizer = torch.optim.SGD([
            {'params': model.decoder.parameters(), 'lr': 1e-2, "momentum": 0.9}, 
                
            # decrease lr for encoder in order not to permute 
            # pre-trained weights with large gradients on training
            # start
            {'params': model.encoder.parameters(), 'lr': 1e-2, "momentum": 0.9}
            ])

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


    train_ds = SIIMDataSet(train, imgsize, inchannels, preproc=preprocess_input, augmentations=train_transforms)
    val_ds = SIIMDataSet(val, imgsize, inchannels, preproc=preprocess_input)

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

    loss = smp.utils.losses.BCEDiceLoss(eps=1.)
    #loss = FocalLoss()
    metrics = [
                smp.utils.metrics.IoUMetric(eps=1.),
                smp.utils.metrics.FscoreMetric(eps=1.),
    ]

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

    train_history = []
    valid_history = []

    if pretrained is not None:
        print("Freezing encoder weights")
        for p in model.encoder.parameters():
            p.requires_grad = False

    for i in range(pretrain_epochs):
        print('\nPretrain Epoch: %i/%i' % (i, pretrain_epochs))
        print("Current Learning Rates:")
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)
        train_history.extend([train_logs])
        valid_history.extend([valid_logs])

    if pretrained is not None:
        print("Thawing encoder weights")
        for p in model.encoder.parameters():
            p.requires_grad = True

    for i in range(finetune_epochs):
        print('\nFinetune Epoch: %i/%i' % (i, finetune_epochs))
        print("Current Learning Rates:")
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(val_loader)
        train_history.extend([train_logs])
        valid_history.extend([valid_logs])

        if max_score < train_logs['f-score']:
            max_score = train_logs['f-score']
            torch.save(model, './' + model_file_name)
            print('Model saved!')

        history = {"train":train_history, "val":valid_history}
        with open("./"+model_file_name+".history.pkl", "wb") as f:
            pickle.dump(history, f)

if __name__=="__main__":
    main()

import glob
import torch
import pandas as pd
import cv2
import numpy as np
import pydicom
import os
import sys
import pickle
from skimage import exposure
from mask_utils import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

class SIIMDataFrame:
    def __init__(self, df, only_labeled=False, has_labels=True):
        if (not has_labels) and only_labeled:
            raise ValueError
        self.train_df = df
        self.has_labels = has_labels
        if only_labeled:
            self.train_df = self.train_df[self.train_df["HasMask"]].reset_index()


    @classmethod
    def from_dirs(cls, train_dir, labels_file, only_labeled=False):
        input_df = cls._create_input_dataset(train_dir)
        if labels_file is not None:
            labels_df = cls._create_label_dataset(labels_file)
            train_df = cls._merge_input_with_labels(input_df, labels_df)
            return cls(train_df, only_labeled=only_labeled)
        else:
            return cls(input_df, has_labels=False, only_labeled=only_labeled)

    def get_siim_dataframe(self):
        return self.train_df

    def to_pickle(self, filename):
        self.train_df.to_pickle(filename)

    def train_val_split(self, fraction, stratify=False, sample_frac=None):
        split_params = {}
        if stratify:
            split_params["stratify"] = self.train_df["HasMask"]

        if sample_frac is not None and sample_frac < 0.999:
            dpos = self.train_df[self.train_df["HasMask"]].copy()
            dneg = self.train_df[~self.train_df["HasMask"]].copy()
            ldpos = len(dpos)
            ldneg = len(dneg)

            weights=dpos["MaskCoverage"].value_counts(bins=50).to_dict()

            dpos["freq"] = dpos["MaskCoverage"].map(weights)

            pos_s = dpos.sample(frac=(1.-fraction)*sample_frac, weights="freq")
            neg_s = dneg.sample(frac=(1.-fraction)*sample_frac)

            pos_v = dpos[~dpos.index.isin(pos_s.index)]
            neg_v = dneg[~dneg.index.isin(neg_s.index)]

            pos_v = pos_v.sample(frac=1./(1-(1-fraction)*sample_frac)*fraction*sample_frac, weights="freq")
            neg_v = neg_v.sample(frac=1./(1-(1-fraction)*sample_frac)*fraction*sample_frac)

            train_set = pd.concat([pos_s, neg_s], sort=False).sample(frac=1).reset_index()
            val_set = pd.concat([pos_v, neg_v], sort=False).sample(frac=1).reset_index()
        else:
            train_df = self.train_df

            train_set, val_set, train_lbl, val_lbl = train_test_split(
                    train_df.drop(columns="EncodedPixels"),
                    train_df["EncodedPixels"],
                    test_size = fraction,
                    **split_params)

            train_set["EncodedPixels"] = train_lbl
            val_set["EncodedPixels"] = val_lbl

        train_set = train_set.reset_index()
        val_set = val_set.reset_index()
        return train_set, val_set

    @staticmethod
    def check_input_dataset(df):
        rows = df['Rows'].unique()
        columns = df['Columns'].unique()
        if len(rows) != 1 or len(columns) != 1:
            raise RuntimeError("ERROR: input images don't have the same size")
        print("Image dimensions: %d x %d" % (rows[0], columns[0]))
        print("Input data shape: %s" % str(df.shape))

    @staticmethod
    def _create_input_dataset(directory):
        input_df = pd.DataFrame(glob.glob(directory + "/**/*.dcm", recursive=True), columns = ["Path"])
        print("Found %i files" % len(input_df))
        input_df["ImageId"] = input_df["Path"].apply(lambda x: os.path.basename(x)[:-4])
        def df_extract_metadata_from_path(df):
            dcm = pydicom.dcmread(df['Path'])
            df['Age'] = dcm.PatientAge
            df['Sex'] = dcm.PatientSex
            df['Rows'] = dcm.Rows
            df['Columns'] = dcm.Columns
            return df
        input_df = input_df.apply(df_extract_metadata_from_path, axis=1)
        input_df['Age'] = input_df['Age'].astype('int32')
        return input_df

    @staticmethod
    def _create_label_dataset(filename, width=1024, height=1024):
        NO_MASK_STRING = ' -1'
        labels_df = pd.read_csv(filename)
        # Fix typo in target file
        labels_df = labels_df.rename({" EncodedPixels":"EncodedPixels"}, axis=1)
        # Images may contain multiple lre labels --> combine in df
        labels_df = pd.DataFrame(labels_df.groupby("ImageId")["EncodedPixels"].apply(list))
        labels_df['HasMask'] = labels_df["EncodedPixels"].apply(lambda x: x != [NO_MASK_STRING])
        labels_df['NMasks'] = labels_df['EncodedPixels'].apply(lambda x: len(x) if x != [NO_MASK_STRING] else 0)
        labels_df['OverlappingMasks'] = labels_df['EncodedPixels'].apply(lambda x: (sum([rle2mask(i, width=width, height=height)/255 for i in x]) > 1).any() if len(x) > 1 else False)
        mask_coverages_df = labels_df[labels_df['HasMask']]['EncodedPixels'].apply(lambda y: list(map(lambda x: (rle2mask(x, width=width, height=height)/255).sum()/(width*height),y)))
        labels_df["MaskCoverage"] = mask_coverages_df.apply(sum)
        return labels_df

    @staticmethod
    def _merge_input_with_labels(input_df, labels_df):
        print(10*"*" + " Before merge " + 10*"*")
        print("Input Data Shape:  %s" % str(input_df.shape))
        print("Labels Shape:      %s" % str(labels_df.shape))
        df = pd.merge(labels_df, input_df, on="ImageId", validate="one_to_one")
        print(10*"*" + " After merge " + 10*"*")
        print("Merged data shape: %s" % str(df.shape))
        return df

class SIIMDataSet(Dataset):
    def __init__(self, siim_df, img_size, n_channels, preproc=None, has_labels=True, augmentations=None, image_label=False):
        self.train_df = siim_df
        self.img_size = img_size
        self.n_channels = n_channels
        self.preproc = preproc
        self.has_labels = has_labels
        self.augmentations = augmentations
        self.image_label = image_label

    def __len__(self):
        return len(self.train_df["Path"])

    def __getitem__(self, index):
        if index >= len(self):
            raise IndexError
        img = self._get_img_array(self.train_df["Path"][index])
        img = np.stack((img,)*self.n_channels)
        img = img.transpose(1,2,0)

        label = None
        if self.image_label:
            label = self.train_df["HasMask"][index].astype(int)

        if self.has_labels and (not self.image_label):
            mask = self._create_mask(self.train_df.iloc[index])
            mask = cv2.resize(mask, (self.img_size, self.img_size))
            mask = mask.reshape((-1, self.img_size, self.img_size))
            mask = mask.transpose(1,2,0)
            if self.augmentations is not None:
                sample = self.augmentations(image=img, mask=mask)
                img, mask = sample['image'], sample['mask']
            mask = mask.transpose(2,0,1)

        if self.preproc is not None:
            img = self.preproc(img)

        img = img.transpose(2,0,1)/255.

        if self.image_label:
            return torch.from_numpy(img).float(), torch.from_numpy(label.reshape((-1))).float()
        if self.has_labels:
            return torch.from_numpy(img).float(), torch.from_numpy(mask).float()
        else:
            return torch.from_numpy(img).float(), self.train_df["Path"][index]


    def _create_mask(self, df_row):
        mask = np.zeros((1024, 1024))
        if df_row["HasMask"]:
            mask = sum(map(lambda r: rle2mask(r,1024,1024)/255., df_row["EncodedPixels"]))
        return np.clip(mask, 0, 1)

    def _get_img_array(self, path):
        dcm = pydicom.dcmread(path)
        return cv2.resize(dcm.pixel_array, (self.img_size, self.img_size))

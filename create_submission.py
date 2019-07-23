import torch
import pickle
from skimage import measure
from data_loader import *
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

DEV="cpu"

def create_rle(imgs, thr, min_pix, input_dims=(256, 256), output_dims=(1024,1024)):
    pixels = (imgs > thr/100.).astype(int)
        
    if pixels.sum() <= min_pix:
        return "-1"
                        
    clusters = measure.label(pixels, neighbors=8, background=0)
    clusters_unique = np.delete(np.unique(clusters), 0)
        
    mask = np.zeros((1,input_dims[0], input_dims[1]))
    for c in clusters_unique:
        m = np.zeros((1,input_dims[0],input_dims[1]))
        m[clusters == c] = 1
        if m.sum() > min_pix:
            mask += m    
    mask = np.clip(mask, 0,1)
    
    if mask.sum() > 1:
        mask = mask*255
        mask = cv2.resize(mask[0], (output_dims[0], output_dims[1]))
        mask = (mask > 128).astype(int)*255
        # transpose is important!
        rle = mask2rle(mask.T, output_dims[0], output_dims[1])
        return rle
    else:
        return "-1"

import os
DEV = "cpu"
def predict_rles(model_file, encoder, pretrained, directory, thr, min_pix, img_size):
    results = []
    print("Loading Model ...")
    model = torch.load(model_file, map_location=DEV).to(DEV)
    preprocess_input = get_preprocessing_fn(encoder, pretrained=pretrained)
    
    print("Loading images ...")
    df = SIIMDataFrame.from_dirs(directory, labels_file=None).get_siim_dataframe()
    ds = SIIMDataSet(df, img_size, 3, preproc=preprocess_input, has_labels=False)

    print("Loaded %i images" % len(ds))
    model.eval()

    for i, (image, path) in enumerate(ds):
        print("Prediction %i/%i" % (i+1,len(ds)))
        image = image.reshape((-1,3,img_size,img_size)).to(DEV)
        pred = model.predict(image)
        rle = create_rle(pred.cpu().numpy()[0], thr, min_pix, input_dims=(img_size, img_size))
        results.extend([[os.path.basename(path)[:-4], rle]])
    return results

results = predict_rles("unet_resnet34_input_256_best.pth", "resnet34", "imagenet", "data-original/dicom-images-test", 0.5, 50, 256)

pd.DataFrame(results, columns=["ImageId","EncodedPixels"]).to_csv("submission_unet_resnet34_256_best.csv", index=False)

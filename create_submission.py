import torch
import pickle
from skimage import measure
from data_loader import *
import numpy as np
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from multiprocessing import Pool, cpu_count
import os

DEV="cpu"

def extract_mask_clusters(pixels, min_pix, input_dims):
    clusters = measure.label(pixels, neighbors=8, background=0)
    clusters_unique = np.delete(np.unique(clusters), 0)
        
    mask = np.zeros((1,input_dims[0], input_dims[1]))
    for c in clusters_unique:
        m = np.zeros((1,input_dims[0],input_dims[1]))
        m[clusters == c] = 1
        if m.sum() > min_pix:
            mask += m    
    mask = np.clip(mask, 0,1)
    return mask

def filter_mask(pred_masks, thr, min_pix, input_dims=(256, 256)):
    assert (thr <= 1.0) and (thr >= 0), "Threshold must be in [0,1]"
    assert min_pix > 0, "min_pix must be > 0"
    assert pred_masks.max() <= 1.0, "pred_masks must be normalized to [0,1]"
    assert pred_masks[0].shape == input_dims, "mask shape does not match input dimensions"

    pixels = (pred_masks > thr).astype(int)

    # early exit
    if pixels.sum() <= min_pix:
        return None
    mask = extract_mask_clusters(pixels, min_pix, input_dims)

    if mask.sum() < 1:
        return None

    return mask


def rle_from_predicted_mask(mask, output_dims=(1024,1024)):
    if mask is None:
        return "-1"

    assert mask.max() <= 1., "pred mask must be normalized to [0,1]"

    mask = mask*255
    mask = cv2.resize(mask[0], (output_dims[0], output_dims[1]))
    mask = (mask > 128).astype(int)*255
    return mask2rle(mask.T, output_dims[0], output_dims[1])


def postprocess_predicted_mask(mask, path, thr, min_pix, input_dims):
    mask = mask.cpu().numpy()
    mask = filter_mask(mask, thr, min_pix, input_dims)
    rle = rle_from_predicted_mask(mask)
    return [os.path.basename(path)[:-4], rle]

BATCHSIZE=32
DEV = "cpu"
def predict_rles(model_file, encoder, pretrained, directory, thr, min_pix, img_size):
    results = []
    print("Loading Model", model_file)
    model = torch.load(model_file, map_location=DEV).to(DEV)
    preprocess_input = get_preprocessing_fn(encoder, pretrained=pretrained)
    
    print("Loading images from", directory)
    df = SIIMDataFrame.from_dirs(directory, labels_file=None).get_siim_dataframe()
    ds = SIIMDataSet(df, img_size, 3, preproc=preprocess_input, has_labels=False)

    pool = Pool(cpu_count())
    print("Loaded %i images" % len(ds))
    model.eval()

    dl = DataLoader(ds,BATCHSIZE)
    print("Batchsize:", BATCHSIZE)
    for i, (images, paths) in enumerate(dl):
        print("Batch %i/%i" % (i+1,len(dl)))
        images = images.to(DEV)
        pred = model.predict(images)
        batch_results = [pool.apply_async(postprocess_predicted_mask, (mask, path, thr, min_pix, (img_size, img_size))) for mask, path in zip(pred, paths)]
        for r in batch_results:
            results.extend([r.get()])
    pool.close()
    return results

if __name__ == "__main__":
    print("No. of CPUs:", cpu_count())
    print("batch size: ", BATCHSIZE)
    results = predict_rles("models/unet_resnet34_input_256_best.pth", "resnet34", "imagenet", "data-original/dicom-images-test", 0.5, 50, 256)
    pd.DataFrame(results, columns=["ImageId","EncodedPixels"]).to_csv("test_submission.csv", index=False)

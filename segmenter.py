import numpy as np
from cellpose import models
from cellpose import models, io
import glob
import os
import gcsfs
import imageio
from natsort import natsorted
import re

def main():
    # root_dir needs a trailing slash (i.e. /root/dir/)
    root_dir = '/media/prakashlab/T7/new_data/'#'gs://octopi-codex-data-processing/' #"/home/prakashlab/Documents/kmarx/pipeline/tstflat/"# 'gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/'#
    exp_id   = "20220822_20x_intestines/"
    channel =  "Fluorescence_405_nm_Ex" # only run segmentation on this channel
    cpmodel = "/home/prakashlab/Documents/newcodex/train/models/cellpose_residual_on_style_on_concatenation_off_train_2022_08_25_11_46_07.357844"
    channels = [0,0] # grayscale only
    key = '/home/prakashlab/Documents/fstack/codex-20220324-keys.json'
    use_gpu = True
    gcs_project = 'soe-octopi'
    run_seg(root_dir, exp_id, channel, cpmodel, channels, key, use_gpu, gcs_project)

def run_seg(root_dir, exp_id, channel, cpmodel, channels, key, use_gpu, gcs_project):
    # Load remote files if necessary
    root_remote = False
    if root_dir[0:5] == 'gs://':
        root_remote = True

    model_remote = False
    if cpmodel[0:5] == 'gs://':
        model_remote = True
    fs = None
    if model_remote or root_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)
    
    modelpath = "./cellpose_temp"
    if model_remote:
        fs.get(cpmodel, modelpath)
        cpmodel = modelpath

    print("Reading image paths")
    # filter - only look for specified channel and cycle 0
    path = root_dir + exp_id + "**/**/**/**" + channel + '.png'
    print(path)
    if root_remote:
        allpaths = [p for p in fs.glob(path, recursive=True) if p.split('/')[-2] == '0']
    else:
        allpaths = [p for p in glob.iglob(path, recursive=True) if p.split('/')[-2] == '0']
    # remove duplicates
    imgpaths = list(dict.fromkeys(allpaths))
    imgpaths = np.array(natsorted(imgpaths))
    print(str(len(imgpaths)) + " images to segment")

    print("Starting cellpose")
    # start cellpose
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=cpmodel)
    print("Starting segmentation")

    placeholder = "./placeholder.png"

    # segment one at a time - gpu bottleneck
    for idx, impath in enumerate(imgpaths):
        print(str(idx) + ": " + impath)
        if root_remote:
            im = np.array(imread_gcsfs(fs, impath), dtype=np.uint8)
        else:
            im = np.array(io.imread(impath), dtype=np.uint8)
        if np.max(im) == 0:
            print("no data")
            continue
        # normalize
        im = im - np.min(im)
        im = np.uint8(255 * np.array(im, dtype=np.float64)/float(np.max(im)))

        masks, flows, styles = model.eval(im, diameter=None, channels=channels)
        diams = 0
        if root_remote:
            savepath = placeholder
        else:
            savepath = impath
        # actually run the segmentation
        io.masks_flows_to_seg(im, masks, flows, diams, savepath, channels)

        # move the .npy to remote if necessary
        if root_remote:
            # generate the local and remote path names
            savepath = savepath.rsplit(".", 1)[0] + "_seg.npy"
            rpath = imgpaths[idx].rsplit(".", 1)[0] + "_seg.npy"
            fs.put(savepath, rpath)
            os.remove(savepath)

    if model_remote:
        os.remove(modelpath)

def imread_gcsfs(fs,file_path):
    '''
    imread_gcsfs gets the image bytes from the remote filesystem and convets it into an image
    
    Arguments:
        fs:         a GCSFS filesystem object
        file_path:  a string containing the GCSFS path to the image (e.g. 'gs://data/folder/image.bmp')
    Returns:
        I:          an image object
    
    This code has no side effects
    '''
    img_bytes = fs.cat(file_path)
    im_type = file_path.split('.')[-1]
    I = imageio.core.asarray(imageio.v2.imread(img_bytes, im_type))
    return I

if __name__ == "__main__":
    main()
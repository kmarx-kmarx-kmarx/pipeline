import numpy as np
import glob
import os
import gcsfs
import cv2
import math
import random
from dpc_overlay import generate_dpc
import imageio

def main():
    random.seed(2)
    root_dir = 'gs://octopi-tb-data/20221002'
    dest_dir = 'segments_neg' # must be a local path
    exp_id   = "**"
    channel =  "" # only run segmentation on this channel - leave empty for generating DPC
    zstack  = '0' # select which z to run segmentation on. set to 'f' to select the focus-stacked
    key = '/home/prakashlab/Documents/kmarx/deepzoom-tb-keys.json'
    ftype = 'bmp'
    gcs_project = 'soe-octopi'
    n_rand = 10
    nsub = 2 # cut into a nsub x nsub grid and return a random selection
    get_rand(root_dir, dest_dir, exp_id, channel, zstack, n_rand, key, nsub, gcs_project, ftype)

def get_rand(root_dir, dest_dir, exp_id, channel, zstack, n_rand, key, nsub, gcs_project, ftype):
    # Load remote files if necessary
    root_remote = False
    if root_dir[0:5] == 'gs://':
        root_remote = True

    fs = None
    if root_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)
    
    do_dpc = len(channel) == 0
    # Load flatfield correction if necessary
    flatfield_left = None
    flatfield_right = None
    flatfield_fluorescence = None
    if do_dpc:
        correction_path = os.path.join(dest_dir, "correction")
        os.makedirs(correction_path, exist_ok=True)
        # check if files already exist
        if not os.path.exists(os.path.join(correction_path, "flatfield_left.npy")):
            if root_remote:
                fs.get(os.path.join(root_dir, "illumination correction", "left", "flatfield.npy"), os.path.join(correction_path, "flatfield_left.npy"))
            else:
                print("Left missing")
                raise FileNotFoundError
        if not os.path.exists(os.path.join(correction_path, "flatfield_right.npy")):
            if root_remote:
                fs.get(os.path.join(root_dir, "illumination correction", "right", "flatfield.npy"), os.path.join(correction_path, "flatfield_right.npy"))
            else:
                print("Right missing")
                raise FileNotFoundError
        if not os.path.exists(os.path.join(correction_path, "flatfield_fluorescence.npy")):
            if root_remote:
                fs.get(os.path.join(root_dir, "illumination correction", "fluorescent", "flatfield.npy"), os.path.join(correction_path, "flatfield_fluorescence.npy"))
            else:
                print("fluorescent missing")
                raise FileNotFoundError
        flatfield_left = np.load(os.path.join(correction_path, "flatfield_left.npy"))
        flatfield_right = np.load(os.path.join(correction_path, "flatfield_right.npy"))
        flatfield_fluorescence = np.load(os.path.join(correction_path, "flatfield_fluorescence.npy"))
    
    print("Reading image paths")
    # filter - only look for specified channel and cycle 0
    if do_dpc:
        channel = "_BF_LED_matrix_left_half"
    path = os.path.join(root_dir, exp_id,  "0", "**" + channel + '.' + ftype)
    print(path)
    if root_remote:
        allpaths = [p for p in fs.glob(path, recursive=True) if p.split('/')[-2] == '0']
    else:
        allpaths = list(glob.iglob(path, recursive=True))
    # remove duplicates
    imgpaths = list(dict.fromkeys(allpaths))
    # only get pos. 
    imgpaths = [i for i in imgpaths if "-" in i]
    print(str(len(imgpaths)) + " images to select from")
    savepath = os.path.join(dest_dir, exp_id)
    os.makedirs(savepath, exist_ok=True)

    selected = random.sample(imgpaths, n_rand)
    for impath in selected:
        print(impath)
        if do_dpc:
            img_path = impath
            if root_remote:
                im_left = np.array(imread_gcsfs(fs, img_path), dtype=float)
                img_path = impath.replace(channel, "_BF_LED_matrix_right_half")
                im_right = np.array(imread_gcsfs(fs, img_path), dtype=float)
                img_path = impath.replace(channel, "_Fluorescence_488_nm_Ex")
                im_flr = np.array(imread_gcsfs(fs, img_path), dtype=float)
            else:
                im_left = cv2.imread(img_path)
                img_path = impath.replace(channel, "_BF_LED_matrix_right_half")
                im_right = cv2.imread(img_path)
                img_path = impath.replace(channel, "_Fluorescence_488_nm_Ex")
                im_flr = cv2.imread(img_path)
            # do flatfield correction 
            im_left = im_left / (255 * flatfield_left)
            im_right = im_right / (255 * flatfield_right)
            im_flr = im_flr /  flatfield_fluorescence
            # generate dpc
            im_dpc = generate_dpc(im_left, im_right)
            im = np.stack((im_dpc,  np.zeros(im_dpc.shape), (im_flr).astype(np.uint8)), axis=2)
            im = im[400:-400,400:-400,:]
        else:
            if root_remote:
                im = np.array(imread_gcsfs(fs, impath), dtype=np.uint8)
            else:
                im = np.array(cv2.imread(impath), dtype=np.uint8)
        imshape = im.shape
        x = math.floor(imshape[0]/nsub)
        y = math.floor(imshape[1]/nsub)
        xslice = random.choice(range(nsub))
        yslice = random.choice(range(nsub))
        if do_dpc:
            im = im[x*xslice:(x*xslice + x), y*yslice:(y*yslice + y), :]
        else:
            im = im[x*xslice:(x*xslice + x), y*yslice:(y*yslice + y)]
        
        im = np.uint8(255 * np.array(im, dtype=np.float64)/float(np.max(im)))
        fname = os.path.join(savepath, "s_" + impath.split('/')[-1].rsplit('.', 1)[0] + ".png")
        fname = fname.replace(channel, "_dpc_flr")
        cv2.imwrite(fname, im)

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

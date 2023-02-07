import numpy as np
import glob
import os
import gcsfs
import cv2
import math
import random
import imageio

def main():
    # root_dir needs a trailing slash (i.e. /root/dir/)
    root_dir = '/home/prakashlab/Documents/kmarx/sickle_cell/SampleImages/'#'gs://octopi-codex-data-processing/' #"/home/prakashlab/Documents/kmarx/pipeline/tstflat/"# 'gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/'#
    dest_dir = 'segments/' # must be a local path
    exp_id   = "**/"
    channel =  "_DPC" # only run segmentation on this channel
    zstack  = 'f' # select which z to run segmentation on. set to 'f' to select the focus-stacked
    key = '/home/prakashlab/Documents/kmarx/malaria_deepzoom/deepzoom uganda 2022/uganda-2022-viewing-keys.json'
    ftype = 'npy'
    gcs_project = 'soe-octopi'
    n_rand = 9
    nsub = 6 # cut into a 3x3 grid and return a random selection
    get_rand(root_dir, dest_dir, exp_id, channel, zstack, n_rand, key, nsub, gcs_project, ftype)

def get_rand(root_dir, dest_dir, exp_id, channel, zstack, n_rand, key, nsub, gcs_project, ftype):
    # Load remote files if necessary
    root_remote = False
    if root_dir[0:5] == 'gs://':
        root_remote = True

    fs = None
    if root_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)

    print("Reading image paths")
    # filter - only look for specified channel and cycle 0
    path = root_dir + exp_id + "**" + channel + '.TIF'
    print(path)
    if root_remote:
        allpaths = [p for p in fs.glob(path, recursive=True) if p.split('/')[-2] == '0']
    else:
        allpaths = list(glob.iglob(path, recursive=True))
    # remove duplicates
    imgpaths = list(dict.fromkeys(allpaths))
    print(str(len(imgpaths)) + " images to select from")
    savepath = dest_dir + exp_id
    os.makedirs(savepath, exist_ok=True)

    selected = random.sample(imgpaths, n_rand)
    for impath in selected:
        print(impath)
        # fs.get(impath, savepath + impath.split('/')[-3] + "_s_" + impath.split('/')[-1])
        if root_remote:
            im = np.array(imread_gcsfs(fs, impath), dtype=np.uint8)
        else:
            im = np.array(cv2.imread(impath), dtype=np.uint8)
        shape = im.shape
        x = math.floor(shape[0]/nsub)
        y = math.floor(shape[1]/nsub)
        xslice = random.choice(range(nsub))
        yslice = random.choice(range(nsub))
        im = im[x*xslice:(x*xslice + x), y*yslice:(y*yslice + y)]
        
        im = im - np.min(im)
        im = np.uint8(255 * np.array(im, dtype=np.float64)/float(np.max(im)))

        fname = savepath + "s_" + impath.split('/')[-1]

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

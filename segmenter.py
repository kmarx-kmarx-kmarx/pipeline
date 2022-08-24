import numpy as np
from cellpose import models
from cellpose import models, io
import glob
import os
import gcsfs
import imageio
import xarray as xr


def main():
    # root_dir needs a trailing slash (i.e. /root/dir/)
    is_zarr = True
    root_dir = "/home/prakashlab/Documents/kmarx/pipeline/ztest/"# 'gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/'#
    exp_id   = "20220601_20x_75mm"
    channel =  0 # only run segmentation on this channel; 0 is first channel, 1 is second, etc.
    cpmodel = "gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/cellposemodel"
    channels = [0,0] # grayscale only
    key = '/home/prakashlab/Documents/fstack/codex-20220324-keys.json'
    use_gpu = True
    gcs_project = 'soe-octopi'
    run_seg(is_zarr, root_dir, exp_id, channel, cpmodel, channels, key, use_gpu, gcs_project)

def run_seg(is_zarr, root_dir, exp_id, channel, cpmodel, channels, key, use_gpu, gcs_project):
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
    # filter - only look for specified channel
    if is_zarr:
        path = root_dir + exp_id + "/**/**/**" + exp_id + '.zarr'
        print(path)
        if root_remote:
            allpaths = [p for p in fs.glob(path, recursive=True)]
        else:
            allpaths = [p for p in glob.iglob(path, recursive=True)]
        allpaths = list(dict.fromkeys(allpaths))
        allpaths.sort()
        for i in allpaths:
            print(i)
        # get the shape of a zarr
        imgpaths = []


        
    # else:
    #     path = root_dir + exp_id + "/**/**/**/" + str(channel) + '**.png'
    #     print(path)
    #     if root_remote:
    #         allpaths = [p for p in fs.glob(path, recursive=True)]
    #     else:
    #         allpaths = [p for p in glob.iglob(path, recursive=True)]
    #     # remove duplicates
    #     imgpaths = list(dict.fromkeys(allpaths))
    #     imgpaths.sort()
    #     imgpaths = np.array(imgpaths)
    #     print("Reading images")
    #     # load images
    #     if root_remote:
    #         imgs = [imread_gcsfs(fs, path) for path in imgpaths]
    #     else:
    #         imgs = [io.imread(path) for path in imgpaths]

    # print("Starting cellpose")
    # # start cellpose
    # model = models.CellposeModel(gpu=use_gpu, pretrained_model=cpmodel)
    # print("Starting segmentation")

    # placeholder = "./placeholder.png"

    # print(str(len(imgs)) + " images to segment")

    # # segment one at a time - gpu bottleneck
    # for idx, im in enumerate(imgs):
    #     print(idx)
    #     imlist  = [im]
    #     masks, flows, styles = model.eval(imlist, diameter=None, channels=channels)
    #     diams = 0
    #     if root_remote:
    #         savepath = placeholder
    #     else:
    #         savepath = imgpaths[idx]
    #     # actually run the segmentation
    #     io.masks_flows_to_seg(imlist, masks, flows, diams, [savepath], channels)

    #     # move the .npy to remote if necessary
    #     if root_remote:
    #         # generate the local and remote path names
    #         savepath = savepath.rsplit(".", 1)[0] + "_seg.npy"
    #         rpath = imgpaths[idx].rsplit(".", 1)[0] + "_seg.npy"
    #         fs.put(savepath, rpath)
    #         os.remove(savepath)

    # if model_remote:
    #     os.remove(modelpath)

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
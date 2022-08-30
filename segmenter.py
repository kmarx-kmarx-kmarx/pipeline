import numpy as np
from cellpose import models
from cellpose import models, io
import glob
import os
import gcsfs
import imageio
from natsort import natsorted
import xarray as xr

def main():
    is_zarr = True
    # root_dir needs a trailing slash (i.e. /root/dir/)
    root_dir = '/media/prakashlab/T7/'#'gs://octopi-codex-data-processing/' #"/home/prakashlab/Documents/kmarx/pipeline/tstflat/"# 'gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/'#
    exp_id   = "20220823_20x_PBMC_2"
    channel =  "Fluorescence_405_nm_Ex" # only run segmentation on this channel
    zstack  = 'f' # select which z to run segmentation on. set to 'f' to select the focus-stacked
    cpmodel = "./cpmodel_20220827"
    channels = [0,0] # grayscale only
    key = '/home/prakashlab/Documents/fstack/codex-20220324-keys.json'
    use_gpu = True
    gcs_project = 'soe-octopi'
    run_seg(is_zarr, root_dir, exp_id, channel, zstack, cpmodel, channels, key, use_gpu, gcs_project)

def run_seg(is_zarr, root_dir, exp_id, channel, zstack, cpmodel, channels, key, use_gpu, gcs_project):
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
    # filter - only look for specified channel, z, and cycle 0
    if is_zarr:
        path = root_dir + exp_id + "/**/**/**" + exp_id + '.zarr'
        print(path)
        if root_remote:
            allpaths = [p for p in fs.glob(path, recursive=True)]
        else:
            allpaths = [p for p in glob.iglob(path, recursive=True)]
        segpaths = list(dict.fromkeys(allpaths))
        segpaths.sort()
        imgpaths = [p.rsplit('/', 1)[0] + '/' + p.split('/')[-1].split('.')[-2] + '.png' for p in segpaths]
        print(str(len(segpaths)) + " images to segment")
    else:
        path = root_dir + exp_id + "**/0/**_" + zstack + "_" + channel + '.png'
        print(path)
        if root_remote:
            allpaths = [p for p in fs.glob(path, recursive=True) if p.split('/')[-2] == '0']
        else:
            allpaths = [p for p in glob.iglob(path, recursive=True) if p.split('/')[-2] == '0']
        # remove duplicates
        imgpaths = list(dict.fromkeys(allpaths))
        imgpaths = np.array(natsorted(imgpaths))
        # only look at ch 0
        ch0 = imgpaths[0].split('/')[-3]
        segpaths = [path for path in imgpaths if ch0 in path]
        print(str(len(segpaths)) + " images to segment")
    print("Starting cellpose")
    # start cellpose
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=cpmodel)
    print("Starting segmentation")

    placeholder = "./placeholder."

    # segment one at a time - gpu bottleneck
    for idx, impath in enumerate(segpaths):
        print(str(idx) + ": " + impath)
        if is_zarr:
            local = impath
            if root_remote:
                local = placeholder + 'zarr'
                fs.get(impath, local)
            im = xr.open_zarr(local).to_array()
            # get first cycle, first channel
            im = im[0, 0, 0, :, :]
        else:
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
            savepath = placeholder + 'png'
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

        if is_zarr and root_remote:
            os.remove(local)
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
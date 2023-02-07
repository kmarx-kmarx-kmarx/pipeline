import numpy as np
from cellpose import models
from cellpose import models, io
import glob
import os	
import gcsfs
import imageio
from natsort import natsorted
import time

def main():
    # root_dir needs a trailing slash (i.e. /root/dir/)
    root_dir = "/home/prakashlab/Documents/kmarx/sickle_cell/SampleImages/"#'gs://octopi-codex-data-processing/' #"/home/prakashlab/Documents/kmarx/pipeline/tstflat/"# 'gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/'#
    exp_id   = ["AA/", "AS/", "SS/"]
    channel =  "_DPC" # only run segmentation on this channel
    zstack  = 'f' # select which z to run segmentation on. set to 'f' to select the focus-stacked
    cpmodel = "/home/prakashlab/Documents/kmarx/sickle_cell/pipeline/cp_dpc_new"#"/home/prakashlab/Documents/kmarx/pipeline/cpmodel_20220827"#"/home/octopi-codex/Documents/pipeline_test/subsets/20220811_10x_zstacks/models/cellpose_residual_on_style_on_concatenation_off_20220811_10x_zstacks_2022_09_18_11_18_32.611351"
    channels = [0,0] # grayscale only
    key = "/home/prakashlab/Documents/fstack/codex-20220324-keys.json"#'/home/prakashlab/Documents/kmarx/malaria_deepzoom/deepzoom uganda 2022/uganda-2022-viewing-keys.json'
    use_gpu = True
    segment_all = True
    gcs_project = 'soe-octopi'
    t0 = time.time()
    for ids in exp_id:
        run_seg(root_dir, ids, channel, zstack, cpmodel, channels, key, use_gpu, segment_all, gcs_project)
    t1 = time.time()
    print(t1-t0)

def run_seg(root_dir, exp_id, channel, zstack, cpmodel, channels, key, use_gpu, segment_all, gcs_project):
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
    path = root_dir + exp_id + "**" + channel + '.TIF'
    print(path)
    if root_remote:
        allpaths = [p for p in fs.glob(path, recursive=True)]
    else:
        allpaths = list(glob.glob(path, recursive=True))
    # remove duplicates
    imgpaths = list(dict.fromkeys(allpaths))
    imgpaths = np.array(natsorted(imgpaths))
    if not segment_all:
        ch0 = imgpaths[0].split('/')[-3]
        ch1 = imgpaths[1].split('/')[-3]
        segpaths = [path for path in imgpaths if ch0 in path]
        segpaths_alt = [path for path in imgpaths if ch1 in path]
    else:
        segpaths = imgpaths
    print(str(len(segpaths)) + " images to segment")

    print("Starting cellpose")
    # start cellpose
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=cpmodel)
    print("Starting segmentation")

    placeholder = "./placeholder.png"
    dest = root_dir + exp_id + "segmentation/"
    os.makedirs(dest, exist_ok=True)

    # segment one at a time - gpu bottleneck
    for idx, impath in enumerate(segpaths):
        print(str(idx) + ": " + impath)
        if root_remote:
            im = np.array(imread_gcsfs(fs, impath), dtype=np.uint8)
        else:
            im = np.array(io.imread(impath), dtype=np.uint8)
        # normalize
        im = im - np.min(im)
        im = np.uint8(255 * np.array(im, dtype=np.float64)/float(np.max(im)))
        # run segmentation
        masks, flows, styles = model.eval(im, diameter=None, channels=channels)
        # If we didn't get any cells, try again
        if np.max(masks) == 0 and not segment_all:
            impath = segpaths_alt[idx]
            print("No cells found. Trying again with a different cycle")
            print(str(idx) + ": " + impath)
            if root_remote:
                im = np.array(imread_gcsfs(fs, impath), dtype=np.uint8)
            else:
                im = np.array(io.imread(impath), dtype=np.uint8)
            # normalize
            im = im - np.min(im)
            im = np.uint8(255 * np.array(im, dtype=np.float64)/float(np.max(im)))
            # run segmentation
            masks, flows, styles = model.eval(im, diameter=None, channels=channels)
        elif np.max(masks) == 0 and segment_all:
            print("No cells detected! Next view")
            continue
        diams = 0
        if root_remote:
            savepath = placeholder
        else:
            if segment_all:
                savepath = dest + impath.split('/')[-3] + '/' + impath.split('/')[-2] + '/' 
            else:
                savepath = dest + "first" + '/' + impath.split('/')[-2] + '/' 
            os.makedirs(savepath, exist_ok=True)
            savepath = savepath + impath.split('/')[-1]
        # save the data
        io.masks_flows_to_seg(im, masks, flows, diams, savepath, channels)

        # move the .npy to remote if necessary
        if root_remote:
            # generate the local and remote path names
            savepath = savepath.rsplit(".", 1)[0] + "_seg.npy"
            rpath ="gs://" + imgpaths[idx].rsplit(".", 1)[0] + "_seg.npy"
            fs.put(savepath, rpath)
            print(rpath)
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

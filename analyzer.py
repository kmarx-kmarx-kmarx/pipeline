import glob
import numpy as np
import cv2
import re
import pandas as pd
import gcsfs
import imageio
import os
from natsort import natsorted
import xarray as xr

def main():
    is_zarr = True
    # Cycle indices are 0-12, we can choose a subset of the cycles to analyze
    start_idx = 0 #2
    end_idx   = 14 #11
    # 4 channels
    n_ch      = 4
    # How many pixels around the mask to expand
    expansion = 9   
    # root_dir needs a trailing slash (i.e. /root/dir/)
    root_dir = '/media/prakashlab/T7/'#'gs://octopi-codex-data-processing/' #"/home/prakashlab/Documents/kmarx/pipeline/tstflat/"# 'gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/'#
    exp_id   = "20220823_20x_PBMC_2/"
    zstack  = 'f' # select which z to run segmentation on. set to 'f' to select the focus-stacked
    channel =  "Fluorescence_405_nm_Ex" # use only this channel as masks
    key = '/home/prakashlab/Documents/fstack/codex-20220324-keys.json'
    gcs_project = 'soe-octopi'
    out = "/media/prakashlab/T7/" + exp_id + "/meanbright_" + str(expansion) + ".csv"
    
    run_analysis(is_zarr, start_idx, end_idx, n_ch, zstack, expansion, root_dir, exp_id, channel, key, gcs_project, out)

def run_analysis(is_zarr, start_idx, end_idx, n_ch, zstack, expansion, root_dir, exp_id, channel, key, gcs_project, out):
    root_remote = False
    if root_dir[0:5] == 'gs://':
        root_remote = True
    out_remote = False
    out_placeholder = "temp.csv"
    out_path = out
    if out[0:5] == 'gs://':
        out_remote = True
        out_path = out_placeholder
    fs = None
    if root_remote or out_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)

    print("Reading .npy paths")
    if is_zarr:
        path = root_dir + exp_id + "**/**/**_seg.npy"
    else:
        path = root_dir + exp_id + "**/0/**_" + zstack + "_" + channel + '_seg.npy'
    print(path)
    if root_remote:
        allpaths = [p for p in fs.glob(path, recursive=True)]
    else:
        allpaths = [p for p in glob.iglob(path, recursive=True)]
    # remove duplicates
    allpaths = list(dict.fromkeys(allpaths))
    allpaths = natsorted(allpaths)
    npypaths = np.array(allpaths)
    # only the first cycle is segmented - nothing more to do

    # repeat to get png paths
    print("Reading image paths")
    if is_zarr:
        path = root_dir + exp_id + "**/**/**.zarr"
    else:
        path = root_dir + exp_id + "**/0/**_" + zstack  + '**.png'
    print(path)
    if root_remote:
        allpaths = [p for p in fs.glob(path, recursive=True)]
    else:
        allpaths = [p for p in glob.iglob(path, recursive=True)]
    # remove duplicates
    allpaths = list(dict.fromkeys(allpaths))
    allpaths = natsorted(allpaths)
    allpaths = np.array(allpaths)
    if is_zarr:
        print(str(n_ch * len(allpaths) * (end_idx - start_idx + 1)) + " images to analyze")
    else:
        # remove images out of cycle bounds
        all_cycles = natsorted(list(dict.fromkeys([i.split('/')[-3] for i in allpaths])))
        target_cycles = all_cycles[start_idx:end_idx+1]
        pngpaths = [p for p in allpaths if p.split('/')[-3] in target_cycles]
        print(str(len(pngpaths)) + " images to analyze")

    # make a dataframe
    header = ['i', 'j', 'x', 'y', 'sz_msk', 'sz_nuc']
    for cy in range(start_idx, end_idx + 1,1):
        for ch in range(0, n_ch):
            header.append(str(cy) + "_" + str(ch))
    print("header: " + str(len(header)))
    # process one at a time
    placeholder = "./placeholder"
    print(str(len(npypaths)) + " masks")
    for idx, path in enumerate(npypaths):
        # make a fresh df
        df = pd.DataFrame(columns=header)
        #   1 row for each cell
        #   1 column for each cycle x channel
        print("Loading mask " + path)
        if is_zarr:
            zpath = path.rsplit('_', 1)[0] + '.zarr'
            local = zpath
            if root_remote:
                local = placeholder + '.zarr'
                fs.get(zpath, local)
            imgs = np.array(xr.open_zarr(local).to_array())
            splitpath = path.split('/')
            i = splitpath[-3]
            j = splitpath[-2]
        else:
            splitpath = path.split("/")
            cy = splitpath[-3]
            splitpath = splitpath[-1].split('_')
            ch = int(splitpath[4])
            j  = int(splitpath[1])
            i  = int(splitpath[0])
            # only get images 
            pattern = "\/" + str(i) + "_" + str(j) + "_" + zstack + ".*\.png"
            imgpath = [p for p in pngpaths if re.search(pattern, p)]
            imgpath = natsorted(imgpath)
            print(str(len(imgpath)) + " images in this (i,j)")
            # preload each image
            if root_remote:
                imgs = np.array([imread_gcsfs(fs, path) for path in imgpath])
            else:
                imgs = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in imgpath])

        # get the mask
        if root_remote:
            fs.get(path, placeholder + '_seg.npy')
            nppath = placeholder + '_seg.npy'
        else:
            nppath = path
        reading = np.load(nppath, allow_pickle=True).item()
        masks = reading['masks']
        # expand the mask (capture more than just the nucleus)
        kernel = np.zeros((expansion,expansion),np.uint8)
        kernel = cv2.circle(kernel, (int(expansion/2), int(expansion/2)), int(expansion/2), (255,255,255), -1)
        dilation = cv2.dilate(masks,kernel,iterations = 1)
        
        # for each cell, calculate the mean pixel value
        print("mask " + str(idx) + " has " + str(np.max(masks)) + " cells")
        for l in range(np.max(masks)):
            # Each cell in the mask gets its own row in the CSV
            # mask an individual cell
            cell_mask = dilation == (l+1)
            # create a row for holding cell data
            row = [str(i), str(j)]
            # find the coordinates of the cell
            contours, hierarchy = cv2.findContours(np.array(cell_mask, dtype="uint8"), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            M = cv2.moments(contours[0])
            # avoid divide-by-zero errors
            try:
                x = round(M['m10'] / M['m00'])
            except:
                x = -1
            try:
                y =  round(M['m01'] / M['m00'])
            except:
                y = -1
            coord = [x, y]
            row = row + coord
            # get mean brightness
            
            # for each image find the avg brightness around that cell
            if is_zarr:
                # 2D array w/ avg for each cycle, channel
                cell_mask
                avg = np.mean(imgs[0,:,:,:,:], axis=2)
                avg = np.mean(avg, axis=2)
                # reshape into list
                
            else:
                brightness = np.zeros(n_ch * (end_idx - start_idx + 1))
                for m, im in enumerate(imgs):
                    avg = np.mean(im[cell_mask])
                    brightness[m]   = avg
            # write the row
    #         sz = [str(np.sum(cell_mask)), str(np.sum(masks==(l+1)))]
    #         brightness = [str(b) for b in brightness]
    #         row = row + sz + brightness
    #         df.loc[len(df.index)] = row
    #     # append to csv
    #     print("writing")
    #     df.to_csv(out_path, mode='a')
    #     # delete .npy if remote
    #     if root_remote:
    #         os.remove(placeholder + '_seg.npy')
    # # move the csv to remote
    # if out_remote:
    #     fs.put(out_placeholder, out)
    #     os.remove(out_placeholder)


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
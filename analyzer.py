import glob
import numpy as np
import cv2
import re
import pandas as pd
import gcsfs
import imageio
import os

def main():
    # Cycle indices are 0-11, we can choose a subset of the cycles to analyze
    start_idx = 0 #2
    end_idx   = 1 #11
    # 4 channels
    n_ch      = 4
    # How many pixels around the mask to expand
    expansion = 7   
    # root_dir needs a trailing slash (i.e. /root/dir/)
    root_dir = "/home/prakashlab/Documents/newcodex/"#'gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/' #"/home/prakashlab/Documents/kmarx/pipeline/test/" #'gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/'
    exp_id   = "data for processing/"
    channel =  "Fluorescence_405_nm_Ex" # use only this channel as masks
    key = '/home/prakashlab/Documents/fstack/codex-20220324-keys.json'
    gcs_project = 'soe-octopi'
    out = "/home/prakashlab/Documents/newcodex/meanbright_" + str(expansion) + ".csv"
    
    run_analysis(start_idx, end_idx, n_ch, expansion, root_dir, exp_id, channel, key, gcs_project, out)

def run_analysis(start_idx, end_idx, n_ch, expansion, root_dir, exp_id, channel, key, gcs_project, out):
    root_remote = False
    if root_dir[0:5] == 'gs://':
        root_remote = True
    out_remote = False
    out_placeholder = "./temp.csv"
    out_path = out
    if out[0:5] == 'gs://':
        out_remote = True
        out_path = out_placeholder
    fs = None
    if root_remote or out_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)

    print("Reading .npy paths")
    path = root_dir + exp_id  + "**/**/**/**" + channel + "_seg.npy"
    print(path)
    if root_remote:
        allpaths = [p for p in fs.glob(path, recursive=True)]
    else:
        allpaths = [p for p in glob.iglob(path, recursive=True)]
    # remove duplicates
    allpaths = list(dict.fromkeys(allpaths))
    allpaths.sort()
    allpaths = np.array(allpaths)

    # get cycle from image paths
    c = np.array([int(i.split('/')[-2]) for i in allpaths])
    # only consider the earliest cycle
    npypaths = allpaths[c == start_idx]

    # # repeat to get png paths
    print("Reading .png paths")
    path = root_dir + exp_id  + "**/**/**/**.png"
    print(path)
    if root_remote:
        allpaths = [p for p in fs.glob(path, recursive=True)]
    else:
        allpaths = [p for p in glob.iglob(path, recursive=True)]
    # remove duplicates
    allpaths = list(dict.fromkeys(allpaths))
    allpaths.sort()
    allpaths = np.array(allpaths)
    # remove images out of cycle bounds
    c = np.array([int(i.split('/')[-2]) for i in allpaths])
    pngpaths = allpaths[(c >= start_idx) * (c <= end_idx)]

    # make a dataframe
    header = ['i', 'j', 'x', 'y', 'sz_msk', 'sz_nuc']
    for cy in range(start_idx, end_idx + 1,1):
        for ch in range(0, n_ch):
            header.append(str(cy) + "_" + str(ch))

    # process one at a time
    placeholder = "./placeholder_seg.npy"
    print(str(len(npypaths)) + " masks")
    for idx, path in enumerate(npypaths):
        # make a fresh df
        df = pd.DataFrame(columns=header)
        #   1 row for each cell
        #   1 column for each cycle x channel
        print("Loading mask " + path)
        splitpath = path.split("/")
        ch = int(splitpath[-1].split("_")[0])
        cy = int(splitpath[-2])
        j  = int(splitpath[-3])
        i  = int(splitpath[-4])
        pattern = '\/' + str(i) + '\/' + str(j) + '\/\d+\/\d+_.*\.png'
        imgpath = [p for p in pngpaths if re.search(pattern, p)]
        imgpath.sort()
        # preload each image
        if root_remote:
            imgs = np.array([imread_gcsfs(fs, path) for path in imgpath])
        else:
            imgs = np.array([cv2.imread(path, cv2.IMREAD_GRAYSCALE) for path in imgpath])

        # get the mask
        if root_remote:
            fs.get(path, placeholder)
            nppath = placeholder
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
            brightness = np.zeros(len(imgpath))
            # for each image find the avg brightness around that cell
            for m, im in enumerate(imgs):
                avg = np.mean(im[cell_mask])
                brightness[m]   = avg
            # write the row
            sz = [str(np.sum(cell_mask)), str(np.sum(masks==(l+1)))]
            brightness = [str(b) for b in brightness]
            row = row + sz + brightness
            df.loc[len(df.index)] = row
        # append to csv
        df.to_csv(out_path, mode='a')
        # delete .npy if remote
        if root_remote:
            os.remove(placeholder)
    # move the csv to remote
    if out_remote:
        fs.put(out_placeholder, out)
        os.remove(out_placeholder)


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
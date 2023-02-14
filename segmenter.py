import numpy as np
from cellpose import models, utils
import glob
import os	
import gcsfs
import imageio
import cv2
from dpc_overlay import generate_dpc
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.itertools import product
import logging
import json

def main():
    logging.basicConfig(filename='example.log')
    debug = False
    root_dir = 'gs://octopi-tb-data/20221002'
    dest_dir = 'gs://octopi-tb-data-processing/20230214'
    exp_id   = ['SP1+_2022-10-02_20-10-50.529345', 'SP1-_2022-10-02_20-23-56.661506', 'SP2+_2022-10-02_20-41-41.247131', 'SP2-_2022-10-02_20-58-10.547699', 'SP3+_2022-10-02_21-13-35.929780', 'SP3-_2022-10-02_21-31-39.783478', 'SP4+_2022-10-02_21-47-12.117750', 'SP4-_2022-10-02_21-59-26.387251', 'SP5+_2022-10-02_22-16-53.962897', 'SP5-_2022-10-02_22-29-35.714940']
    channel =  ""
    cpmodel = "/home/prakashlab/Documents/kmarx/tb_segment/pipeline/segments/**/train/models/cellpose_residual_on_style_on_concatenation_off_train_2023_02_10_10_52_15.693300_epoch_4001"
    channels = [1,3] # red and blue
    key = '/home/prakashlab/Documents/kmarx/tb_key.json'
    use_gpu = True
    gcs_project = 'soe-octopi'
    correction_path = "/home/prakashlab/Documents/kmarx/tb_segment/pipeline/segments/correction"
    center_crop = 400 # crop away this many pixels from each edge
    dilation_sz = 2 # amount to dilate to merge adjacent cells
    for ids in tqdm(exp_id):
        run_seg(debug, root_dir, dest_dir, ids, channel, cpmodel, channels, key, use_gpu, gcs_project, correction_path, center_crop, dilation_sz)


def run_seg(debug, root_dir, dest_dir, exp_id, channel, cpmodel, channels, key, use_gpu, gcs_project, correction_path, center_crop, dilation_sz):
    # Load remote files if necessary
    root_remote = False
    if root_dir[0:5] == 'gs://':
        root_remote = True

    dest_remote = False
    if dest_dir[0:5] == 'gs://':
        dest_remote = True

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
    do_dpc = len(channel) == 0
    # Load flatfield correction if necessary
    flatfield_left = None
    flatfield_right = None
    flatfield_fluorescence = None
    if do_dpc:
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
    # load acquisition params
    if root_remote:
        json_file = fs.cat(os.path.join(root_dir, exp_id, "acquisition parameters.json"))
        acquisition_params = json.loads(json_file)
    else:
        acquisition_params = json.loads(os.path.join(root_dir, exp_id, "acquisition parameters.json"))

    print(str(acquisition_params['Nx'] * acquisition_params['Ny'] * acquisition_params['Nz']) + " images to segment")
    xrng = range(acquisition_params['Nx'])
    yrng = range(acquisition_params['Ny'])
    zrng = range(acquisition_params['Nz'])
    if debug:
        xrng = min(xrng, 2)
        yrng = min(yrng, 2)
        zrng = min(zrng, 2)
    # start cellpose
    model = models.CellposeModel(gpu=use_gpu, pretrained_model=cpmodel)

    savepath = os.path.join(dest_dir, exp_id)
    if not dest_remote:
        os.makedirs(savepath, exist_ok=True)
    
    # Create dataframes for number of cells in the view and the blob info
    n_cell_df = pd.DataFrame(columns=['x_id', 'y_id', 'z_id', "n_cell"])
    blob_df = pd.DataFrame(columns=['x_id', 'y_id', 'z_id', "blob_x", "blob_y", "blob_w", "blob_h", "blob_x_center", "blob_y_center"])
    # segment one at a time - gpu bottleneck
    for x_id, y_id, z_id in product(xrng, yrng, zrng):
        x_id = str(x_id)
        y_id = str(y_id)
        z_id = str(z_id)
        if do_dpc:
            try:
                if root_remote:
                    img_path = os.path.join(root_dir, exp_id, '0', f"{x_id}_{y_id}_{z_id}_BF_LED_matrix_left_half.bmp")
                    im_left = np.array(imread_gcsfs(fs, img_path), dtype=float)
                    img_path = img_path.replace("BF_LED_matrix_left_half", "BF_LED_matrix_right_half")
                    im_right = np.array(imread_gcsfs(fs, img_path), dtype=float)
                    img_path = img_path.replace("BF_LED_matrix_right_half", "Fluorescence_488_nm_Ex")
                    im_flr = np.array(imread_gcsfs(fs, img_path), dtype=float)
                else:
                    img_path = os.path.join(root_dir, exp_id, '0', f"{x_id}_{y_id}_{z_id}_BF_LED_matrix_left_half.bmp")
                    im_left = cv2.imread(img_path)
                    img_path = img_path.replace("BF_LED_matrix_left_half", "BF_LED_matrix_right_half")
                    im_right = cv2.imread(img_path)
                    img_path = img_path.replace("BF_LED_matrix_right_half", "Fluorescence_488_nm_Ex")
                    im_flr = cv2.imread(img_path)
            except:
                img_path = os.path.join(root_dir, exp_id, '0', f"{x_id}_{y_id}_{z_id}_BF_LED_matrix_left_half.bmp")
                logging.warning(img_path)
                img_path = img_path.replace("BF_LED_matrix_left_half", "BF_LED_matrix_right_half")
                logging.warning(img_path)
                img_path = img_path.replace("BF_LED_matrix_right_half", "Fluorescence_488_nm_Ex")
                logging.warning(img_path)
                continue
            # do flatfield correction 
            im_left = im_left / (255 * flatfield_left)
            im_right = im_right / (255 * flatfield_right)
            im_flr = im_flr /  flatfield_fluorescence
            # generate dpc
            im_dpc = generate_dpc(im_left, im_right)
            im = np.stack((im_dpc,  np.zeros(im_dpc.shape), (im_flr).astype(np.uint8)), axis=2)
        else:
            img_path = os.path.join(root_dir, exp_id, 0, f"{x_id}_{y_id}_{z_id}_{channel}.bmp")
            if root_remote:
                im = np.array(imread_gcsfs(fs, impath), dtype=np.uint8)
            else:
                im = np.array(cv2.imread(impath), dtype=np.uint8)
        # normalize
        im = im - np.min(im)
        im = np.uint8(255 * np.array(im, dtype=np.float64)/float(np.max(im)))
        # run segmentation
        masks, flows, styles = model.eval(im, diameter=None, channels=channels)
        # crop the outside - get rid of edge effects
        masks = masks[center_crop:-center_crop, center_crop:-center_crop]
        # get number of cells
        n_cells = np.max(masks)
        n_cell_df.loc[len(n_cell_df)] = [x_id, y_id, z_id, n_cells]
        # make binary mask with outlines
        outlines = masks * utils.masks_to_outlines(masks)
        bin_mask = (masks  > 0) * 1.0
        outlines = (outlines  > 0) * 1.0
        bin_mask = (bin_mask * (1.0 - outlines) * 255).astype(np.uint8)
        flat_mask = np.zeros((masks.shape[0] + 2 * center_crop, masks.shape[1] + 2 * center_crop))
        flat_mask[center_crop:-center_crop, center_crop:-center_crop] = bin_mask
        # find blobs - dilate the flat mask to merge adjacet cells, then find contours
        shape = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(shape, (2 * dilation_sz + 1, 2 * dilation_sz + 1), (dilation_sz, dilation_sz))
        dilate_mask = np.array(cv2.dilate(bin_mask, element))
        contours, __ = cv2.findContours(dilate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # for each contour, find center and bounding box
        for cnt in contours:
            x,y,w,h = cv2.boundingRect(cnt)
            x = x + center_crop
            y = y + center_crop
            x_center = int(x + w/2)
            y_center = int(y + h/2)
            # add info to dataframe
            blob_df.loc[len(blob_df)] = [x_id, y_id, z_id, x, y, w, h, x_center, y_center]

        # save all data
        impath = f"{savepath}/{x_id}_{y_id}_{z_id}_mask.bmp"
        if dest_remote:
            img_bytes = cv2.imencode('.bmp', flat_mask)[1]
            with fs.open(impath, 'wb') as f:
                f.write(img_bytes)
        else:
            cv2.imwrite(impath,flat_mask)
        if debug and not dest_remote:
            impath = f"{savepath}/{x_id}_{y_id}_{z_id}_image.bmp"
            cv2.imwrite(impath,im)
        
    # save dataframes
    blobpath = f"{savepath}/cell_group_locations.csv"
    ncellpath = f"{savepath}/n_cells.csv"
    if dest_remote:
        with fs.open(blobpath, 'w') as f:
            blob_df.to_csv(f)
        with fs.open(ncellpath, 'w') as f:
            n_cell_df.to_csv(f)
    else:
        blob_df.to_csv(blobpath)
        n_cell_df.to_csv(ncellpath)

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

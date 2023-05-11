import numpy as np
from cellpose import models, utils
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

'''
This file reads fluorescence images ("Fluorescence_{flr_nm}_nm_Ex.bmp" suffix) from a folder and
saves binary masks for each view and cell locations and bounding boxes for each cell
detected in each view. 
'''


def main():
    logging.basicConfig(filename='example.log')
    debug = False
    root_dir = 'gs://octopi-tb-data/20230303'
    dest_dir = 'gs://octopi-tb-data-processing/20230303_masks'
    exp_id = ['T1_1_75ms_2023-03-03_19-39-14.508916', 'T1_2_75ms_2023-03-05_00-56-29.771155', 'T1_3_75ms_2023-03-05_01-49-57.857155', 'T1f_1_100ms_2023-03-03_17-11-36.080318', 'T1f_2_200ms_2023-03-04_20-51-7.819815', 'T1f_2_50ms_2023-03-04_21-06-45.839723', 'T1f_3_50ms_2023-03-04_21-26-3.572300', 'T2_1_200ms_2023-03-05_00-03-14.655334', 'T2_2_100ms_2023-03-05_00-22-38.274591', 'T2_3_100ms_2023-03-05_00-33-56.211684', 'T2f_1_100ms_2023-03-03_17-36-23.426629', 'T2f_2_200ms_2023-03-03_22-40-15.371414', 'T2f_3_200ms_2023-03-04_20-26-12.786909', 'T3_1_200ms_2023-03-04_22-35-30.525396', 'T3_2_200ms_2023-03-04_23-05-5.200717', 'T3_3_200ms_2023-03-04_23-18-17.838165', 'T3f_1_200ms_2023-03-04_18-26-18.577325', 'T3f_2_200ms_2023-03-04_18-44-0.396752', 'T3f_3_200ms_2023-03-04_19-49-17.419955', 'UT1F_200ms_2023-03-03_16-36-8.995969', 'UT1_75ms_2023-03-03_18-23-20.715225', 'UT2F_100ms_2023-03-03_17-24-44.929085', 'UT2_50ms_2023-03-04_23-31-39.657212', 'UT3F_100ms_2023-03-04_17-55-48.325422', 'UT3_50ms_2023-03-04_22-16-57.750483']#['SP1+_2022-10-02_20-10-50.529345', 'SP1-_2022-10-02_20-23-56.661506', 'SP2+_2022-10-02_20-41-41.247131', 'SP2-_2022-10-02_20-58-10.547699', 'SP3+_2022-10-02_21-13-35.929780', 'SP3-_2022-10-02_21-31-39.783478', 'SP4+_2022-10-02_21-47-12.117750', 'SP4-_2022-10-02_21-59-26.387251', 'SP5+_2022-10-02_22-16-53.962897', 'SP5-_2022-10-02_22-29-35.714940']
    cpmodel = "cellpose_tb_model"
    channels = [3, 0]  # red only
    key = '/home/prakashlab/Documents/keys/tb_key.json'
    use_gpu = True
    gcs_project = 'soe-octopi'
    correction_path = "illumination correction tb"
    flr_nm = "405"
    center_crop = 0  # crop away this many pixels from each edge
    dilation_sz = -1  # amount to dilate to merge adjacent cells, set negative to disable
    run_seg(debug, root_dir, dest_dir, exp_id, cpmodel, channels, key,
            use_gpu, gcs_project, correction_path, flr_nm, center_crop, dilation_sz)


def run_seg(debug, root_dir, dest_dir, exp_ids, cpmodel, channels, key, use_gpu, gcs_project, correction_path, flr_nm, center_crop, dilation_sz):
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
    if model_remote or root_remote or dest_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project, token=key)

    modelpath = "./cellpose_temp"
    if model_remote:
        fs.get(cpmodel, modelpath)
        cpmodel = modelpath

    # Load flatfield correction if necessary
    os.makedirs(correction_path, exist_ok=True)
    # check if files already exist
    if not os.path.exists(os.path.join(correction_path, "flatfield_fluorescence.npy")):
        if root_remote:
            fs.get(os.path.join(root_dir, "illumination correction", "fluorescent",
                   "flatfield.npy"), os.path.join(correction_path, "flatfield_fluorescence.npy"))
        else:
            print("fluorescent missing")
            raise FileNotFoundError
    flatfield_fluorescence = np.load(os.path.join(
        correction_path, "flatfield_fluorescence.npy"))
    for exp_id in tqdm(exp_ids):
        # load acquisition params
        if root_remote:
            json_file = fs.cat(os.path.join(
                root_dir, exp_id, "acquisition parameters.json"))
            acquisition_params = json.loads(json_file)
        else:
            acquisition_params = json.loads(os.path.join(
                root_dir, exp_id, "acquisition parameters.json"))

        if debug:
            acquisition_params['Ny'] = min(acquisition_params['Ny'], 2)
            acquisition_params['Nx'] = min(acquisition_params['Nx'], 2)
            acquisition_params['Nz'] = min(acquisition_params['Nz'], 2)

        print(str(acquisition_params['Nx'] * acquisition_params['Ny']
              * acquisition_params['Nz']) + " images to segment")
        xrng = range(acquisition_params['Ny'])
        yrng = range(acquisition_params['Nx'])
        zrng = range(acquisition_params['Nz'])
        # start cellpose
        model = models.CellposeModel(gpu=use_gpu, pretrained_model=cpmodel)

        savepath = os.path.join(dest_dir, exp_id)
        if not dest_remote:
            os.makedirs(savepath, exist_ok=True)

        # Create dataframes for number of cells in the view and the blob info
        n_cell_df = pd.DataFrame(columns=['x_id', 'y_id', 'z_id', "n_cell"])
        blob_df = pd.DataFrame(columns=['x_id', 'y_id', 'z_id', "blob_x",
                               "blob_y", "blob_w", "blob_h", "blob_x_center", "blob_y_center"])
        # segment one at a time - gpu bottleneck
        for x_id, y_id, z_id in product(xrng, yrng, zrng):
            x_id = str(x_id)
            y_id = str(y_id)
            z_id = str(z_id)
            impath = os.path.join(
                root_dir, exp_id, '0', f"{x_id}_{y_id}_{z_id}_Fluorescence_{flr_nm}_nm_Ex.bmp")
            if root_remote:
                im = np.array(imread_gcsfs(fs, impath), dtype=np.uint8)
            else:
                im = np.array(cv2.imread(impath), dtype=np.uint8)
            im_flr = im / flatfield_fluorescence
            im_full = np.stack(
                (np.zeros(im_flr.shape),  np.zeros(im_flr.shape), im_flr), axis=2)
            # crop the outside - get rid of edge effects
            if center_crop > 0:
                im = np.zeros(im_full.shape)
                im = im_full[center_crop:-center_crop, center_crop:-center_crop, :]
            else:
                im = im_full
            # preproccess the image
            im = im.astype(np.float32)
            im -= np.min(im)
            if np.max(im) > np.min(im) + 1e-3:
                im /= (np.max(im) - np.min(im))
            im = im * 255
            # run segmentation
            masks, __, __ = model.eval(im, channels=channels, diameter=None)
            # get number of cells
            n_cells = np.max(masks)
            n_cell_df.loc[len(n_cell_df)] = [x_id, y_id, z_id, n_cells]
            # make binary mask with outlines
            outlines = masks * utils.masks_to_outlines(masks)
            bin_mask = (masks > 0) * 1.0
            outlines = (outlines > 0) * 1.0
            bin_mask = (bin_mask * (1.0 - outlines) * 255).astype(np.uint8)
            # find blobs - dilate the flat mask to merge adjacet cells, then find contours
            if dilation_sz > 0:
                shape = cv2.MORPH_ELLIPSE
                element = cv2.getStructuringElement(
                    shape, (2 * dilation_sz + 1, 2 * dilation_sz + 1), (dilation_sz, dilation_sz))
                bin_mask = np.array(cv2.dilate(bin_mask, element))
            contours, __ = cv2.findContours(
                bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            # for each contour, find center and bounding box
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                x = x
                y = y
                x_center = int(x + w/2)
                y_center = int(y + h/2)
                # add info to dataframe
                blob_df.loc[len(blob_df)] = [x_id, y_id, z_id,
                                             x, y, w, h, x_center, y_center]

            # save all data
            impath = f"{savepath}/{x_id}_{y_id}_{z_id}_mask.bmp"
            if dest_remote:
                img_bytes = cv2.imencode('.bmp', bin_mask)[1]
                with fs.open(impath, 'wb') as f:
                    f.write(img_bytes)
            else:
                cv2.imwrite(impath, bin_mask)
            if debug and not dest_remote:
                impath = f"{savepath}/{x_id}_{y_id}_{z_id}_image.bmp"
                cv2.imwrite(impath, im)

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


def imread_gcsfs(fs, file_path):
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

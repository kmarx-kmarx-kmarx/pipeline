# For each dataset, make a N x 3 x a x a .npy file where N is the number of spots per view, 3 is flatfield white-fluorescence-mask channels, and a is the width/height of the view.

import numpy as np
import os
import gcsfs
import imageio
from dpc_overlay import generate_dpc
from tqdm import tqdm
import pandas as pd
import cv2


def main():
    debug = False
    root_dir = 'gs://octopi-tb-data/20230303'
    root_proc = 'gs://octopi-tb-data-processing/20230303_masks'
    dest_dir = 'gs://octopi-tb-data-processing/20230303_cells'
    key = '/home/prakashlab/Documents/keys/tb_key.json'
    gcs_project = 'soe-octopi'
    correction_path = "/home/prakashlab/Documents/kmarx/tb_segment/pipeline/illumination correction tb"
    flr_nm = "405"
    a = 64   # crop a x a around the spot
    brightness_th = 130  # if either the left or right channel avg brightness is below th, fill with 0 instead of making the mask
    mask_cells = False
    dilation_sz = -1  # amount to expand mask to capture area around cell. set negative to disable
    make_view(debug, root_dir, root_proc, dest_dir, key, gcs_project,
              correction_path, a, flr_nm, brightness_th, mask_cells, dilation_sz)


def make_view(debug, root_dir, root_proc, dest_dir, key, gcs_project, correction_path, a, flr_nm, brightness_th, mask_cells, dilation_sz):
    # Load remote files if necessary
    root_remote = False
    if root_dir[0:5] == 'gs://':
        root_remote = True
    dest_remote = False
    if dest_dir[0:5] == 'gs://':
        dest_remote = True
    else:
        os.makedirs(dest_dir, exist_ok=True)
    proc_remote = False
    if root_proc[0:5] == 'gs://':
        proc_remote = True
    fs = None
    if root_remote or dest_remote or proc_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project, token=key)

    if not os.path.exists(os.path.join(correction_path, "flatfield_left.npy")):
        if root_remote:
            fs.get(os.path.join(root_dir, "illumination correction", "left",
                   "flatfield.npy"), os.path.join(correction_path, "flatfield_left.npy"))
        else:
            print("Left missing")
            raise FileNotFoundError
    if not os.path.exists(os.path.join(correction_path, "flatfield_right.npy")):
        if root_remote:
            fs.get(os.path.join(root_dir, "illumination correction", "right",
                   "flatfield.npy"), os.path.join(correction_path, "flatfield_right.npy"))
        else:
            print("Right missing")
            raise FileNotFoundError
    if not os.path.exists(os.path.join(correction_path, "flatfield_fluorescence.npy")):
        if root_remote:
            fs.get(os.path.join(root_dir, "illumination correction", "fluorescent",
                   "flatfield.npy"), os.path.join(correction_path, "flatfield_fluorescence.npy"))
        else:
            print("fluorescent missing")
            raise FileNotFoundError

    # get correction files
    ff_left = np.load(os.path.join(correction_path, "flatfield_left.npy"))
    ff_right = np.load(os.path.join(correction_path, "flatfield_right.npy"))
    ff_fluorescence = np.load(os.path.join(
        correction_path, "flatfield_fluorescence.npy"))

    # Create dilation element
    if dilation_sz >= 0:
        shape = cv2.MORPH_ELLIPSE
        element = cv2.getStructuringElement(
            shape, (2 * dilation_sz + 1, 2 * dilation_sz + 1), (dilation_sz, dilation_sz))

    # get exp ids
    if proc_remote:
        exp_ids = fs.ls(root_proc)
        exp_ids = [i.split('/')[-1] for i in exp_ids]
    else:
        exp_ids = os.listdir(root_proc)

    exp_ids = exp_ids[5:]
    for id in tqdm(exp_ids):
        # get blob csv
        csv_path = os.path.join(root_proc, id, "cell_group_locations.csv")
        print(csv_path)
        if root_remote:
            with fs.open(csv_path, 'rb') as f:
                spot_data = pd.read_csv(f)
        else:
            spot_data = pd.read_csv(csv_path)
        if debug:
            spot_data = spot_data.head(10)
        n_ims = spot_data.shape[0]
        # pre-allocate memory for a np array
        if mask_cells:
            view = np.zeros((n_ims, a, a, 5), dtype=np.uint8)
        else:
            view = np.zeros((n_ims, a, a, 3), dtype=np.uint8)
        # Iterate through each image - make DPC, get fluorescence, get mask
        prev_x = None
        prev_y = None
        prev_z = None
        im_dpc = None
        im_flr = None
        im_lft = None
        im_rht = None
        mask = None
        for i, row in tqdm(spot_data.iterrows(), total=n_ims):
            # if the current idx is different from what we read previously, read the new image.
            # otherwise we can get away with using what we already had
            if not (row['x_id'] == prev_x and row['y_id'] == prev_y and row['z_id'] == prev_z):
                img_path = os.path.join(
                    root_dir, id, '0', f"{row['x_id']}_{row['y_id']}_{row['z_id']}_BF_LED_matrix_left_half.bmp")
                mask_path = os.path.join(
                    root_proc, id, f"{row['x_id']}_{row['y_id']}_{row['z_id']}_mask.bmp")

                if root_remote:
                    i_left = np.array(imread_gcsfs(fs, img_path), dtype=float)
                    img_path = img_path.replace(
                        "BF_LED_matrix_left_half", "BF_LED_matrix_right_half")
                    i_right = np.array(imread_gcsfs(fs, img_path), dtype=float)
                    img_path = img_path.replace(
                        "BF_LED_matrix_right_half", f"Fluorescence_{flr_nm}_nm_Ex")
                    i_flr = np.array(imread_gcsfs(fs, img_path), dtype=float)
                else:
                    i_left = imageio.imread(img_path)
                    img_path = img_path.replace(
                        "BF_LED_matrix_left_half", "BF_LED_matrix_right_half")
                    i_right = imageio.imread(img_path)
                    img_path = img_path.replace(
                        "BF_LED_matrix_right_half", f"Fluorescence_{flr_nm}_nm_Ex")
                    i_flr = imageio.imread(img_path)
                if proc_remote:
                    m = np.array(imread_gcsfs(fs, mask_path), dtype=np.uint8)
                else:
                    m = imageio.imread(mask_path)

                # do flatfield correction
                i_left = i_left / (255 * ff_left)
                i_right = i_right / (255 * ff_right)
                i_flr = i_flr / ff_fluorescence
                # generate DPC
                i_dpc = generate_dpc(i_left, i_right)
                # expand edges
                im_dpc = np.zeros(
                    (i_dpc.shape[0]+a, i_dpc.shape[1]+a), dtype=np.uint8)
                im_dpc[int(a/2):int(a/2)+i_dpc.shape[0], int(a/2):int(a/2)+i_dpc.shape[1]] = i_dpc.astype(np.uint8)

                im_flr = np.zeros(
                    (i_flr.shape[0]+a, i_flr.shape[1]+a), dtype=np.uint8)
                im_flr[int(a/2):int(a/2)+i_flr.shape[0], int(a/2):int(a/2)+i_flr.shape[1]] = i_flr.astype(np.uint8)

                mask = np.zeros((m.shape[0]+a, m.shape[1]+a), dtype=np.uint8)
                mask[int(a/2):int(a/2)+m.shape[0], int(a/2):int(a/2)+m.shape[1]] = m.astype(np.uint8)

                im_lft = np.zeros(
                    (i_flr.shape[0]+a, i_flr.shape[1]+a), dtype=np.uint8)
                im_lft[int(a/2):int(a/2)+i_flr.shape[0], int(a/2):int(a/2)+i_flr.shape[1]] = (255*i_left).astype(np.uint8)
                im_rht = np.zeros(
                    (i_flr.shape[0]+a, i_flr.shape[1]+a), dtype=np.uint8)
                im_rht[int(a/2):int(a/2)+i_flr.shape[0], int(a/2):int(a/2) +
                       i_flr.shape[1]] = (255*i_right).astype(np.uint8)

                if debug:
                    imageio.imwrite(
                        f"{dest_dir}/{id}_{row['x_id']}_{row['y_id']}_{row['z_id']}_mask.bmp", mask)
                    imageio.imwrite(
                        f"{dest_dir}/{id}_{row['x_id']}_{row['y_id']}_{row['z_id']}_dpc.bmp", im_dpc)
                    imageio.imwrite(
                        f"{dest_dir}/{id}_{row['x_id']}_{row['y_id']}_{row['z_id']}_flr.bmp", im_flr)
                    imageio.imwrite(
                        f"{dest_dir}/{id}_{row['x_id']}_{row['y_id']}_{row['z_id']}_lft.bmp", i_left)
                    imageio.imwrite(
                        f"{dest_dir}/{id}_{row['x_id']}_{row['y_id']}_{row['z_id']}_rht.bmp", i_right)

                prev_x = row['x_id']
                prev_y = row['y_id']
                prev_z = row['z_id']

            # crop down to size - centers already shifted by a/2 because of added padding
            xmin = row['blob_x_center']
            xmax = row['blob_x_center']+a
            ymin = row['blob_y_center']
            ymax = row['blob_y_center']+a
            # get crops
            i_dpc = im_dpc[ymin:ymax, xmin:xmax]
            i_flr = im_flr[ymin:ymax, xmin:xmax]
            # get only relevant mask
            m = mask[ymin:ymax, xmin:xmax]
            contours, _ = cv2.findContours(
                m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            m = np.zeros(m.shape, 'uint8')
            w_rec, h_rec = (row['blob_w'], row['blob_h'])
            found_flag = False
            # print(f"target: {(w_rec, h_rec)}")
            bboxes = []
            for cnt in contours:
                _, _, w_blb, h_blb = cv2.boundingRect(cnt)
                bboxes.append((w_blb, h_blb))
                if w_blb == w_rec and h_blb == h_rec:
                    cv2.drawContours(m, [cnt], -1, 255, -1)
                    found_flag = True
                    break
                    
            if found_flag ==  False:
                # check if there is an off by one error
                for w_blb, h_blb in bboxes:
                    if (np.abs(w_rec - w_blb) <= 1 and np.abs(h_rec-h_blb) <= 1) :
                        cv2.drawContours(m, [cnt], -1, 255, -1)
                        found_flag = True
                        break
            
            if found_flag == False:
                m = mask[ymin:ymax, xmin:xmax]
                cv2.imwrite("bad_image_dpc.png", i_dpc)
                cv2.imwrite("bad_image_flr.png", i_flr)
                cv2.imwrite("bad_image_mask.png", m)
                a = [cv2.boundingRect(cnt) for cnt in contours]
                print(a)
                print((w_rec, h_rec))
                raise UserWarning
            if np.sum(m) == 0 and np.sum(mask[ymin:ymax, xmin:xmax])!= 0:
                print(contours)
                print(f"target: {(w_rec, h_rec)}")
                raise UserWarning
            # if brightfields are below threshold, throw out the data
            i_lft = im_lft[ymin:ymax, xmin:xmax]
            i_rht = im_rht[ymin:ymax, xmin:xmax]
            if debug:
                print(np.mean(i_lft))
                print(np.mean(i_rht))
            if (np.mean(i_lft) < brightness_th) or (np.mean(i_rht) < brightness_th):
                if debug:
                    print(f"{i} is bad")
                i_dpc = i_dpc * 0
                i_flr = i_flr * 0
                m = m * 0
            if mask_cells:
                # Mask the area around the cells
                if dilation_sz > 0:
                    masking = np.array(cv2.dilate(m, element))
                else:
                    masking = np.copy(m)
                # Mask the DPC
                i_dpc_msk = cv2.bitwise_and(i_dpc, i_dpc, mask=masking)
                # Mask the fluorescence
                i_flr_msk = cv2.bitwise_and(i_flr, i_flr, mask=masking)
                im = np.stack((i_dpc, i_flr, i_dpc_msk, i_flr_msk, m), axis=2)
            else:
                im = np.stack((i_dpc, i_flr, m), axis=2)
            if debug:
                print(
                    f"{np.min(i_dpc)}, {np.max(i_dpc)}, {np.min(m)}, {np.max(m)}, {np.min(i_flr)}, {np.max(i_flr)}")
                imageio.imwrite(
                    f"{dest_dir}/{i}_{id}_{row['x_id']}_{row['y_id']}_{row['z_id']}_{row['blob_x_center']}_{row['blob_y_center']}_im.bmp", im)
            # put im into view
            view[i, :, :, :] = im[:, :, :]
        # save view
        savepath = os.path.join(dest_dir, f"{id}_spots.npy")
        if dest_remote:
            with fs.open(savepath, "wb") as f:
                np.save(f, view)
        else:
            np.save(savepath, view)


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

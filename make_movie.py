import imageio
from tqdm import tqdm
import numpy as np
import cv2
import imageio
import gcsfs
import os
import random
import pandas as pd
from npy_loader import load_npy
import logging


def main():
    logging.basicConfig(filename='log.log')
    root_dir = 'gs://octopi-tb-data-processing/20221002_cells'
    dest_dir = 'movies_indiv_cell'
    csv_dir = 'gs://octopi-tb-data-processing/20221002_masks_2'
    key = '/home/prakashlab/Documents/keys/tb_key.json'
    gcs_project = 'soe-octopi'
    exp_ids = ['SP1+_2022-10-02_20-10-50.529345', 'SP1-_2022-10-02_20-23-56.661506', 'SP2+_2022-10-02_20-41-41.247131', 'SP2-_2022-10-02_20-58-10.547699', 'SP3+_2022-10-02_21-13-35.929780', 'SP3-_2022-10-02_21-31-39.783478', 'SP4+_2022-10-02_21-47-12.117750', 'SP4-_2022-10-02_21-59-26.387251', 'SP5+_2022-10-02_22-16-53.962897', 'SP5-_2022-10-02_22-29-35.714940']
    # ['SP1-_2022-10-02_20-23-56.661506', 'SP2-_2022-10-02_20-58-10.547699', 'SP3-_2022-10-02_21-31-39.783478', 'SP4-_2022-10-02_21-59-26.387251', 'SP5-_2022-10-02_22-29-35.714940']
    # Only include cells with average brightness larger than this. Set to 1 to disable
    brightness_thresh = 0.01
    center_crop = 0  # Set to 0 to disable
    n_frames = 0
    save_stills = False
    save_movie = True
    randomize = True
    seed = 1
    fps = 3

    make_movies(csv_dir, root_dir, dest_dir, exp_ids, center_crop, n_frames, save_stills,
                save_movie, brightness_thresh, fps, randomize, seed, key, gcs_project)


def make_movies(csv_dir, root_dir, dest_dir, exp_ids, center_crop, n_frames, save_stills, save_movie, brightness_thresh, fps, randomize, seed, key, gcs_project):
    # Load remote files if necessary
    csv_remote = False
    if csv_dir[0:5] == 'gs://':
        csv_remote = True
        os.makedirs(dest_dir, exist_ok=True)
    root_remote = False
    if root_dir[0:5] == 'gs://':
        root_remote = True
        os.makedirs(dest_dir, exist_ok=True)
    dest_remote = False
    if dest_dir[0:5] == 'gs://':
        dest_remote = True
    else:
        os.makedirs(dest_dir, exist_ok=True)
    fs = None
    if root_remote or dest_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project, token=key)

    for id in tqdm(exp_ids):
        if csv_remote:
            csv_path = os.path.join(csv_dir, id, "cell_group_locations.csv")
            with fs.open(csv_path, 'rb') as f:
                data = pd.read_csv(f)
        else:
            csv_path = os.path.join(csv_dir, id+"_cell_group_locations.csv")
            data = pd.read_csv(csv_path)

        filepath = os.path.join(root_dir, f"{id}_spots.npy")
        images_numpy = load_npy(
            filepath, id, fs=fs, root_remote=root_remote, save_local=True, dest_dir=dest_dir)

        n_im, w, h, ch = images_numpy.shape
        if center_crop > 0 and center_crop < w and center_crop < h:
            x = int(h/2 - center_crop/2)
            y = int(w/2 - center_crop/2)
            images_numpy = images_numpy[:, int(y):int(
                y+center_crop), int(x):int(x+center_crop), :]

        indices = list(range(n_im))
        if randomize:
            random.seed(seed)
            random.shuffle(indices)

        make_movie(images_numpy, indices, data, os.path.join(dest_dir, "fp"+id), n_frames,
                   save_stills, save_movie, brightness_thresh=brightness_thresh, fps=fps)


def make_movie(images_numpy, indices, data, output_file, n_frames, save_stills, save_movie, brightness_thresh=1, scale_factor=2, fps=24):
    if save_movie:
        writer = imageio.get_writer(output_file + '.mp4', fps=fps)
    for i in tqdm(indices):
        frame = images_numpy[i, :, :, :]
        # im = np.stack((i_dpc, i_flr, i_dpc_msk, i_flr_msk, m), axis=2)
        if frame.shape[2] == 5:
            img_fluorescence = frame[:, :, 1]
            img_dpc = frame[:, :, 0]
            img_flr_mask = frame[:, :, 3]
            img_dpc_mask = frame[:, :, 2]
            mask = frame[:, :, 4]
        elif frame.shape[2] == 3:
            # im = np.stack((i_dpc, i_flr, m), axis=2)
            img_fluorescence = frame[:, :, 1]
            img_dpc = frame[:, :, 0]
            mask = frame[:, :, 2]
        else:
            print(frame.shape)
            raise UserWarning

        # Don't bother checking brightness if not necessary
        if np.count_nonzero(mask) == 0:
            # print(f"Mask {j} Empty")
            continue
        elif frame.shape[2] == 3:
            flr_avg = 255
        else:  # check if we are bright enough
            flr_avg = (np.sum(img_flr_mask)/255)/(np.count_nonzero(mask))

        if brightness_thresh >= 1 or flr_avg > brightness_thresh:
            if frame.shape[2] == 5:
                frame = np.hstack(
                    [img_dpc, img_fluorescence, img_dpc_mask, img_flr_mask, mask]).astype('uint8')
            elif frame.shape[2] == 3:
                frame = np.hstack(
                    [img_dpc, img_fluorescence, mask]).astype('uint8')

            new_height, new_width = int(
                frame.shape[0] * scale_factor), int(frame.shape[1] * scale_factor)
            frame = cv2.resize(frame, (new_width, new_height),
                               interpolation=cv2.INTER_NEAREST)
            if save_stills:
                imageio.imwrite(output_file + '/' + str(i) + '.png', frame)
            if save_movie:
                writer.append_data(frame)

    if save_movie:
        writer.close()


if __name__ == "__main__":
    main()

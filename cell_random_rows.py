import gcsfs
import imageio
import numpy as np
import pandas as pd
import cv2
import os
import time
from tqdm import trange

def main():
    # root_dir needs a trailing slash (i.e. /root/dir/)
    root_dir = "gs://octopi-codex-data-processing/UUlABKZIWxiZP5UnJvx6z1CZMhtxx9tm/"
    exp_id   = "20220823_20x_PBMC_2/"
    dest_dir = "./cell_crop_rows/"#"gs://octopi-codex-data-processing/foF3pmoJguvzNdwqbUmpOkwRzAsv39FO/" + exp_id + "image_crop/"
    channels =  ["Fluorescence_638_nm_Ex", "Fluorescence_561_nm_Ex", "Fluorescence_488_nm_Ex", "Fluorescence_405_nm_Ex"]
    cellposition_file = "./meanbright_9.csv"
    zstack  = 'f' # select which z to crop. set to 'f' to select the focus-stacked
    key = '/home/prakashlab/Documents/kmarx/soe-octopi-27a4691943f1.json'
    gcs_project = 'soe-octopi'
    cell_radius = 25
    n_of_strips = 30 * 300
    ftype = 'png'
    subtract_min = True
    t0 = time.time()
    make_crops(root_dir, exp_id, channels, zstack, cellposition_file, key, gcs_project, dest_dir, cell_radius, n_of_strips, ftype, subtract_min)
    t1 = time.time()
    print(t1-t0)

def make_crops(root_dir, exp_id, channels, zstack, cellposition_file, key, gcs_project, dest_dir, cell_radius, n_of_strips, ftype, subtract_min):
    img_remote = False
    if root_dir[0:5] == 'gs://':
        img_remote = True
    data_remote = False
    if cellposition_file[0:5] == 'gs://':
        data_remote = True
    dest_remote = False
    dest_local = dest_dir
    if dest_dir[0:5] == 'gs://':
        dest_local = "./imgs_"+exp_id
        dest_remote = True
    
    fs = None
    if img_remote or data_remote or dest_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)
    
    # get remote data
    if data_remote:
        with fs.open(celltype_file, 'r' ) as f:
            celltype_df = pd.read_csv(celltype_file)
    else:
        with open(celltype_file, 'r' ) as f:
            celltype_df = pd.read_csv(celltype_file)
    
    df = None
    path = root_dir + exp_id + 'index.csv'
    if img_remote:
        with fs.open(path, 'r' ) as f:
            df = pd.read_csv(f)
    else:
        with open( path, 'r' ) as f:
            df = pd.read_csv(f)
    ids = df['Acquisition_ID']
    seen_types = dict()
    
    for i in trange(celltype_df.shape[0]):
        i_idx = int(celltype_df["i"][i])
        j_idx = int(celltype_df["j"][i])
        # note - x and y are swapped here
        ypos  = int(celltype_df["x"][i])
        xpos  = int(celltype_df["y"][i])
        cell_type = celltype_df["cell_type"][i]

        # check if we have seen this cell type already
        if cell_type in seen_types.keys():
            if seen_types[cell_type] >= n_of_each_type:
                continue
            else:
                seen_types[cell_type] += 1
        else:
            seen_types[cell_type] = 1
        #imgs = np.zeros((2*cell_radius, 2*cell_radius * (len(ids)*(len(channels) - 1) +1)))
        imgs = np.zeros((2*cell_radius, 2*cell_radius * len(ids)*len(channels)))
        err_flag = False
        for ch_idx, channel in enumerate(channels):
            if err_flag:
                break
            for id_idx, id in enumerate(ids):
                # load image 
                filename = id + '/0/' + str(i_idx) + '_' + str(j_idx) + '_' + str(zstack) + '_' + channel + '.' + ftype
                image_path = root_dir + exp_id + filename
                #print(image_path)
                try:
                    if img_remote:
                        im = imread_gcsfs(fs,image_path)
                    else:
                        im = cv2.imread(image_path)
                except FileNotFoundError:
                    print(image_path)
                    print(f"Invalid index at celltype_df index {i}!")
                    err_flag = True
                    break
                im = np.array(im)
                if(np.max(im) == 0):
                    print("error - image blank")
                    continue
                xmax, ymax = im.shape

                xcrop = [int(max(0, xpos - cell_radius)), int(min(xmax, xpos + cell_radius))]
                ycrop = [int(max(0, ypos - cell_radius)), int(min(ymax, ypos + cell_radius))]

                cropped_image = np.array(im[xcrop[0]:xcrop[1], ycrop[0]:ycrop[1]])

                if subtract_min:
                    cropped_image = cropped_image - np.min(cropped_image)
                    
                a = 2*cell_radius*(ch_idx * len(ids) + id_idx)
                b = 2*cell_radius*(1 + ch_idx * len(ids) + id_idx)
                #print((a,b))
                imgs[:, a : b] = cropped_image

                # only get 1 fluorescence 405nm image
                # if channel == "Fluorescence_405_nm_Ex":
                #     break
        if err_flag:
            seen_types[cell_type] -= 1
            continue
        cell_type = celltype_df["cell_type"][i]
        filename = str(i) + "_" + str(ypos) + "_" + str(xpos) + '.bmp'
        savepath = dest_local + cell_type + '/'
        os.makedirs(savepath, exist_ok=True)
        savepath = savepath + filename
        #print(savepath)
        remotepath = dest_dir + cell_type + '/' + filename
        #print(remotepath)
        cv2.imwrite(savepath, imgs)
        if dest_remote:
            fs.put(savepath, remotepath)
            os.remove(savepath)




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


if __name__ == '__main__':
    main()
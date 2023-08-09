import numpy as np
import glob
import os
import gcsfs
import cv2
import math
import random
import imageio
import os

def main():
    # root_dir needs a trailing slash (i.e. /root/dir/)
    root_dir = "gs://octopi-malaria-tanzania-2021-data/"#/media/prakashlab/T7/malaria-tanzina-2021/"#'gs://octopi-codex-data-processing/' #"/home/prakashlab/Documents/kmarx/pipeline/tstflat/"# 'gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU/'#
    dest_dir = "/home/prakashlab/Documents/kmarx/16C/"
    exp_id   = "U16C_2022-10-18_21-41-21.249946/" # experiment ID - needs a trailing '/'
    left_light = "BF_LED_matrix_left_half" # name of the left illumination
    right_light = "BF_LED_matrix_right_half" # set emppty to not do DPC
    fluorescence = "Fluorescence_405_nm_Ex"  # set empty to not do overlay
    zstack  = '0' # select which z to run segmentation on. set to 'f' to select the focus-stacked
    key = '/home/prakashlab/Documents/kmarx/data-20220317-keys.json'
    gcs_project = 'soe-octopi'
    ftype = 'bmp'
    do_dpc_overlay(root_dir, dest_dir, exp_id, left_light, right_light, fluorescence, zstack, key, gcs_project, ftype)

def do_dpc_overlay(root_dir, dest_dir, exp_id, left_light, right_light, fluorescence, zstack, key, gcs_project, ftype):
    # load fluorescence correction
    flatfield_fluorescence = np.load('illumination correction/flatfield_fluorescence.npy')
    flatfield_fluorescence = np.dstack((flatfield_fluorescence,flatfield_fluorescence,flatfield_fluorescence))
    flatfield_left = np.load('illumination correction/flatfield_left.npy')
    flatfield_right = np.load('illumination correction/flatfield_right.npy')
    # Load remote files if necessary
    root_remote = False
    if root_dir[0:5] == 'gs://':
        root_remote = True
    dest_remote = False
    if dest_dir[0:5] == 'gs://':
        dest_remote = True
    fs = None
    if root_remote or dest_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)
    
    path = os.path.join(root_dir , exp_id , "0/**_" + zstack + "_" + left_light + '.' + ftype)
    print(path)
    if root_remote:
        allpaths = [p for p in fs.glob(path, recursive=True) if p.split('/')[-2] == '0']
    else:
        allpaths = [p for p in glob.iglob(path, recursive=True) if p.split('/')[-2] == '0']
    # remove duplicates
    lftpaths = list(dict.fromkeys(allpaths))
    lftpaths = [p.replace('//', '/') for p in lftpaths]
    print(str(len(lftpaths)) + " views")

    # get right, fluorescent image paths if able
    rhtpaths = None
    flrpaths = None
    if right_light != "":
        rhtpaths = [p.replace(left_light, right_light) for p in lftpaths]
    if fluorescence != "":
        flrpaths = [p.replace(left_light, fluorescence) for p in lftpaths]
    
    for idx in range(len(lftpaths)):
        if idx > 10:
            break
        print(lftpaths[idx])
        print(rhtpaths[idx])
        print(flrpaths[idx])
        # read the images
        rht = None
        flr = None
        if root_remote:
            lft = np.array(imread_gcsfs(fs, lftpaths[idx]), dtype=np.uint8)
            if type(rhtpaths) != type(None):
                rht = np.array(imread_gcsfs(fs, rhtpaths[idx]), dtype=np.uint8)
            if type(flrpaths) != type(None):
                flr = np.array(imread_gcsfs(fs, flrpaths[idx]), dtype=np.uint8)
        else:
            lft = np.array(cv2.imread(lftpaths[idx]), dtype=np.uint8)
            if type(rhtpaths) != type(None):
                rht = np.array(cv2.imread(rhtpaths[idx]), dtype=np.uint8)
            if type(flrpaths) != type(None):
                flr = np.array(cv2.imread(flrpaths[idx]), dtype=np.uint8)
        if len(lft.shape) == 3:
            lft=lft[:,:,0]
        if len(rht.shape) == 3:
            rht=rht[:,:,0]
        # flatfield correction
        flr = flr.astype('float')/255
        lft = lft.astype('float')/255
        rht = rht.astype('float')/255
        flr = flr/flatfield_fluorescence
        lft = lft/flatfield_left
        rht = rht/flatfield_right
        # fluorescence enhance
        flr = flr*1.25
        flr[flr>1] = 1
        
        # generate DPC (converts from float to uint8)
        dpc = None
        if type(rht) != type(None):
            dpc = generate_dpc(lft, rht)
            if(len(dpc.shape)<3):
                dpc = np.dstack((dpc,dpc,dpc))
        # generate overlay (converts from float to uint8)
        over = None
        if type(flr) != type(None):
            if type(dpc) == type(None):
                over = overlay(lft, flr)
            else:
                over = overlay(dpc, flr)
        
        # save results:
        path = lftpaths[idx].rsplit('_', 2)[0]
        pth = path.split('/')
        path = dest_dir + pth[-4] + '/' + pth[-3] + '/' + pth[-2] + '/' + pth[-1]

        if type(over) != type(None):
            savepath = path + "_overlay." + ftype
            print(savepath)
            if dest_remote:
                cv2.imwrite("./img." + ftype, over)
                fs.put("./img." + ftype, savepath)
                os.remove("./img." + ftype)
            else:
                os.makedirs(savepath.rsplit('/', 1)[0], exist_ok=True)
                cv2.imwrite(savepath, over)
        if type(dpc) != type(None):
            savepath = path + "_dpc." + ftype
            print(savepath)
            if dest_remote:
                cv2.imwrite("./img." + ftype, dpc)
                fs.put("./img." + ftype, savepath)
                os.remove("./img." + ftype)
            else:
                os.makedirs(savepath.rsplit('/', 1)[0], exist_ok=True)
                cv2.imwrite(savepath, dpc)

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

def generate_dpc(I_1,I_2):
    I_dpc = np.divide(I_1-I_2,I_1+I_2)
    I_dpc = I_dpc + 0.5
    I_dpc[I_dpc<0] = 0
    I_dpc[I_dpc>1] = 1

    I_dpc = (255*I_dpc)

    return I_dpc.astype('uint8')

def overlay(I_dpc, I_flr):
    dpc = I_dpc.astype("float")/255
    I_overlay = 0.64*I_flr + 0.36*dpc

    I_overlay = (255 * I_overlay)

    return I_overlay.astype('uint8')

if __name__ == "__main__":
    main()
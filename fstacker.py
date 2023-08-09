# CLI/script to focus-stack a series of local or remote images 
import gcsfs
import imageio
import argparse
import numpy as np
import pandas as pd
import cv2
from itertools import product
from skimage.morphology import white_tophat
from skimage.morphology import disk
from skimage.registration import phase_cross_correlation
from skimage.transform import warp, AffineTransform
import os
import json
debugging = False
import time
from tqdm import tqdm

def main():
    
    CLI = False     # set to true for CLI, if false, the following constants are used:
    use_gpu = True  # use GPU accelerated focus stacking
    prefix = ""     # if index.csv DNE, use prefix, else keep empty
    key = '/home/prakashlab/Documents/kmarx/soe-octopi-27a4691943f1.json'
    gcs_project = 'soe-octopi'
    src = "gs://octopi-codex-data/"
    dst = "gs://octopi-codex-data-processing/" #"./test"
    exp = ['20230525_20x_PBMC/']
    cha = ["Fluorescence_405_nm_Ex", "Fluorescence_638_nm_Ex", "Fluorescence_561_nm_Ex", "Fluorescence_488_nm_Ex"] # ['BF_LED_matrix_full', 'BF_LED_matrix_left_half', 'BF_LED_matrix_low_NA', 'BF_LED_matrix_right_half', 'Fluorescence_405_nm_Ex']
    typ = "bmp"
    colors = {'0':[255,255,255],'1':[255,200,0],'2':[30,200,30],'3':[0,0,255]} # BRG
    remove_background = False
    invert_contrast = False
    shift_registration = True
    subtract_background = False
    use_color = False
    crop_start = 250 # crop settings
    crop_end = 3000-250
    full_size = 3000
    WSize = 9     # Focus stacking params
    alpha = 0.2
    sth = 13
    verbose = True
    # CLI parsing
    parser = argparse.ArgumentParser(description='focus stack parameters')
    # settings for CPU vs GPU
    hardware_args = parser.add_argument_group("hardware arguments")
    hardware_args.add_argument('--use_gpu', action='store_true', help='use gpu if cuda installed')

    # settings for locating and formatting images
    img_args = parser.add_argument_group("input image arguments")
    img_args.add_argument('--key', default=[], type=str, help='path to JSON key to GCSFS server')
    img_args.add_argument('--gcs_project', default=[], type=str, help='Name of GCS project')
    img_args.add_argument('--src', default=[], type=str, help='source directory or GCSFS bucket name')
    img_args.add_argument('--typ', default=[], type=str, help='image type')
    img_args.add_argument('--exp', default=[], type=str, nargs='+', help='experiment ID (one or more)')
    img_args.add_argument('--cha', default=[], type=str, nargs='+', help='channel name (e.g. Fluorescence_488_nm_Ex, one or more)')
    img_args.add_argument('--dst', default=[], type=str, help='destination directory or GCSFS path to save images')
    img_args.add_argument('--crop_start', default=[], type=int, help='position to start cropping the image')
    img_args.add_argument('--crop_end', default=[], type=int, help='position to stop cropping the image')

    # image processing settings
    prc_args = parser.add_argument_group("image processing arguments")
    prc_args.add_argument('--remove_background', action='store_true', help='use a filter to remove background')
    prc_args.add_argument('--subtract_background', action='store_true', help='subtract off the minimum value')
    prc_args.add_argument('--invert_contrast', action='store_true', help='flip contrast on ch 0')
    prc_args.add_argument('--shift_registration', action='store_true', help='transform the images to register them properly')
    prc_args.add_argument('--use_color', action='store_true', help='preserve color data')

    # settings for stacking the images
    stack_args = parser.add_argument_group("stacking behavior arguments")
    stack_args.add_argument('--WSize', default=9,   type=int,   help='Filter size')
    stack_args.add_argument('--alpha', default=0.2, type=float, help='blending parameter')
    stack_args.add_argument('--sth',   default=13,  type=int,   help='blending parameter')
    # misc
    parser.add_argument('--verbose', action='store_true', help='show information about running and settings')
    args = parser.parse_args()

    if not CLI:
        args.use_gpu = use_gpu
        args.key = key
        args.gcs_project = gcs_project
        args.src = src
        args.exp = exp
        args.cha = cha
        args.dst = dst
        args.typ = typ
        args.crop_start = crop_start
        args.crop_end   = crop_end
        args.remove_background = remove_background
        args.subtract_background = subtract_background
        args.invert_contrast = invert_contrast
        args.shift_registration = shift_registration
        args.use_color = use_color
        args.WSize = WSize
        args.alpha = alpha
        args.sth   = sth
        args.verbose = verbose
    
    perform_stack(colors, prefix, full_size, args.use_gpu, args.key, args.gcs_project, args.src, args.exp, args.cha, args.dst, args.typ, args.crop_start, args.crop_end, args.remove_background, args.subtract_background, args.invert_contrast, args.shift_registration, args.use_color, args.WSize, args.alpha, args.sth, args.verbose)
    return
    
def perform_stack(colors, prefix, full_size, use_gpu, key, gcs_project, src, exp, cha, dst, typ, crop_start, crop_end, remove_background, subtract_background, invert_contrast, shift_registration, use_color, WSize, alpha, sth, verbose):
    os.makedirs(dst, exist_ok = True)
    t0 = time.time()
    a = crop_end - crop_start
    # Initialize arguments
    error = 0
    # verify source is given
    if len(src) == 0:
        print("Error: no source provided")
        error += 1
    # verify experiment ID is given
    if len(exp) == 0:
        print("Error: no experiment ID provided")
        error += 1
    # verify channel is given
    if len(cha) == 0:
        print("Error: no channel name provided")
        error += 1
    # verify file type is given
    if len(typ) == 0:
        print("Error: no file type provided")
        error += 1
    # check for destination
    if len(dst) == 0:
        dst = src
        if verbose:
            print("dst not given, set to src by default")
    # check if using gpu - load the appropriate version of fstack_images
    if use_gpu:
        from fstack_cu import fstack_images
        if verbose:
            print("Using GPU")
    else:
        from fstack import fstack_images
        if verbose:
            print("Using CPU")
    # check if remote
    root_remote = False
    if src[0:5] == 'gs://':
        root_remote = True

    dest_remote = False
    if dst[0:5] == 'gs://':
        dest_remote = True

    if (root_remote or dest_remote) and len(gcs_project) == 0:
        print("Remote source/destination but no project given")
        error += 1

    if (root_remote or dest_remote) and len(key) == 0:
        print("Remote source/destination but no key ")
        error += 1
        
    # if there are any errors, stop
    if error > 0:
        print(str(error) + " errors detected")
        return

    # Initialize the remote filesystem
    fs = None
    if (root_remote or dest_remote):
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)

    for exp_i in exp:
        # load index.csv for each top-level experiment index
        df = None
        path = src + exp_i + 'index.csv'
        if len(prefix) == 0:
            if root_remote:
                with fs.open(path, 'r' ) as f:
                    df = pd.read_csv(f)
            else:
                with open( path, 'r' ) as f:
                    df = pd.read_csv(f)
        if debugging and len(df) > 4:
            df = df.head(4)

        if verbose and len(prefix)==0:
            print(path + " opened")
            n = df.shape[0] # n is the number of cycles
            print("n cycles = " + str(n))
            loc = [id for id in df.loc[:, 'Acquisition_ID']]

        if len(prefix) > 0:
            if root_remote:
                if prefix == '*':
                    loc = [a.split('/')[-1] for a in fs.ls(src + exp_i)]
                else:
                    loc = [a.split('/')[-1] for a in fs.ls(src + exp_i) if a.split('/')[-1][0:len(prefix)] == prefix ]
            else:  
                if prefix == '*':
                    loc = [a.split('/')[-1] for a in os.listdir(src  + exp_i)]
                else:
                    loc = [a.split('/')[-1] for a in os.listdir(src  + exp_i) if a.split('/')[-1][0:len(prefix)] == prefix ]
        for id in loc:
            print(id)
        

        # Load the acquisition params for first view
        id = loc[0]
        if root_remote:
            json_file = fs.cat(os.path.join(src, exp_i, id, "acquisition parameters.json"))
            acquisition_params = json.loads(json_file)
        else:
            acquisition_params = json.loads(os.path.join(src, exp_i, id, "acquisition parameters.json"))

        if debugging:
            acquisition_params['Ny'] = min(acquisition_params['Ny'], 2)
            acquisition_params['Nx'] = min(acquisition_params['Nx'], 2)
            acquisition_params['Nz'] = min(acquisition_params['Nz'], 2)

        # pre-allocate arrays
        if use_color:
            I_zs = np.zeros((acquisition_params['Nz'],full_size,full_size,3))
        else:
            I_zs = np.zeros((acquisition_params['Nz'],full_size,full_size))
        # store total # of imgs
        print(f"{acquisition_params['Nz'] * acquisition_params['Ny'] * acquisition_params['Nx']} images in {id}")
        print(f"{len(loc)*acquisition_params['Nz'] * acquisition_params['Ny'] * acquisition_params['Nx']} total images")
       # perform fstack for each experiment and for each channel and for each i,j
        for i, j in tqdm(product(range(acquisition_params['Ny']), range(acquisition_params['Nx'])), total=(acquisition_params['Nx']*acquisition_params['Ny'])):
            for id in loc:
                if verbose:
                    print(id)
                for l in range(len(cha)):
                    if use_color:
                        color = colors[str(l)]
                        if l == 0 and invert_contrast:
                            color = [0,0,0]
                    
                    channel = cha[l]

                    for k in range(acquisition_params['Nz']):
                        filename = id + '/0/C4_' + str(i) + '_' + str(j) + '_' + str(k) + '_' + channel + '.' + typ
                        target = src + exp_i + filename
                        if verbose:
                            print(target)
                        try:
                            if root_remote:
                                I = imread_gcsfs(fs, target)
                            else:
                                I = cv2.imread(target)
                        except:
                            # Log missing data
                            print(f"Missing data: {target}")
                            raise UserWarning

                        if use_color:
                            I_zs[k,:,:,:] = I
                        else:
                            if len(I.shape)==3:
                                I = np.squeeze(I[:,:,0])
                            I_zs[k,:,:] = I
                    # if more than 4 images in zstack, focus-stack them
                    if acquisition_params['Nz'] > 4:
                        I = fstack_images(I_zs, list(range(acquisition_params['Nz'])), verbose=verbose, WSize=WSize, alpha=alpha, sth=sth)
                    # otherwise, just pick the middle image
                    else:
                        I = I_zs[int((acquisition_params['Nz'])/2),:,:]

                    if remove_background:
                        selem = disk(30) 
                        I = white_tophat(I,selem)
                        
                    if subtract_background:
                        I = I - np.amin(I)

                    # registration across channels
                    if shift_registration:
                        # take the first channel of the first cycle as reference
                        if id == loc[0]:
                            if l == 0:
                                I0 = normalize(I)
                                ref = f"{l}, {id}"
                        else:
                            if l == 0:
                                # compare the first channel of later cycles to the first channel of the first cycle
                                if verbose:
                                    print(f"compare({ref}) to ({l}, {id})")
                                try:
                                    shift, b, c = phase_cross_correlation(I0, normalize(I), upsample_factor=8)
                                    
                                    if verbose:
                                        print((shift, b, c))
                                except:
                                    shift = [0,0]
                                    print("Phase cross correlation failed")
                                if verbose:
                                    print(shift)
                                # create the transform
                                transform = AffineTransform(translation=(shift[0],shift[1]))
                                I = warp(I, transform)
                            else:
                                # apply shift to all channels
                                I = warp(I, transform)
                    if use_color:
                        for m in range(3):
                            if invert_contrast:
                                I[:,:,m] = 255 - I*(1-color[m]/255.0)
                            else:
                                I[:,:,m] = I*color[m]/255.0
                    else:
                        if invert_contrast:
                            I = 255 - I
                    # crop image
                    I = I[crop_start:crop_end,crop_start:crop_end]
                    # save images 
                    fname =  str(i) + '_' + str(j) + '_f_' + channel + '.png'
                    savepath = dst + exp_i + id + '/0/'
                    if verbose:
                        print(savepath+fname)
                    if dest_remote:
                        img_bytes = cv2.imencode('.bmp', I)[1]
                        with fs.open(savepath+fname, 'wb') as f:
                            f.write(img_bytes)
                    else:
                        os.makedirs(savepath, exist_ok=True)
                        cv2.imwrite(savepath+fname, I)

def normalize(img):
    '''
    put image in range 0 to 255
    '''
    im = np.copy(img)
    im = im - np.min(im)
    im = im/np.max(im)
    im = 255.0 * im
    return im

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

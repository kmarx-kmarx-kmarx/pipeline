# CLI to focus-stack a series of local or remote images 
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

def main():
    CLI = False     # set to true for CLI, if false, the following constants are used:
    use_gpu = True  # use GPU accelerated focus stacking
    key = '/home/prakashlab/Documents/fstack/codex-20220324-keys.json'
    gcs_project = 'soe-octopi'
    src = "gs://octopi-codex-data"
    dst = './test' # "gs://octopi-codex-data-processing/TEST_1HDcVekx4mrtl0JztCXLn9xN6GOak4AU" #"./test"
    exp = ["20220601_20x_75mm"]
    cha = ["Fluorescence_405_nm_Ex", "Fluorescence_488_nm_Ex", "Fluorescence_561_nm_Ex", "Fluorescence_638_nm_Ex"]
    typ = "bmp"
    colors = {'0':[255,255,255],'1':[255,200,0],'2':[30,200,30],'3':[0,0,255]} # BRG
    remove_background = False
    invert_contrast = False
    shift_registration = True
    subtract_background = False
    use_color = False
    imin = 0    # view positions
    imax = 5
    jmin = 0
    jmax = 5
    kmin = 0
    kmax = 9
    cmin = 2
    cmax = 11
    crop_start = 0 # crop settings
    crop_end = 3000
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
    img_args.add_argument('--imin', default=[], type=int, help='minimum i value')
    img_args.add_argument('--imax', default=[], type=int, help='maximum i value')
    img_args.add_argument('--jmin', default=[], type=int, help='minimum j value')
    img_args.add_argument('--jmax', default=[], type=int, help='maximum j value')
    img_args.add_argument('--kmax', default=[], type=int, help='maximum k value')
    img_args.add_argument('--kmin', default=[], type=int, help='minimum k value')
    img_args.add_argument('--cmax', default=[], type=int, help='maximum cycle value')
    img_args.add_argument('--cmin', default=[], type=int, help='minimum cycle value')
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
        args.imin = imin
        args.imax = imax
        args.jmin = jmin
        args.jmax = jmax
        args.kmin = kmin
        args.kmax = kmax
        args.cmin = cmin
        args.cmax = cmax
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
    
    perform_stack(colors, args.use_gpu, args.key, args.gcs_project, args.src, args.exp, args.cha, args.dst, args.typ, args.imin, args.imax, args.jmin, args.jmax, args.kmin, args.kmax, args.cmin, args.cmax, args.crop_start, args.crop_end, args.remove_background, args.subtract_background, args.invert_contrast, args.shift_registration, args.use_color, args.WSize, args.alpha, args.sth, args.verbose)
    return
    
def perform_stack(colors, use_gpu, key, gcs_project, src, exp, cha, dst, typ, imin, imax, jmin, jmax, kmin, kmax, cmin, cmax, crop_start, crop_end, remove_background, subtract_background, invert_contrast, shift_registration, use_color, WSize, alpha, sth, verbose):
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
    is_remote = False
    fs = None
    if src[0:5] == 'gs://':
        is_remote = True

    if is_remote and len(gcs_project) == 0:
        print("Remote source but no project given")
        error += 1

    if is_remote and len(key) == 0:
        print("Remote source but no key ")
        error += 1
        
    # if there are any errors, stop
    if error > 0:
        print(str(error) + " errors detected")
        return

    # Initialize the remote filesystem
    fs = None
    if is_remote:
        fs = gcsfs.GCSFileSystem(project=gcs_project,token=key)

    # pre-allocate arrays
    if use_color:
        I_zs = np.zeros((kmax-kmin + 1,a,a,3))
    else:
        I_zs = np.zeros((kmax-kmin + 1,a,a))
    # perform fstack for each experiment and for each channel and for each i,j
    for exp_i in exp:
        # load index.csv for each top-level experiment index
        df = None
        path = src + '/' + exp_i + '/' + 'index.csv'
        try:
            if is_remote:
                with fs.open(path, 'r' ) as f:
                    df = pd.read_csv(f)
            else:
                with open( path, 'r' ) as f:
                    df = pd.read_csv(f)
        except:
            print(path + " cannot be opened")
            break # exit this loop
        if verbose:
            print(path + " opened")
            n = df.shape[0] # n is the number of cycles
            print("n cycles = " + str(n))
        
        
        for i, j in product(range(imin, imax+1), range(jmin, jmax+1)):
            for c in range(cmin, cmax+1):
                id = df.loc[c, 'Acquisition_ID']
                if verbose:
                    print(id)  
                for l in range(len(cha)):
                    color = colors[str(l)]
                    channel = cha[l]
                    if verbose:
                        print(channel)

                    if l == 0 and invert_contrast:
                        color = [0,0,0]

                    for k in range(0, kmax+1-kmin):
                        if verbose:
                            print(k)
                        filename = id + '/0/' + str(i) + '_' + str(j) + '_' + str(k+kmin) + '_' + channel + '.' + typ
                        if is_remote:
                            I = imread_gcsfs(fs,src + '/' + exp_i + '/'+ filename)
                        else:
                            I = cv2.imread(src + '/' + exp_i + '/'+ filename)
                        # crop the image
                        I = I[crop_start:crop_end,crop_start:crop_end]
                        if use_color:
                            I_zs[k,:,:,:] = I
                        else:
                            I_zs[k,:,:] = I

                    I = fstack_images(I_zs, list(range(kmin, kmax+1)), verbose=verbose, WSize=WSize, alpha=alpha, sth=sth)
                    if remove_background:
                        selem = disk(30) 
                        I = white_tophat(I,selem)
                        
                    if subtract_background:
                        I = I - np.amin(I)
                    
                    # normalize
                    I = I.astype('float')
                    I = 255*I/np.amax(I)
                    # registration across channels
                    if shift_registration:
                        # take the first channel of the first cycle as reference
                        if c == cmin:
                            if l == 0:
                                I0 = I
                        else:
                            if l == 0:
                                # compare the first channel of later cycles to the first channel of the first cycle
                                shift, __, __ = phase_cross_correlation(I, I0, upsample_factor = 5)
                                if verbose:
                                    print(shift)
                                # create the transform
                                transform = AffineTransform(translation=(shift[1],shift[0]))
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
                    
                    # save images
                    fname = id + '_' + str(i) + '_' + str(j) + '_' + channel + '.' + typ
                    savepath = dst + '/' + fname
                    if dst[0:5] == 'gs://':
                        cv2.imwrite(fname, I)
                        fs.put(fname, savepath)
                        os.remove(fname)
                    else:
                        cv2.imwrite(savepath, I)


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
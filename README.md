# CODEX Pipeline

todo: explain the `use_all` flag

This repository contains everything necessary to process image data, from focus stacking to basic analysis.

Future revisions will include .zarr support (more compression options, faster read/writes).

## Repo Contents

### focus stack

- `requirements-fstack.txt`: List of requirements to run fstack and fstack_cu. Activate your conda environment and run `pip install -r requirements-fstack.txt`
- fstack
  - `__init__.py`: this is the fstack code.
  - Python code in the pipeline/ directory can import fstack with `from fstack import fstack`
- fstack_cu
  - `__init__.py`: this is the CUDA-accelerated fstack code.
  - Python code in the pipeline/ directory can import fstack with `from fstack_cu import fstack_cu` or `from fstack_cu import fstack_cu_images`
- `fstacker.py`: perform a focus stack using local (provide path) or remote (using GCSFS) images. This code assumes the existence of an `index.csv` file in the source directory containing cycle names.

### generate training data

- `random_image_segments.py`: pick a channel and get a random selection of images from that channel across all cycles. This is useful for generating training data if you have to train your own cellpose model.

### segmentation

- `requirements-segment.txt`: see `requirements-fstack.txt` description
- `segmenter.py`: segment nuclei using cellpose using a pretrained model. Use the cellpose gui to create the model. `segmenter.py` assumes a folder and file structure created by `fstacker.py`.

### analysis

- `requirements-analyze.txt`: see `requirements-fstack.txt` description
- `analyzer.py`: measure the size and average brightness of each cell in each channel and save as a csv. `analyzer.py` assumes a folder and file structure created by `segmenter.py`.

## Guides

### fstacker usage

#### fstacker theory of operation

Focus-stack every imageset from a codex experiment. Suppose you have the following file structure:

- `src` source directory (can be any valid string)
  - `exp_id_1` experiment ID (can be any string)
    - `index.csv` CSV with the valid cycle names (must be named "index.csv")
    - `cycle1` folder with name of fisrt cycle (can be any string as long as it matches index.csv)
      - `0` folder named "0" (must be named "0")
        - `0_0_0_Fluorescence_405_nm_Ex.bmp` bmp image with this name format. The first digit represents the i coordinate, the second digit represents the j coordinate, the third represents the z height, and the rest of the filename represnts the channel wavelength
        - `0_0_0_Fluorescence_488_nm_Ex.bmp` bmp with same coordinates as above but with a different channel
        - more BMPs
        - `6_7_5_Fluorescence_488_nm_Ex.bmp` for our example, suppose i ranges from 0 to 6, j ranges from 0 to 7, and z index ranges from 0 to 5
  - `exp_id_2` another experiment (can have any number)
    - `0`
      - identical structure to `exp_id_1`

For each experiment ID, for each channel `Fluorescence_NNN_nm_Ex`, and for each i index `i` and j index `j` in the range, `fstacker.py` generates an image called "`i`\_`j`\_f_Fluorescence\_`NNN`\_nm_Ex.png" image (with different values for `i`, `j`, and `NNN` for each image stacked) and saves it to either the `src` directory or a different directory of your choosing.

#### set fstacker parameters

there are many parameters to set which images get focus stacked and where to save them. here's the complete list of the parameters and what they mean:

- `CLI`: set to `True` or `False`. Keep it set to false, the command line interface development is lagging behind the script development.
- `use_gpu`: set to `True` or `False`. Focus stacking can be accelerated with the GPU. Depending on the dataset and computer, it might be faster to set it to `False` and use the CPU; try running with either option and compare focus stack times.
- `prefix`: string. If you have an index.csv, leave this string empty. If you don't have an index.csv with the names of the cycles to analyze, you can select which cycles to run by prefix. For example, if you have three cycles with names, `cycle_good1`, `cycle_also_good`, and `cycle_bad` and you only want to run focus stacking on the two good datasets, you can set `prefix="cy"` and rename `cycle_bad` to `_cycle_bad` so it will be excluded.
- `key`: string. If you are connecting to a Google Cloud File Storage, set this to the local path to the authentication token .json. If you are running segmentation locally, this doesn't matter.
- `gcs_project`: string. Set this to the Google Cloud Storage project name if you are connecting to GCS. Otherwise, it doesn't matter.
- `src`: string. path to the folder that contains the experiment folder. Can be a GCS locator (e.g.`gs://octopi-data`) or a local path (e.g. `/home/user/Documents/src/`). Note - must have a trailing slash (`/`)
- `dst`: string. path to folder to save data. If left blank, the images will be stored in the same directory as `src`. `fstacker.py` will recreate the source folder structure in this folder. Also must have a trailing slash.
- `exp`: list of strings. List of experiment IDs. In the example above, this would be `["exp_id_1", "exp_id_2"]`
- `cha`: list of strings. List of channel names. In the example above, this would be `["Fluorescence_405_nm_Ex", "Fluorescence_488_nm_Ex"]` but it should be left as-is.
- `typ`: string. Filetype of the images to read. Leave as-is.
- `colors`: Dictionary of lists. Leave as-is.
- `remove_background`: boolean. Set `True` to run a white-tophat filter. This will increase runtime considerably.
- `invert_contrast`: boolean. Set `True` to invert the image.
- `shift_registration`: boolean. Set `True` to ensure images stay aligned across cycles.
- `subtract_background`: boolean. Set `True` to subtract the minimum brightess off from the entire image.
- `use_color`: boolean. Set `True` when processing RGB images (untested).
- `imin`, `imax`, `jmin`, and `jmax`: integers. The range of i and j values to process. Generally, they should be the actual values for the experiment
- `kmin` and `kmax`: integers. The range of z index values to image stack. If (kmax+1-kmin) is less than or equal to 4, focus stacking will not run and the midpoint k index will be used. Set `kmin`=`kmax` if there's only one valid z index to bypass focus stacking.
- `crop_start` and `crop_end`: integers. Crop the image; x and y coordinates less than or greater than `crop_start` or `crop_end` respectively will be cropped out.
- `WSize`: integer. Must be odd. Size of blurring kernel for focus stacking. 9 is a good value.
- `alpha`: float. Must be between 0 and 1. Parameter for blending z stacks. 0.2 is a good value.
- `sth`: integer. Parameter for blending z stacks. 13 is a good value.
- `verbose`: boolean. Set `True` to print extra details during operation (good for determining whether GPU stacking is faster than CPU)

#### run fstacker

1. to use fstacker as a script, set `CLI = TRUE` in `main()` in `fstacker.py` and set the constants. Then, run the file
2. to use the command line interface, set `CLI = TRUE` in `main()` in `fstacker.py`. Navigate to the directory `fstacker.py` is in, activate the conda environment if necessary, and run `python -m fstacker --help` to see all the flags and how to set them

### generate training data usage

#### theory of operation

cellpose uses machine learning models to segment the nuclei (405 nm channel). In my experience, the "cyto" built-in model detects the cells but does a bad job at isolating the nuclei while the "nuclei" model does a good job at isolating the nuclei but fails to detect all the nuclei in the image. In theory, training the cellpose model on just one view of the 405 nm channel should be sufficent but in practice this leads to overfitting. Using several different views of the 405 nm channel solves this problem. Manually segmenting full images for training is tedious and redundant and using smaller subsets of each view is sufficient. `random_image_segmentation.py` randomly selects a set quantity of images, crops the image, and saves the image locally (no option for remote saving). The Cellpose GUI only works on local images so there's no point in saving them remotely. This script assumes the images are saved in the same folder structure as described in the fstacker theory of operation.

#### set parameters

- `root_dir`: string. local or remote path to where the images are stored
- `dest_dir` : string. local path to store the randomly selected and cropped images. this should be different from `root_dir` to make it easier to work with them.
- `exp_id`: string. experiment ID to get the images from. see fstacker theory of operation for more detail.
- `channel`: string. image channel to randomly select from. see fstacker theory of operation for more detail.
- `zstack`: string. if you ran `fstacker.py` and want to use the focus-stacked images, set this value to `"f"`. otherwise set it to the z stack you want to use (for example, if z=5 is in focus for all cells across all images, you can set `zstack="5"` and those images will be utilized)
- `key`: string. If you are connecting to a Google Cloud File Storage, set this to the local path to the authentication token .json.
- `gcs_project`: string. Set this to the Google Cloud Storage project name if you are connecting to GCS. Otherwise, it doesn't matter.
- `n_rand`: int. number of images to sample. 20 seems to be a good number.
- `n_sub`: int. cut the images into `n_sub`^2 equal-size sub-images and randomly select one to save. 3 seems to be a good choice but 4 probably will also work. Make sure the sub-images have a handfull of cells in them.

#### run it

1. set the parameters and run the script. the images will be saved to `dest_dir + exp_id` directory.

### segmenter usage

#### segmenter theory of operation

cells don't move between cycles so we only need to segment one cycle for each (i,j) view. We generally choose to segment the nuclear channel because it is brightest. The script first loads all the channel paths, sorts them to find the 0th channel, then filters the image paths so only the 0th channel images are loaded. Images are then segmented one at a time. In principle, Cellpose can work faster by segmenting multi-image batches but in my experience not all GPUs can handle segmenting multiple images. Cellpose then saves a .npy file with the masks to the destination directory.

if you are having trouble installing cellpose, try uninstalling all other python packages that use QTpy (e.g. cv2), install cellpose, then reinstall everthing you uninstalled. If you are using CUDA, ensure your CUDA version is compatible with your version of torch.

#### set segmenter parameters

- `root_dir`: string. local or remote path to where the images are stored
- `exp_id`: string. experiment ID to get the images from. see fstacker theory of operation for more detail.
- `channel`: string. image channel to randomly select from. see fstacker theory of operation for more detail.
- `zstack`: string. if you ran `fstacker.py` and want to use the focus-stacked images, set this value to `"f"`. otherwise set it to the z stack you want to use (for example, if z=5 is in focus for all cells across all images, you can set `zstack="5"` and those images will be utilized)
- `key`: string. If you are connecting to a Google Cloud File Storage, set this to the local path to the authentication token .json.
- `gcs_project`: string. Set this to the Google Cloud Storage project name if you are connecting to GCS. Otherwise, it doesn't matter.
- `cpmodel`: string. path to cellpose model. Can be remote or local.
- `use_gpu`: boolean. Set to `True` to try segmenting using the GPU.
- `channels`: list of ints. Which channel to run segmentation on. Set to `[0,0]` for the monochrome channel

#### run segmenter

1. to use segmenter as a script, set the constants and run the file. Note that it takes a pretrained model as a parameter; you can change it to one of the built-in pretrained models or set a path to a custom pretrained model
2. cellpose already has a CLI; `segmenter.py` itself does not have a CLI

### analyzer usage

#### analyzer theory of operation

We assume cell positions don't change between cycles and we can mask the cells by expanding the nucleus masks from `segmenter.py`. We have a mask for each (i,j) view; for each view we load all the images across all cycles at that view and load the nucleus mask. The mask is expanded to mask the entire cell. Then for each cell in the view, for each channel, for each cycle we calculate the average brightness of the cell and store it in a csv.

The CSV columns are cell index in a given view, the i index, j index, x position in the image, y position in the image, the number of pixels in the expanded mask, and the number of pixels in the nuclear mask. Then, there is a column for each element in the cartesian product of channel index and cycle index. The header is re-printed in the csv and the cell index resets for each view.

#### set analyzer parameters

- `start_idx`, `end_idx`: integers. Select a range of cycles to analyze
- `n_ch`: integer. Number of channels to analyze
- `expansion`: integer, must be odd. The number of pixels to expand around the nucleus mask to create the cell masks.
- `root_dir`: string. local or remote path to where the images are stored
- `exp_id`: string. experiment ID to get the images from. see fstacker theory of operation for more detail.
- `channel`: string. image channel to randomly select from. see fstacker theory of operation for more detail.
- `zstack`: string. if you ran `fstacker.py` and want to use the focus-stacked images, set this value to `"f"`. otherwise set it to the z stack you want to use (for example, if z=5 is in focus for all cells across all images, you can set `zstack="5"` and those images will be utilized)
- `key`: string. If you are connecting to a Google Cloud File Storage, set this to the local path to the authentication token .json.
- `gcs_project`: string. Set this to the Google Cloud Storage project name if you are connecting to GCS. Otherwise, it doesn't matter.
- `out`: string. Path to store the csv. Local or remote path.

#### run analyzer

1. to use analyzer as a script, set the constants and run the file
2. there currently is no CLI version

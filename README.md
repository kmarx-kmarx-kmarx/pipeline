# CODEX Pipeline

This repository contains everything necessary to process image data, from focus stacking to basic analysis.

note: this code has only been tested with single channel monochromatic images.

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

### segmentation

- `requirements-segment.txt`: see `requirements-fstack.txt` description
- `segmenter.py`: segment nuclei using cellpose using a pretrained model. Use the cellpose gui to create the model. `segmenter.py` assumes a folder and file structure created by `fstacker.py`.

## Guides

### fstacker usage

note that `use_gpu = True` isn't always faster than using the CPU for focus stacking. It is unclear what determines the speed.

1. to use fstacker as a script, set `CLI = TRUE` in `main()` in `fstacker.py` and set the constants. Then, run the file.
2. to use the command line interface, set `CLI = TRUE` in `main()` in `fstacker.py`. Navigate to the directory `fstacker.py` is in, activate the conda environment if necessary, and run `python -m fstacker --help` to see all the flags and how to set them.

### segmenter.py

if you are having trouble installing cellpose, try uninstalling all other python packages that use QTpy (e.g. cv2), install cellpose, then reinstall everthing you uninstalled.

1. to use segmenter as a script, set the constants and run the file. Note that it takes a pretrained model as a parameter; you can change it to one of the built-in pretrained models or set a path to a custom pretrained model.
2. cellpose already has a CLI; `segmenter.py` itself does not have a CLI

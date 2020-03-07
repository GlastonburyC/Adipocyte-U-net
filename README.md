[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/GlastonburyC/Adipocyte-U-net/master)

# Adipocyte-U-net

The Adipocyte U-net is a deep U-net architecture trained to segment Adipocytes from histology imaging slides (both H&E and florescent). 

![alt text](overview.png)

### Running the tutorial using Binder

We have made this repository work with Binder. By clicking the binder logo above, a docker image will launch on the Binder website and you'll be able to use the tutorial notebook `Tutorial.ipynb` as if it were installed on your own laptop.

`Tutorial.ipynb` is a walk through example of how to use adipocyte U-net to perform image segmentation. In the tutorial we predict segmentations and use these predictions to obtain surface area estimates of the cell population present in the image.
This notebook will work on either a CPU or GPU, but will be many times faster in a GPU environment.

### Installation instructions

We strongly recommend following these instructions using python 3.5+:

1. Install `virtualenv` and create a virtual environment `virtualenv unet`
2. source the environment `source unet/bin/activate`
3. `git clone https://github.com/GlastonburyC/Adipocyte-U-net.git`
4. `cd Adipocyte-U-net`
5. Install the requirements `pip install -r requirements.txt`
6. If some installs fail, it maybe the version of OS X you're using, in that case `export MACOSX_DEPLOYMENT_TARGET=10.14` and reinstall the requirements (step 5).

 
All the data to reproduce the manuscript are available below:

1. training images for classifying adipocyte containing tiles [here](https://drive.google.com/open?id=1hsmMGTQSOvicUr50fiCol_Gr5z8U0koC)
2. trained InceptionV3 adipocyte tile classifier [weights here.](https://drive.google.com/open?id=1dGZ1amjkRfRzSO9etWwtsadylG6wGvF0)
3. U-net weights are in /checkpoints/ folder.
4. All annotations, training and validation images splits [here](https://drive.google.com/open?id=1MDY_CYcLSKbCrjMBvGZ5sFaqh5rRmrRk)
5. All montage images and numpy arrays [here](https://drive.google.com/open?id=1qCb13kFdN3mxukcnz7IwfarfaZU3ygsr)

If you are predicting on your own Adipocyte images and they significantly deviate from the H&E images used here, consider fine-tuning the Adipocyte U-net, otherwise, similar to the tutorial notebook, you can use the trained weights provided here.

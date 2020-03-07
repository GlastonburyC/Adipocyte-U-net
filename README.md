# Adipocyte-U-net

The Adipocyte U-net is a deep U-net architecture trained to segment Adipocytes from histology imaging slides (both H&E and florescent). 

![alt text](overview.png)


### Installation instructions

This software was developed using Python3.5. It has a lot of dependencies and *will not work* on later versions of tensorflow.

We strongly recommend following these instructions:

1. Download and install [Python 3.5.2](https://www.python.org/downloads/release/python-352/)
2. Install `virtualenv` and create a virtual environment `virtualenv adipo --python=python3.5`
3. `git clone https://github.com/GlastonburyC/Adipocyte-U-net.git`
4. `cd Adipocyte-U-net`
3. Install the requirements `pip3.5 install -r requirements.txt`
4. If you're on a mac and this fails due to compiler/clang issues, set `export MACOSX_DEPLOYMENT_TARGET=10.14`
5. re-run requirements install (if neccessary) and you're good to go.

 ### Example tutorial using Adipocyte U-net on GTEx images:
Checkout `Tutorial.ipynb` for an example of how to use adipocyte U-net to perform image segmentation. In the tutorial we use these predicted segmentations to obtain surface area estimates of the cell population present in the image.
This notebook will work on either a CPU or GPU, but will be many times faster in a GPU environment.
 
All the data to reproduce the manuscript are available below:

1. training images for classifying adipocyte containing tiles [here](https://drive.google.com/open?id=1hsmMGTQSOvicUr50fiCol_Gr5z8U0koC)
2. trained InceptionV3 adipocyte tile classifier [weights here.](https://drive.google.com/open?id=1dGZ1amjkRfRzSO9etWwtsadylG6wGvF0)
3. U-net weights are in /checkpoints/ folder.
4. All annotations, training and validation images splits [here](https://drive.google.com/open?id=1MDY_CYcLSKbCrjMBvGZ5sFaqh5rRmrRk)
5. All montage images and numpy arrays [here](https://drive.google.com/open?id=1qCb13kFdN3mxukcnz7IwfarfaZU3ygsr)

If you are predicting on your own Adipocyte images and they significantly deviate from the H&E images used here, consider fine-tuning the Adipocyte U-net, otherwise, similar to the tutorial notebook, you can use the trained weights provided here.

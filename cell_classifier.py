import os
import numpy as np
import shutil
import random
import argparse
import time
import sys
import keras
import matplotlib
import matplotlib.pyplot as plt
import os
from scipy import ndimage, misc
import scipy.misc
import re,glob
import cv2
from keras.models import load_model
from keras.layers import Flatten, Dense, AveragePooling2D
from keras.models import Model, load_model
from keras.optimizers import RMSprop, SGD, adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator

from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys


def get_parser():
    """Define the input arguments"""
    parser = argparse.ArgumentParser('Classify adipocyte tiles')
    parser.add_argument(
        '--out-dir',
        type=str,
        default="output",
        help='The output directory.',
    )
    parser.add_argument(
        '--weight_dir',
        type=str,
        required=True,
        help='The path to the InceptionV3 model trained weights',
    )
    parser.add_argument(
        '--image-path',
        type=str,
        required=True,
        help='The path to the images we want to classify',
    )
    return parser


def main():
    args = get_parser().parse_args()

    img_width = 299
    img_height = 299
    batch_size = 1
    nbr_test_samples = len(glob.glob(args.image_path + '*'))

    print('Number of total patches to classify: {}'.format(nbr_test_samples))

    test_data_dir = os.path.join('example_class_tiles/','test')

    test_datagen = ImageDataGenerator(
            rescale=1./255)

    print('Loading model and weights from training process ...')
    InceptionV3_model = load_model(args.weight_dir)


    import time
    nbr_augmentation=1
    random_seed=[]

    start = time.time()
    for idx in range(nbr_augmentation):
        print('{}th augmentation for testing ...'.format(idx))
        if idx == 0:
            random_seed.append(np.random.random_integers(0, 1000000))
        else:
            random_seed.insert(idx,np.random.random_integers(0, 1000000))
        test_generator = test_datagen.flow_from_directory(
                test_data_dir,
                target_size=(299, 299),
                batch_size=batch_size,
                shuffle = False,
                seed = random_seed[idx],
                classes = None,
                class_mode = None)
        test_image_list = test_generator.filenames
        print('Begin to predict for testing data ...')
        if idx == 0:
            predictions = InceptionV3_model.predict_generator(test_generator, 
                                                              2414950,
                                                              use_multiprocessing=True,
                                                              workers=40)
        else:
            predictions += InceptionV3_model.predict_generator(test_generator, 
                                                               2414950,
                                                               use_multiprocessing=True, 
                                                               workers=40)


    predictions /= nbr_augmentation

    print('Begin to write submission file ..')
    seed_submit=open(os.path.join(args.out_dir, 'seed.csv'), 'w')

    for item in random_seed:
        seed_submit.write("%s\n" % item)

    f_submit = open(os.path.join(args.out_dir, 'out.inception.adipocytes.csv'), 'w')
    f_submit.write('image,empty,not_adipocyte,adipocyte\n')

    for i, image_name in enumerate(test_image_list):
        pred = ['%.6f' % p for p in predictions[i, :]]
        #print('{} / {}'.format(i, nbr_test_samples))
        f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

    f_submit.close()

    end = time.time()

    print(end-start)


if __name__ == "__main__":
    main()


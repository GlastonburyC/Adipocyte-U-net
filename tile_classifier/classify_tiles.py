import os
import numpy as np
import shutil
import random
import sys
import keras
import matplotlib
#matplotlib.use('Agg');
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

print('Number of total patches to classify: {}'.format(len(glob.glob('test/test/*.jpg'))))


from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

img_width = 299
img_height = 299
batch_size = 1
nbr_test_samples = 2414950

root_path = '/well/lindgren/craig/tile_classifier/'
weights_path = '/well/lindgren/craig/tile_classifier/tile_adipocyte.weights.h5'
test_data_dir = os.path.join(root_path,'test')

test_datagen = ImageDataGenerator(
        rescale=1./255)


#### print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)


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
seed_submit=open(os.path.join(root_path, 'seed.csv'), 'w')

for item in random_seed:
    seed_submit.write("%s\n" % item)

f_submit = open(os.path.join(root_path, 'out.inception.adipocytes.csv'), 'w')
f_submit.write('image,empty,not_adipocyte,adipocyte\n')

for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    #print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

end = time.time()

print(end-start)

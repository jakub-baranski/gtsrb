import logging
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from skimage import io
from skimage import transform, exposure
from sklearn.utils import shuffle
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

from TSCNN import TSCNN

LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
LOGGER.addHandler(logging.StreamHandler(sys.stdout))


def get_training_images_paths() -> List:
    return [p.absolute() for p in Path(f'./GTSRB_Final_Training_Images').rglob('*/Images/*/*.ppm')]


def get_class_from_filepath(filepath: str) -> int:
    return int(filepath.split('/')[-2])


def get_classes_paths() -> List[Path]:
    return [p for p in Path(f'./GTSRB_Final_Training_Images').rglob('*/Images/*')]


def load_train_data():
    """
    Initial method that didn't crop image according to provided metadata. Worker but could be
    improved...
    """
    data_list = []
    labels_list = []

    images_paths = get_training_images_paths()
    random.seed(42)
    random.shuffle(images_paths)

    LOGGER.info(f'Loading {len(images_paths)} train images')

    for path in images_paths:
        image = io.imread(path)
        image = transform.resize(image, (32, 32))
        image = exposure.equalize_adapthist(image, clip_limit=0.15)
        # TODO: Since we have also info about where sign is located on the picture, we should probably crop it https://stackoverflow.com/questions/33287613/crop-image-in-skimage

        data_list.append(image)
        labels_list.append(get_class_from_filepath(str(path)))

    return np.array(data_list), np.array(labels_list)


def load_cropped_train_data():
    """
    Added cropping. SInce it required a data from csv file I had to change the way files are listed.
    """
    data_list = []
    labels_list = []

    classes_directories_paths = get_classes_paths()

    LOGGER.info(f'Loading training data for {len(classes_directories_paths)} classes')

    for i, class_dir in enumerate(classes_directories_paths):
        images_csv_file_path = next(class_dir.glob('*.csv'))
        with images_csv_file_path.open('r') as images_csv_file:
            _heading = images_csv_file.readline()
            for image_data in images_csv_file.readlines():
                split = image_data.strip().split(';')
                name = split[0]
                roi_x1, roi_y1, roi_x2, roi_y2 = map(int, split[3:7])
                class_id = split[7]

                image = io.imread(str(class_dir.joinpath(name)))
                image = image[roi_y1:roi_y2, roi_x1:roi_x2]
                image = transform.resize(image, (32, 32))
                image = exposure.equalize_adapthist(image, clip_limit=0.1)

                data_list.append(image)
                labels_list.append(class_id)
        LOGGER.info(f'{i + 1}/{len(classes_directories_paths)} done')

    shuffle(data_list, labels_list, random_state=0)

    return np.array(data_list), np.array(labels_list)


EPOCHS = 30
LEARNING_RATE = 0.01
BATCH_SIZE = 64
num_labels = 43


LOGGER.info('Loading train data')
train_X, train_Y = load_cropped_train_data()

LOGGER.info('Loading test data')

train_X = train_X.astype(np.float64) / 255.0

train_Y = to_categorical(train_Y, num_labels)

# Since there are not even number of images in each class we should assign some kind of weight to
# a class...
# https://datascience.stackexchange.com/questions/13490/how-to-set-class-weights-for-imbalanced-classes-in-keras
class_totals = train_Y.sum(axis=0)
class_weight = class_totals.max() / class_totals
class_weight = {i: class_weight[i] for i in range(len(class_weight))} # TODO: FIx


image_data_augmenter = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    horizontal_flip=False,
    vertical_flip=False,
    fill_mode="nearest")

optimizer = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / (EPOCHS / 2))
model = TSCNN.create(width=32, height=32, depth=3, classes=num_labels)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])


# TODO: Split training data and use some as validate data?

model.fit(
    image_data_augmenter.flow(train_X, train_Y, batch_size=BATCH_SIZE),
    steps_per_epoch=train_X.shape[0] / BATCH_SIZE,
    epochs=EPOCHS,
    class_weight=class_weight,
    verbose=1)


model.save(f'trained-model/{datetime.now()}')

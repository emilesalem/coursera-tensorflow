import os
import zipfile
import random
import tensorflow as tf
import shutil
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from shutil import copyfile
from os import getcwd

root = "tmp/cats-v-dogs"

#wget --no-check-certificate   https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip   -O ./cats_and_dogs_filtered.zip
# --2021-02-15 16:12:58--  https://storage.googleapis.com/mledu-datasets/cats_and_dogs_filtered.zip

# def makeSampleDirs(label):
#     for dir in [
#             os.path.join(x, label) for x in [
#                 os.path.join(root, y) for y in ["training", "testing"]
#             ]
#         ]:
#             os.makedirs(dir)

def split_data(SOURCE, TRAINING, TESTING, SPLIT_SIZE):
    f = os.listdir(SOURCE)
    random.shuffle(f)
    filtered = filter(lambda x: os.path.getsize(os.path.join(SOURCE, x)) > 0, f)
    for i, file in enumerate(filtered):
        if i/len(f) < SPLIT_SIZE:
            copyfile(os.path.join(SOURCE, file), os.path.join(TRAINING, file))
        else:
            copyfile(os.path.join(SOURCE, file), os.path.join(TESTING, file))


CAT_SOURCE_DIR = "tmp/files/cats/"
TRAINING_CATS_DIR = "tmp/cats-v-dogs/training/cats/"
TESTING_CATS_DIR = "tmp/cats-v-dogs/testing/cats/"
DOG_SOURCE_DIR = "tmp/files/dogs/"
TRAINING_DOGS_DIR = "tmp/cats-v-dogs/training/dogs/"
TESTING_DOGS_DIR = "tmp/cats-v-dogs/testing/dogs/"

# split_size = .9
# split_data(CAT_SOURCE_DIR, TRAINING_CATS_DIR, TESTING_CATS_DIR, split_size)
# split_data(DOG_SOURCE_DIR, TRAINING_DOGS_DIR, TESTING_DOGS_DIR, split_size)


model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2), 
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'), 
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(), 
    tf.keras.layers.Dense(512, activation='relu'), 
    tf.keras.layers.Dense(1, activation='sigmoid')  
])


model.compile(optimizer=RMSprop(lr=0.001), loss='binary_crossentropy', metrics=['acc'])

TRAINING_DIR = "tmp/cats-v-dogs/training"
train_datagen = ImageDataGenerator( rescale = 1.0/255. )

# NOTE: YOU MUST USE A BATCH SIZE OF 10 (batch_size=10) FOR THE 
# TRAIN GENERATOR.
train_generator = train_datagen.flow_from_directory(TRAINING_DIR,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

VALIDATION_DIR = "tmp/cats-v-dogs/testing"
validation_datagen = ImageDataGenerator( rescale = 1.0/255. )

# NOTE: YOU MUST USE A BACTH SIZE OF 10 (batch_size=10) FOR THE 
# VALIDATION GENERATOR.
validation_generator = train_datagen.flow_from_directory(VALIDATION_DIR,
                                                    batch_size=10,
                                                    class_mode='binary',
                                                    target_size=(150, 150))


history = model.fit(train_generator,
                              epochs=10,
                              verbose=1,
                              validation_data=validation_generator)



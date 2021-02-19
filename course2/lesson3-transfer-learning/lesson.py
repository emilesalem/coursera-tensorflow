import os
import tensorflow as tf
from tensorflow.keras import Model, layers
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# !wget --no-check-certificate \
#     https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5 \
#     -O /tmp/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if(logs.get('acc') > 0.97):
            print("\nReached 97.0% accuracy so cancelling training!")
            self.model.stop_training = True


def getGenerators(train_dir, test_dir):
    # Add our data-augmentation parameters to ImageDataGenerator
    train_datagen = ImageDataGenerator(rescale=1./255.,
                                       rotation_range=40,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.2,
                                       zoom_range=0.2,
                                       horizontal_flip=True)

    # Note that the validation data should not be augmented!
    test_datagen = ImageDataGenerator(rescale=1.0/255.)

    # Flow training images in batches of 20 using train_datagen generator
    x = train_datagen.flow_from_directory(train_dir,
                                          batch_size=20,
                                          class_mode='binary',
                                          target_size=(150, 150))

    # Flow validation images in batches of 20 using test_datagen generator
    y = test_datagen.flow_from_directory(test_dir,
                                         batch_size=20,
                                         class_mode='binary',
                                         target_size=(150, 150))
    return x, y

def getModel():
  local_weights_file = './inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'

  pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                  include_top=False,
                                  weights=None)

  pre_trained_model.load_weights(local_weights_file)

  for layer in pre_trained_model.layers:
      layer.trainable = False

  last_layer = pre_trained_model.get_layer('mixed7')
  print('last layer output shape: ', last_layer.output_shape)
  last_output = last_layer.output

  # pre_trained_model.summary()
  x = layers.Flatten()(last_output)

  x = layers.Dense(1024, activation='relu')(x)
  # Add a dropout rate of 0.2
  x = layers.Dropout(.2)(x)
  # Add a final sigmoid layer for classification
  x = layers.Dense(1, activation='sigmoid')(x)

  model = Model(pre_trained_model.input, x)

  model.compile(optimizer=RMSprop(lr=0.0001),
                loss='binary_crossentropy',
                metrics=['acc'])
  return model

# Define our example directories and files
base_dir = '/home/emile/Development/coursera/tensorflow/course2/lesson2/files'

train_dir = os.path.join(base_dir, 'training')

validation_dir = os.path.join(base_dir, 'validation')

train_generator, validation_generator = getGenerators(train_dir, validation_dir)

getModel().fit(
    train_generator,
    validation_data=validation_generator,
    epochs=3,
    verbose=2,
    callbacks=[myCallback()])

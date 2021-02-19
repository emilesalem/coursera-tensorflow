# In the feature extraction experiment, you were only training a few layers on top of a MobileNet V2 base model.
# The weights of the pre-trained network were not updated during training.
# One way to increase performance even further is to train (or "fine-tune") the weights of the top layers 
# of the pre-trained model alongside the training of the classifier you added. The training process will 
# force the weights to be tuned from generic feature maps to features associated specifically with the dataset.
# Note: This should only be attempted after you have trained the top-level classifier with the pre-trained 
# model set to non-trainable. If you add a randomly initialized classifier on top of a pre-trained model and attempt 
# to train all layers jointly, the magnitude of the gradient updates will be too large (due to the random weights 
# from the classifier) and your pre-trained model will forget what it has learned.

# # Also, you should try to fine-tune a small number of top layers rather than the whole MobileNet model. 
# In most convolutional networks, the higher up a layer is, the more specialized it is. 
# The first few layers learn very simple and generic features that generalize to almost all types of images. 
# As you go higher up, the features are increasingly more specific to the dataset on which the model was trained. 
# The goal of fine-tuning is to adapt these specialized features to work with the new dataset, rather than 
# overwrite the generic learning.
import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
import os

base_learning_rate = 0.0001
BATCH_SIZE = 32
IMG_SIZE = (160, 160)
initial_epochs = 10

PATH = '/home/emile/.keras/datasets/cats_and_dogs_filtered'

train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

train_dataset = image_dataset_from_directory(
    train_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)

validation_dataset = image_dataset_from_directory(
    validation_dir,
    shuffle=True,
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE
)



base_model =  tf.keras.models.load_model("model/featureextract")


base_model.trainable = True

# Let's take a look to see how many layers are in the base model
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards
fine_tune_at = 100

# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
  layer.trainable =  False

model = tf.keras.models.load_model("model/featureextract-trained")

model.compile(loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              optimizer = tf.keras.optimizers.RMSprop(lr=base_learning_rate/10),
              metrics=['accuracy'])

model.summary()


# model.trainable_variables
len(model.trainable_variables)

fine_tune_epochs = 10
total_epochs =  initial_epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=9,
    validation_data=validation_dataset
)

val_batches = tf.data.experimental.cardinality(validation_dataset)

test_dataset = validation_dataset.take(val_batches // 5).prefetch(buffers_size=tf.data.AUTOTUNE)

loss, accuracy = model.evaluate(test_dataset)

print('Test accuracy :', accuracy)

model.save("model/finetuned")

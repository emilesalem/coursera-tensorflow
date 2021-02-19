from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
      rescale=1./255,
    #   rotation_range=30,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.8,
      zoom_range=0.8,
      horizontal_flip=True,
      fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1/255)

train_generator = train_datagen.flow_from_directory(
        'files/training',
        target_size=(300, 300),
        batch_size=128,
        class_mode='binary')

validation_generator = validation_datagen.flow_from_directory(
        'files/validation',
        target_size=(300, 300),
        batch_size=32,
        class_mode='binary')

model = keras.models.load_model("model/model3")

model.fit(
      train_generator,
      epochs=50,
      validation_data = validation_generator,
      validation_steps=8)

model.save('model/model4')


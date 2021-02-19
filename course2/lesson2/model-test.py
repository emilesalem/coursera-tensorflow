from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np 


img_width, img_height = 300, 300

def printPredictions(title,models, filePath):
    print("\n")
    img = image.load_img(filePath, target_size = (img_width, img_height))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis = 0)
    print(title)
    for x in models:
        print(x.predict(img))

model1 = keras.models.load_model("model/model1")
model2 = keras.models.load_model("model/model2")
model3 = keras.models.load_model("model/model3")

models = [model1, model2, model3]


printPredictions('horse1',models, '/home/emile/Downloads/horse1.jpeg')
printPredictions('person1',models, '/home/emile/Downloads/person.jpeg')
printPredictions('horse2',models, '/home/emile/Downloads/mini-horse.jpg')
printPredictions('person2',models, '/home/emile/Downloads/person2.jpeg')



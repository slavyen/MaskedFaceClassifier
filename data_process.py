import numpy as np
import os
from imutils import paths

from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


data = []
labels = []
imageSize = 128

imagePaths = list(paths.list_images('dataset'))

for imagePath in imagePaths:
    label = imagePath.split(os.path.sep)[-2]

    image = load_img(imagePath, target_size=(imageSize, imageSize))
    image = img_to_array(image)
    image = preprocess_input(image)

    data.append(image)
    labels.append(label)

data = np.array(data, dtype="float32")
labels = np.array(labels)

np.save('data.npy', data)
np.save('labels.npy', labels)
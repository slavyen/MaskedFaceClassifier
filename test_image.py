import numpy as np
import argparse
import cv2
import os
import time
from tqdm import tqdm

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image",
                default="",
                help="path to images")
ap.add_argument("-o", "--out",
                default="",
                help="path to output images")
args = vars(ap.parse_args())

suffix = [".jpg", ".jpeg", ".png"]
img_list = [f for f in os.listdir(args["image"]) if f.endswith(tuple(suffix))]

faceDetectorPath = 'caffe_face_detector'
modelFile = 'mask_detector.model'
minConfidence = 0.5
imageSize = 224

boxLabelWithMask = 'Wearing mask'
boxLabelWithoutMask = 'Not wearing mask'

prototxtPath = os.path.sep.join([faceDetectorPath, "deploy.prototxt"])
weightsPath = os.path.sep.join([faceDetectorPath,
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

maskDetectorModel = load_model(modelFile)

for n, img_name in enumerate(tqdm(img_list)):
    print(img_name)
    image = cv2.imread(os.path.join(args["image"], img_name))
    orig = image.copy()
    (h, w) = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))

    faceNet.setInput(blob)
    detections = faceNet.forward()

    for i in range(0, detections.shape[2]):

        confidence = detections[0, 0, i, 2]
        if confidence > minConfidence:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            try:
                face = image[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (imageSize, imageSize))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

                start_time = time.time()
                withMask = maskDetectorModel.predict(face)[0][0]
                print(f"Inference Time: {time.time() - start_time} s.")

                label = "With Mask" if withMask > 0.5 else "No Mask"
                withoutMask = 1 - withMask
                if label == "With Mask":
                    boxLabel = boxLabelWithMask
                    color = (50, 205, 50)
                else:
                    boxLabel = boxLabelWithoutMask
                    color = (50, 50, 205)
                boxLabel = f"{boxLabel}: {max(withMask, withoutMask) * 100} %"

                cv2.putText(image, boxLabel, (startX, startY - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
                cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)

            except Exception as e:
                print(str(e))

    cv2.imwrite(os.path.join(args["out"], img_name), image)

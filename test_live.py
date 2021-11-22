import numpy as np
import imutils
import time
import cv2
import os

from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream


def detect_and_predict_mask(frame, faceNet, maskDetectorModel):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locations = []
    predictions = []

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > minConfidence:

            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            try:
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                try:
                    face = cv2.resize(face, (imageSize, imageSize))
                except Exception as e:
                    print(str(e))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)

            except Exception as e:
                print(str(e))

            faces.append(face)
            locations.append((startX, startY, endX, endY))

    if len(faces) > 0:
        start_time = (time.time())
        predictions = maskDetectorModel.predict(faces)
        print(f"Inference time: {time.time() - start_time} s")

    return locations, predictions


faceDetectorPath = 'caffe_face_detector'
modelFile = 'mask_detector.model'
minConfidence = 0.5
imageSize = 128

boxLabelWithMask = 'Wearing mask'
boxLabelWithoutMask = 'Not wearing mask'

prototxtPath = os.path.sep.join([faceDetectorPath, "deploy.prototxt"])
weightsPath = os.path.sep.join([faceDetectorPath,
                                "res10_300x300_ssd_iter_140000.caffemodel"])
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
maskDetectorModel = load_model(modelFile)

vs = VideoStream(src=0).start()
time.sleep(2.0)
i = 0

while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=600)

    (locations, predictions) = detect_and_predict_mask(frame, faceNet,
                                                       maskDetectorModel)
    i = i + 1
    for (box, prediction) in zip(locations, predictions):
        (startX, startY, endX, endY) = box
        (withMask, withoutMask) = prediction
        label = "With Mask" if withMask > withoutMask else "No Mask"

        if label == "With Mask":
            boxLabel = boxLabelWithMask
            color = (50, 205, 50)

        else:
            boxLabel = boxLabelWithoutMask
            color = (50, 50, 205)

        boxLabel = f"{boxLabel}: {max(withMask, withoutMask) * 100} %"
        cv2.putText(frame, boxLabel, (startX, startY - 10),
                    cv2.FONT_HERSHEY_PLAIN, 1, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("MaskedFaceClassifier", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cv2.destroyAllWindows()
vs.stop()

import numpy as np
import plotly.express
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

import cv2
video_path=r"C:\Users\murar\OneDrive\Desktop\VID_20230430_151040.mp4"

cap = cv2.VideoCapture(video_path)

from tensorflow import keras
model = keras.models.load_model(r'C:\Users\murar\Downloads\model.h5')

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output3.mp4', fourcc, 30,(1200, 1200))

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break
    # Preprocess the frame for input to the model
    preprocessed_frame = cv2.resize(frame, (1200, 1200))
    preprocessed_frame1 = cv2.resize(frame, (120, 120))
    # Apply the model on the preprocessed frame
    np_img1 = np.array(preprocessed_frame1)
    np_img1 = np.expand_dims(np_img1, axis=0)
    prediction = model.predict(np_img1)

    np_img = np.array(preprocessed_frame)
    if prediction > 0.49999:
        for j in range(0, 1200, 120):
            for k in range(0, 1200, 120):
                a = np_img[j:j + 120, k:k + 120]
                b = np.expand_dims(a, axis=0)
                if model.predict(b) >= 0.5:
                    # np_img[j:j+120,k:k+120]=np_img[j:j+120,k:k+120]*255
                    np_img[j:j + 120, k:k + 120, 1] = np_img[j:j + 120, k:k + 120, 1] / 2
                    np_img[j:j + 120, k:k + 120, 2] = np_img[j:j + 120, k:k + 120, 2] / 2
                    np_img[j:j + 120, k:k + 120] = np_img[j:j + 120, k:k + 120]
                else:
                    np_img[j:j + 120, k:k + 120] = np_img[j:j + 120, k:k + 120]
    data = Image.fromarray(np_img.astype(np.uint8))

    out.write(np_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# Release the capture and destroy all windows
cap.release()
cv2.destroyAllWindows()
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


model = tf.keras.models.load_model('handwritten_model.keras')

image_number = 1
while os.path.isfile(f"digits/digit{image_number}.png"):
    try:
        img = cv2.imread(f"digits/digit{image_number}.png")[:, :, 0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"The number is probably: {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
        
    except:
        print(f"Error processing the image digit{image_number}.png")
        
    finally:
        image_number += 1
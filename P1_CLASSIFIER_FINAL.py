import os
import cv2
import numpy as np
import csv
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

#use own directory of test and training images
model_path = r'C:\Users\ACER\Desktop\OPENCV_WISE_AI\genuine_vs_spoof_classifier.h5'
test_dir = r'C:\Users\ACER\Desktop\OPENCV_WISE_AI\data-annotator-interview\PART1_face_data'
results_path = r'C:\Users\ACER\Desktop\OPENCV_WISE_AI\predictions.csv'  


model = load_model(model_path)

def load_image(image_path, size=(224, 224)):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, size)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img /= 255.0
    return img

def make_predictions(model, directory):
    results = []
    for filename in os.listdir(directory):
        image_path = os.path.join(directory, filename)
        if image_path.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = load_image(image_path)
            prediction = model.predict(img)
            label = 'Genuine' if prediction < 0.5 else 'Spoof'
            results.append([filename, label, prediction[0][0]])

  
    with open(results_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['File Name', 'Label', 'Confidence'])
        for result in results:
            writer.writerow([result[0], result[1], f'{result[2]*100:.2f}%'])


make_predictions(model, test_dir)

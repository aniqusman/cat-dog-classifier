from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from sklearn.utils import shuffle
from tkinter import filedialog
from PIL import Image, ImageTk
import streamlit as st
from glob import glob
from PIL import Image
import tkinter as tk
import numpy as np
import random

print("Start")

cat_images = glob("Images/Cats/*.jpg")
dog_images = glob("Images/Dogs/*.jpg")

images_list = cat_images + dog_images
labels_list = [0] * len(cat_images) + [1] * len(dog_images)

print(labels_list)

x = np.array([img_to_array(load_img(img_path, target_size=(128, 128))) / 255. for img_path in images_list])
y = np.array(labels_list)

x, y = shuffle(x, y, random_state=42)
print(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = Sequential([
    Conv2D(32, (3, 3), activation="relu"),
    Dropout(0.25),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(128, (3, 3), activation="relu"),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

early_stop = EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    brightness_range=[0.8,1.2],
    fill_mode="nearest",
)

datagen.fit(x_train)

history = model.fit(datagen.flow(x_train, y_train, batch_size=16), epochs=50, validation_data=[x_test, y_test], callbacks=[early_stop])

print("Testing predictions on test set:")

root = tk.Tk()
root.title("Aniq's AI App")

title = tk.Label(root, text="Cat vs Dog Classifier", font=("Helvetica", 36, "bold"))
title.pack(pady=10)
image_label = tk.Label(root)
image_label.pack(pady=10)
status_label = tk.Label(root, text="Ready", font=("Helvetica", 12))
status_label.pack(pady=5)

result_label = tk.Label(root, text=f"Computer's prediction: ")
result_label.pack()

def upload_and_predict():
    global image_label
    global status_label
    print("Upload and predict started") 
    global result_label
    try:
        user_image = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        print(f"user_image: {user_image}")
        
        if not user_image:
            return
        
        status_label.config(text="Loading")
        
        image = Image.open(user_image)
        resized_image = image.resize((128, 128))
        photo = ImageTk.PhotoImage(resized_image)
        image_label.config(image=photo)
        image_label.photo = photo
        normalized_image = datagen.standardize(np.array(resized_image) / 255)
        normalized_image = np.expand_dims(normalized_image, axis=0)
        prediction = model.predict(normalized_image)
        threshold = 0.5  
        prediction_value = prediction[0][0]
        print(prediction_value)
        predicted_label = "dog" if prediction_value > threshold else "cat"
        if predicted_label == "dog":
            confidence = prediction_value * 100
        elif predicted_label == "cat":
            confidence = (1 - prediction_value) * 100
        else:
            print("I LOVE AI!")
        
        status_label.config(text="Loading")
        result_text = f"I think this is a {predicted_label}, confidence: {confidence}%"
        print(result_text)
        result_label.config(text=result_text)
    
    except Exception as e:
        print("Error in prediction:", e)
        status_label.config(text="Error: Could not load image")
        
upload_and_predict()        
upload_button = tk.Button(root, text="Upload Image", font=("Helvetica", 14), bg="White", fg="#141204", command=upload_and_predict)

upload_button.pack()




root.mainloop()
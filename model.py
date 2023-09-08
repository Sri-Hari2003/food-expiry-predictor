import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt

dataset_dir = r"dataset"

categories = ['apples', 'banana', 'orange', 'carrot', 'tomato', 'strawberry', 'potato', 'bellpepper', 'cucumber','mango']
labels = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

images = []
all_labels = []

for label, category in zip(labels, categories):
    fresh_path = os.path.join(dataset_dir, category, 'Fresh')
    expired_path = os.path.join(dataset_dir, category, 'Expired')
    for image_name in os.listdir(fresh_path):
        image_path = os.path.join(fresh_path, image_name)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Unable to read image: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            image = cv2.resize(image, (64, 64)) 
            images.append(image)
            all_labels.append(label * 2) 
        except Exception as e:
            print(f"Error loading image: {image_path}")
            print(e)
            continue

    for image_name in os.listdir(expired_path):
        image_path = os.path.join(expired_path, image_name)
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Unable to read image: {image_path}")
                continue
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
            image = cv2.resize(image, (64, 64)) 
            images.append(image)
            all_labels.append(label * 2 + 1) 
        except Exception as e:
            print(f"Error loading image: {image_path}")
            print(e)
            continue

images = np.array(images)
all_labels = np.array(all_labels)

if len(images) == 0 or len(all_labels) == 0:
    print("Empty dataset. No images loaded.")
    exit()

images = images / 255.0

X_train, X_test, y_train, y_test = train_test_split(images, all_labels, test_size=0.2, random_state=42)

if len(X_train) == 0 or len(y_train) == 0:
    print("Empty train set. Adjust the train-test split parameters.")
    exit()

model = tf.keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(20, activation='softmax') 
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

train_loss, train_accuracy = model.evaluate(X_train, y_train, verbose=2)
print('Training accuracy:', train_accuracy)

test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print('Testing accuracy:', test_accuracy)
model.save("trial.h5")

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

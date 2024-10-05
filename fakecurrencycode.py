#!/usr/bin/env python
# coding: utf-8

# In[3]:


get_ipython().system('pip install opencv-python')


# In[4]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import os
import pandas as pd
import cv2


# In[5]:


# Define paths to the datasets
train_path ="C:\\Users\\nandi\\OneDrive\\Desktop\\clg project\\Indian Currency Dataset\\train"
test_path="C:\\Users\\nandi\\OneDrive\\Desktop\\clg project\\Indian Currency Dataset\\test"


# In[6]:


# List categories (labels) in the training directory
categories = os.listdir(train_path)


# In[7]:


# Calculate the number of images per category
nums = {}
for label in categories:
    nums[label] = len(os.listdir(os.path.join(train_path, label)))


# In[9]:


# Convert the nums dictionary to a pandas DataFrame
img_per_class = pd.DataFrame(nums.values(), index=nums.keys(), columns=["no. of images"])
print(img_per_class)


# In[10]:


# Image dimensions and batch size
img_height, img_width = 180, 180
batch_size = 32


# In[11]:


# Load the training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)


# In[12]:


# Load the validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    label_mode='binary'
)


# In[13]:


# Use ResNet50 as base model
base_model = tf.keras.applications.ResNet50(
    include_top=False,
    weights='imagenet',
    input_shape=(img_height, img_width, 3)
)


# In[14]:


# Freeze the layers of the base model
base_model.trainable = False


# In[15]:


# Create a new model
model = Sequential([
    base_model,
    Flatten(),
    Dense(384, activation='relu'),
    Dense(2, activation='softmax')  # binary classification, so 2 neurons with softmax activation
])


# In[16]:


# Compile the model
model.compile(
    optimizer=Adam(),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)


# In[17]:


# Train the model
history = model.fit(train_ds, validation_data=val_ds, epochs=10)


# In[19]:


# Plot training history
plt.figure(figsize=(8, 4))
plt.plot(range(10), history.history['accuracy'], label="Training Accuracy")
plt.title('Training Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[20]:


# Plot validation history
plt.figure(figsize=(8, 4))
plt.plot(range(10), history.history['val_accuracy'], label="Validation Accuracy")
plt.title('Validation Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()


# In[21]:


# Plot training history
plt.figure(figsize=(8, 8))
epochs_range = range(10)
plt.plot(epochs_range, history.history['accuracy'], label="Training Accuracy")
plt.plot(epochs_range, history.history['val_accuracy'], label="Validation Accuracy")
plt.axis(ymin=0.4, ymax=1)
plt.grid()
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'validation'])
plt.show()


# In[22]:


# Save the model
model.save("fake_vs_real_model.keras")


# In[47]:


# Function to preprocess and predict image
def preprocess_and_predict(image_path, model, class_names):
    # Load and preprocess the image using cv2
    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_height, img_width))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, 0)  # Create a batch
    # Predict the class
    predictions = model.predict(img_array)
    predicted_class_index = np.argmax(predictions)
    predicted_class = class_names[predicted_class_index]
    print("Predicted class:", predicted_class)
    return predicted_class


# In[48]:


# Class names
class_names = ["Fake", "Real"]


# In[50]:


# Predict classes for new images
image_path_1 = "C:\\Users\\nandi\\OneDrive\\Desktop\\miniprj\\real.jpg"
# Load and display an image using cv2 and matplotlib
img = cv2.imread(image_path_1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for correct color display
plt.title("Test Image")
plt.xticks([])
plt.yticks([])
plt.show()
preprocess_and_predict(image_path_1, model,class_names)


# In[ ]:





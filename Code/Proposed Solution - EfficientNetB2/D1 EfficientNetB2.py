#!/usr/bin/env python
# coding: utf-8

# # **Import Libraries**

# In[1]:


import tensorflow as tf
from tensorflow import keras
from keras.applications.efficientnet import EfficientNetB2
from keras.layers import Flatten, Dense, Dropout, BatchNormalization, LeakyReLU
import os
from tensorflow.keras.preprocessing import image_dataset_from_directory
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns


# # **Set epochs, base learning rate and number class value**

# In[2]:


base_learning_rate = 0.0001

# GTSRB
NUM_CLASS = 107


# # **Read Directories & Folders**

# In[3]:


train_dir = os.path.join( '../input/dataset1splited/dataset(Splited)/train')
test_dir = os.path.join( '../input/dataset1splited/dataset(Splited)/test')

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


# # **Get trian and test data**

# In[4]:


train_dataset = image_dataset_from_directory(train_dir,
                                              shuffle=True,
                                              batch_size=BATCH_SIZE,
                                              image_size=IMG_SIZE)


test_dataset = image_dataset_from_directory(test_dir,
                                              shuffle=False,
                                              batch_size=BATCH_SIZE,
                                              image_size=IMG_SIZE)


# # **Model Architecture**

# In[5]:


model = keras.Sequential()
# may try other pretrained models (densenet, resnet, mobilenet, inception, etc), refer to https://keras.io/api/applications/
model.add(EfficientNetB2(weights="imagenet",input_shape=(224,224,3),include_top=False))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization())
model.add(LeakyReLU(0.2))
model.add(Dropout(0.3))
model.add(Dense(512))
model.add(BatchNormalization())
model.add(LeakyReLU(0.2))
model.add(Dropout(0.3))
model.add(Dense(NUM_CLASS,activation="softmax"))


# # **Model Compilation**

# In[6]:


model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])


# In[7]:


model.summary()


# # **Data Augmentation**

# In[8]:


train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rotation_range=5,  # rotation
                                   width_shift_range=0.2,  # horizontal shift
                                   zoom_range=0.2,  # zoom
                                   horizontal_flip=True,  # horizontal flip
                                   brightness_range=[0.2,0.8])

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary'
)


# # **Training and Prediction**

# In[9]:


callback=keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=15, restore_best_weights=True)
history = model.fit(train_generator,
                  epochs=150,
                  validation_data=test_dataset,
                  verbose = 1,
                callbacks=[callback]
                  )


# # **Graphing Accuracy**

# In[10]:


acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(len(acc))

plt.figure(figsize=(16, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy over Epochs')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss over Epochs')
plt.show()


# # **Predictions**

# In[11]:


score = model.evaluate(test_dataset)

y_pred = model.predict(test_dataset)
predicted_categories = tf.argmax(y_pred, axis=1)
true_categories = tf.concat([y for x, y in test_dataset], axis=0)


# # **Confusion Matrix**

# In[12]:


cm = confusion_matrix(predicted_categories, true_categories)

plt.figure(figsize = (128, 128))
sns.set(font_scale=4)
sns.heatmap(cm, linecolor='white', cmap='Blues', linewidth=1, annot=True, fmt='')
plt.title('Confusion Matrix', size=20)
plt.xlabel('Predicted Labels', size=14)
plt.ylabel('Actual Labels', size=14)
plt.show()


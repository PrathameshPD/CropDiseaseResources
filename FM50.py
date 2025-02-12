#!/usr/bin/env python
# coding: utf-8

# In[1]:
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout # type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from imblearn.over_sampling import SMOTE # type: ignore
import numpy as np # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore


# In[2]:


# Data directory
data_dir = 'C:\\Users\\Panda\\Desktop\\Mandya\\MandyaImages'
image_size = (224, 224)

# Initialize data and labels
data = []
labels = []

# Data augmentation for training set
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)

# Load training data (without augmentation for SMOTE)
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    subset='training',
    shuffle=False)  # No shuffling for SMOTE

# Load validation data
validation_data = datagen.flow_from_directory(
    data_dir,
    target_size=image_size,
    batch_size=32,
    class_mode='categorical',
    subset='validation')

# Extract data and labels
for batch_data, batch_labels in train_data:
    data.append(batch_data)
    labels.append(batch_labels)
    if len(data) * 32 >= train_data.samples:
        break

# Convert lists to numpy arrays
data = np.concatenate(data)
labels = np.concatenate(labels)

# Flatten data for SMOTE
X_flat = data.reshape(data.shape[0], -1)
y_flat = np.argmax(labels, axis=1)


# In[3]:


# Check class distribution
class_distribution = np.sum(labels, axis=0)
class_imbalance_ratio = np.max(class_distribution) / np.min(class_distribution)

use_smote = class_imbalance_ratio > 1.5  # Arbitrary threshold; adjust as needed

if use_smote:
    print("Applying SMOTE to balance classes...")
    smote = SMOTE()
    X_resampled, y_resampled = smote.fit_resample(X_flat, y_flat)
    data_resampled = X_resampled.reshape(-1, 224, 224, 3)
    labels_resampled = to_categorical(y_resampled, num_classes=len(class_distribution))
else:
    print("No need for SMOTE; using original data...")
    data_resampled = data
    labels_resampled = labels


# In[4]:


# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(data_resampled, labels_resampled, test_size=0.2, random_state=42)


# In[5]:


# Define the custom CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(class_distribution), activation='softmax')  # Output layer for multi-class classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=32
)


# In[6]:


# Evaluate the model
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')


# In[7]:


# Save the trained model
model.save('custom_plant_disease_classifier.h5')


# In[8]:


from sklearn.metrics import classification_report, confusion_matrix # type: ignore
import seaborn as sns # type: ignore
import matplotlib.pyplot as plt # type: ignore

# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(X_val, y_val)
print(f'Validation Loss: {val_loss}')
print(f'Validation Accuracy: {val_accuracy}')

# Predict the labels for the validation set
y_val_pred = model.predict(X_val)
y_val_pred_classes = np.argmax(y_val_pred, axis=1)
y_val_true = np.argmax(y_val, axis=1)

# Confusion matrix
conf_matrix = confusion_matrix(y_val_true, y_val_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=train_data.class_indices, yticklabels=train_data.class_indices)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Labels")
plt.ylabel("True Labels")
plt.show()

# Classification report (includes precision, recall, F1-score)
class_report = classification_report(y_val_true, y_val_pred_classes, target_names=train_data.class_indices.keys())
print("Classification Report:")
print(class_report)


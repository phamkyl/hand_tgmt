import os
import numpy as np
import cv2
import skimage.transform
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator

batch_size = 64
imageSize = 64
num_classes = 7

train_dir = '../hand_tgmt/data/asl_alphabet_train/asl_alphabet_train/'

def get_data(folder):
    X = []
    y = []
    label_dict = {'A': 0, 'B': 1, 'C': 2, 'L': 3, '0': 4, 'S': 5, 'Y': 6}

    for folderName in os.listdir(folder):
        if not folderName.startswith('.'):
            label = label_dict.get(folderName, 29)
            for image_filename in os.listdir(os.path.join(folder, folderName)):
                img_file = cv2.imread(os.path.join(folder, folderName, image_filename))
                if img_file is not None:
                    img_file = skimage.transform.resize(img_file, (imageSize, imageSize, 3))
                    X.append(img_file)
                    y.append(label)

    return np.array(X), np.array(y)

X_train, y_train = get_data(train_dir)
print("Images successfully imported...")
print("The shape of X_train is : ", X_train.shape)
print("The shape of y_train is : ", y_train.shape)
print("The shape of one image is : ", X_train[0].shape)

X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.3, random_state=42, stratify=y_train)

y_cat_train = to_categorical(y_train, num_classes)
y_cat_test = to_categorical(y_test, num_classes)

# Data Augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')

datagen.fit(X_train)

# Model
model = Sequential([
    Conv2D(32, (5, 5), input_shape=(imageSize, imageSize, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

model.summary()

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=2)

# Training
history = model.fit(datagen.flow(X_train, y_cat_train, batch_size=batch_size),
                    epochs=50, verbose=2, validation_data=(X_test, y_cat_test), callbacks=[early_stop])

# Evaluation
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_cat_test, axis=1)

print(classification_report(y_test_classes, y_pred_classes))

# Confusion Matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Plot Confusion Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_dict.keys(), yticklabels=label_dict.keys())
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

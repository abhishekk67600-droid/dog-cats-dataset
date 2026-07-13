import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Dataset Path
DATASET_PATH = "dogs-vs-cats/train"

# Image Size
IMG_SIZE = 64

images = []
labels = []

# Load Images
for filename in os.listdir(DATASET_PATH):

    path = os.path.join(DATASET_PATH, filename)

    img = cv2.imread(path)

    if img is None:
        continue

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    images.append(img.flatten())

    if filename.startswith("cat"):
        labels.append(0)
    else:
        labels.append(1)

X = np.array(images)
y = np.array(labels)

print("Total Images :", len(X))

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Images :", len(X_train))
print("Testing Images :", len(X_test))

# Train SVM
model = SVC(
    kernel="linear",
    C=1
)

print("Training SVM...")
model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)

print("\nAccuracy :", round(acc * 100, 2), "%")

print("\nClassification Report")
print(classification_report(y_test, y_pred))

print("Confusion Matrix")
print(confusion_matrix(y_test, y_pred))

# Display Sample Predictions
plt.figure(figsize=(12,8))

for i in range(9):

    plt.subplot(3,3,i+1)

    img = X_test[i].reshape(IMG_SIZE, IMG_SIZE)

    plt.imshow(img, cmap="gray")

    title = "Dog" if y_pred[i] == 1 else "Cat"

    plt.title(title)

    plt.axis("off")

plt.tight_layout()
plt.show()

# Predict Single Image
image_path = "test.jpg"

img = cv2.imread(image_path)

if img is not None:

    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sample = gray.flatten().reshape(1, -1)

    prediction = model.predict(sample)

    if prediction[0] == 0:
        print("Prediction : Cat")
    else:
        print("Prediction : Dog")
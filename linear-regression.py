import matplotlib.pyplot as plt
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd

# Load the digits dataset
digits = datasets.load_digits()

# Create a DataFrame for demonstration purposes
df = pd.DataFrame({
    'images': [digits.images[i] for i in range(len(digits.images))],
    'target': digits.target
})

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.5, stratify=digits.target, random_state=42)

# Train a Logistic Regression classifier
classifier = LogisticRegression(max_iter=10000, solver='liblinear')
classifier.fit(X_train, y_train)

# Predict the values
predicted = classifier.predict(X_test)

# Display a few images along with their predicted labels
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, X_test.reshape(-1, 8, 8)[:4], predicted[:4]):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation="nearest")
    ax.set_title(f"Prediction: {label}")

# Save the figure to a file
plt.savefig('plot.png')

# Print classification report
print("Classification report:\n", metrics.classification_report(y_test, predicted))
print("Confusion matrix:\n", metrics.confusion_matrix(y_test, predicted))


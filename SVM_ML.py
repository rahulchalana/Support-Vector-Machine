# =============================================================================
# ------------------- Support Vector Machine-----------------------------------
# =============================================================================
print("-------------------Support Vector Machine----------------\n")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

class SVM:
    def __init__(self, alpha = 0.01, lmbda = 0.1, iterations = 1000):
        self.alpha = alpha
        self.lmbda = lmbda
        self.iterations = iterations
        self.w = None
        self.b = None
        
    def fit(self,x,y):
        n_samples, n_features = x.shape
        
        self.w = np.zeros(n_features)
        self.b = 0
        
        y = np.where(y<=0, -1, 1)
        
        # Gradient descent
        for i in range(self.iterations):
            for index, xi in enumerate(x):
                if (y[index]*(np.dot(xi,self.w)-self.b) >= 1):
                    self.w -= self.alpha*(2*self.lmbda*self.w)
                else:
                    self.w -= self.alpha*(2*self.lmbda*self.w - np.dot(xi,y[index]))
                    self.b -= self.alpha*y[index]
                    
                    
    def predict(self, x):
        output = np.dot(x,self.w) - self.b
        return np.sign(output)


# Loading the dataset
print("A. Loading the iris flower datset")
iris = datasets.load_iris()

# Visualizing the datset
print("B. Visualizing the iris dataset:")
# Creating the scatter plot of sepal length vs. sepal width
plt.scatter(iris.data[:, 0], iris.data[:, 1], c=iris.target)
plt.xlabel('Sepal length (cm)')
plt.ylabel('Sepal width (cm)')
plt.show()

# Creating the scatter plot of petal length vs. petal width
plt.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target)
plt.xlabel('Petal length (cm)')
plt.ylabel('Petal width (cm)')
plt.show()

# Splitting the data into training and test set
print("\nC. Splitting the dataset into training and test sets")
x_train, x_test, y_train, y_test = train_test_split(iris.data[:,:2], iris.target, test_size = 0.2, random_state = 25)

# Initializing and training the model
print("\nD. Initializing and fitting the data into SVM model")
model = SVM(alpha = 0.05, lmbda = 0.01, iterations = 10000)
model.fit(x_train, y_train)

# Prediction
print("\nE. Predicting from the SVM model")
y_pred = model.predict(x_test)

# Performance metrics
print("\nPerformance metrics and Confusion matrix")
precision = np.sum((y_pred==1)&(y_test==1))/np.sum(y_pred==1)
recall = np.sum((y_pred==1)&(y_test==1))/np.sum(y_test==1)
f1_score = 2*precision*recall/(precision+recall)
accuracy = np.mean(y_test==y_pred)
print('\nPrecision: ', precision)
print('\nRecall: ', recall)
print('\nF1-score: ', f1_score)
print('\nAccuracy: ', accuracy)

confusion_matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
for i in range(len(y_test)):
    confusion_matrix[int(y_test[i])][int(y_pred[i])] += 1

# Plot the confusion matrix
print("\nConfusion Matrix:")
for row in confusion_matrix:
    print(row)

for i in range(len(confusion_matrix)):
    for j in range(len(confusion_matrix)):
        plt.text(j, i, str(confusion_matrix[i][j]), ha='center', va='center', color='white')

plt.imshow(confusion_matrix, cmap=plt.cm.Blues)
plt.title("Confusion Matrix for Iris Classification")
plt.colorbar()
plt.xticks([0, 1, 2], iris.target_names)
plt.yticks([0, 1, 2], iris.target_names)
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()

# =============================================================================
# B. Using the digits dataset
# =============================================================================
print("\nB. On the digit's datset\n")
# Loading the dataset
print("A. Loading the digits datset")
digits = datasets.load_digits()

# Visualising the data set
print("B. Visualizing the digits dataset:")
fig, axes = plt.subplots(nrows=4, ncols=10, figsize=(10, 4))

for i, ax in enumerate(axes.flat):
    ax.imshow(digits.images[i], cmap='binary')
    ax.set_axis_off()
    ax.set_title(f"Digit: {digits.target[i]}")

plt.tight_layout()
plt.show()

# Splitting the data into training and testing sets
print("\nC. Splitting the dataset into training and test sets")
x_dig_train, x_dig_test, y_dig_train, y_dig_test = train_test_split(digits.data, digits.target, test_size=0.2, random_state = 25)

model1 = SVM()
print("\nD. Initializing and fitting the data into SVM model")
model1.fit(x_dig_train, y_dig_train)

print("\nE. Predicting from the SVM model")
y_dig_pred = model1.predict(x_dig_test)

# Performance metrics
print("\nPerformance metrics and Confusion matrix")
precision = np.sum((y_dig_pred==1)&(y_dig_test==1))/np.sum(y_dig_pred==1)
recall = np.sum((y_dig_pred==1)&(y_dig_test==1))/np.sum(y_dig_test==1)
f1_score = 2*precision*recall/(precision+recall)
accuracy = np.mean(y_dig_test==y_dig_pred)
print('\nPrecision: ', precision)
print('\nRecall: ', recall)
print('\nF1-score: ', f1_score)
print('\nAccuracy: ', accuracy)

confusion_matrix = np.zeros((10,10))
for i in range(len(y_dig_test)):
    confusion_matrix[int(y_dig_test[i]), int(y_dig_pred[i])] += 1

# Plot the confusion matrix
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(confusion_matrix, cmap=plt.cm.Blues)

# Add labels to the plot
ax.set_xticks(range(10))
ax.set_yticks(range(10))
ax.set_xticklabels(range(10))
ax.set_yticklabels(range(10))
ax.set_xlabel('Predicted Label')
ax.set_ylabel('True Label')
ax.set_title('Confusion Matrix for Digits Dataset')

# Add the counts in each cell
for i in range(10):
    for j in range(10):
        ax.text(j, i, int(confusion_matrix[i, j]), ha='center', va='center')

plt.show()


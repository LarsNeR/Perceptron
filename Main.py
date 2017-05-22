import numpy as np
from Perceptron import Perceptron
import matplotlib.pyplot as plt
from sklearn import datasets, model_selection

# Load dataset - only two classes (you can also load your own with pandas but sklearn offers a range of different datasets)
(X, y) = datasets.load_iris(return_X_y=True)
X = X[:100, ]
y = y[:100, ]

# Divide dataset into train and test data
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, train_size=0.5)

# Instantiate a new Perceptron and call its fit-method with the train data
pe = Perceptron(eta=0.01, n_iter=20)
pe.fit(X=X_train, y=y_train)

# Predict the result with the test data and calculate the number of misclassifications
y_pred = pe.predict(X_test)
costs = np.absolute((y_pred - y_test)).sum()

# Print interesting information
print(costs)
print(costs / len(X_test))

# Plot misclassifications per iteration
plt.plot(pe.misclassifications_per_iter, marker='o')
plt.xlabel('Iteration')
plt.ylabel('Number of Misclassifications')
plt.show()

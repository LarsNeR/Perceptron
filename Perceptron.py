import numpy as np


class Perceptron(object):
    """
    Basic Perceptron

    Parameters
    ----------
    n_iter : int, default 100
        number of iterations the weights should be updated

    eta : int, default 0.01
        the learning rate.
        High learning rate: might overshoot the minimum
        Low learning rate: finding weights could take too long

    Attributes
    ----------
    weights : array, shape(n_features, )
        Coefficients for the hypothesis. The weights will be 'fitted' during the iterations

    misclassifications_per_iter : array, shape(n_iter, )
        Contains the number of misclassifications for every iteration
    """

    def __init__(self, n_iter=100, eta=0.01):
        self.n_iter = n_iter
        self.eta = eta
        self.weights = []
        self.misclassifications_per_iter = []

    def fit(self, X, y):
        """
        Fit Perceptron with learning rule

        Parameters
        ----------
        X : numpy array, shape(m_samples, n_features)
            Training data

        y : numpy array, shape (m_samples, )
            Target values
        """

        self.weights = np.zeros(1 + X.shape[1])

        for _ in range(self.n_iter):
            self.update_weights(X, y)

    def update_weights(self, X, y):
        errors = 0
        for xi, yi in zip(X, y):
            update = self.eta * (yi - self.predict(xi))

            # Weight 0 has to be calculated separately
            self.weights[0] = self.weights[0] + update
            self.weights[1:] = self.weights[1:] + update * xi

            errors += (update != 0)

        # Append number of misclassifications (not necessary, just to visualize improvements)
        self.misclassifications_per_iter.append(errors)

    def predict(self, X):
        """
        Predict the target value(s)

        Parameters
        ----------
        X : numpy array, shape(m_samples, n_features)
            Data to use for prediction

        Returns
        -------
        prediction : Prediction for every sample in X
        """
        return np.where(self.dot_product(X) >= 0, 1, -1)

    def dot_product(self, X):
        return np.dot(X, self.weights[1:]) + self.weights[0]

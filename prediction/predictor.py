"""
"""
import time
import numpy as np
from sklearn import svm

import utils.data as dutils
from prediction.evaluation import Evaluation

class Predictor():

    def __init__(self, target, features, regressor, config=None):
        if config:
            self.clf = regressor(**config)
        else:
            self.clf = regressor()
        self.target = target
        self.features = features

    def train(self, data):
        """
        Fit the model for provided data

        Args:
            data (pandas.DataFrame): train samples with depended variable
        """

        X_train = data[self.features].as_matrix()
        y_train = data[self.target].as_matrix()
        
        start_time = time.time()

        self.clf.fit(X_train, y_train)

        end_time = time.time()

        return end_time - start_time

    def predict(self, X_test):
        """
        Predict for given samples

        Args:
            X_test (pandas.DataFrame): samples to predict y
        Returns:
            (numpy.array) of predicted values
        """

        if (len(X_test.shape)) == 1:
            X_test = X_test[self.features].reshape(1, -1)
        else:
            X_test = X_test[self.features]

        y_pred = self.clf.predict(X_test)

        return y_pred

    def eval(self, predicted, df_test, df_train):
        return Evaluation(self.target, predicted, df_test, df_train)

        
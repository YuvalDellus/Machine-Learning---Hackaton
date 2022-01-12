import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_curve
import seaborn as sns
import pickle

"""
===================================================
     Introduction to Machine Learning (67577)
             IML HACKATHON, June 2020

Author(s):

===================================================
"""


class FlightPredictor:
    def __init__(self, path_to_weather=''):
        """
        Initialize an object from this class.
        @param path_to_weather: The path to a csv file containing weather data.
        """
        pkl_regression = "regression.pkl"
        with open(pkl_regression, 'rb') as file:
            self.regression_model = pickle.load(file)

        pkl_classification = "classifier.pkl"
        with open(pkl_classification, 'rb') as file:
            self.classification_model = pickle.load(file)

    def preprocess(self, data):
        pass

    def predict(self, x):
        """
        Recieves a pandas DataFrame of shape (m, 15) with m flight features, and predicts their
        delay at arrival and the main factor for the delay.
        @param x: A pandas DataFrame with shape (m, 15)
        @return: A pandas DataFrame with shape (m, 2) with your prediction
        """
        x = self.preprocess(x)
        reg_predict = self.regression_model.predict(x)

        classifier_predict = self.classification_model.predict(
            pd.concat([x, reg_predict], axis=1))

        return pd.DataFrame({"PredArrDelay": reg_predict,
                                       "PredDelayFactor": classifier_predict})


def main():
    flight_pred = FlightPredictor()

    print("hello")


if __name__ == '__main__':
    main()

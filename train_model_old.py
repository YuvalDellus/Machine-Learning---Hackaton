import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import roc_curve
import seaborn as sns
from sklearn.metrics import auc
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
from sklearn.multiclass import OneVsRestClassifier
from pre_processing import pre_process
from pre_processing import all_data_pre_processing

import pickle

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# def preprocess(data):
#     """
#
#     :param data: Raw data
#     :return: Preprocessed data
#     """
#     week_dummy = pd.get_dummies(data["DayOfWeek"])
#     airline_dummy = pd.get_dummies(data["Reporting_Airline"])
#     flight_num_dummy = pd.get_dummies(
#         data["Flight_Number_Reporting_Airline"])
#     origin_dummy = pd.get_dummies(data["Origin"])
#     dest_dummy = pd.get_dummies(data["Dest"])
#
#     col_to_drop = ['Unnamed: 0', 'FlightDate',
#                     'Origin', 'Reporting_Airline', 'Tail_Number',
#                    'OriginCityName', 'OriginState', 'Dest', 'DestCityName',
#                    'DestState', 'CRSDepTime', 'CRSArrTime']
#
#     processed_data = pd.concat([data.drop(columns=col_to_drop), week_dummy,
#                                 origin_dummy, airline_dummy, dest_dummy],
#                                axis=1)
#
#     # processed_data.fillna("", inplace=True)
#     return processed_data


def train_regression(x_train, y_train):
    """

    :param data: Preprocessed data
    :return: trained linear regression model
    """
    lin_reg = LinearRegression()
    lin_reg.fit(x_train, y_train)
    print("finished linear regressino fit")

    pkl_filename = "regression.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(lin_reg, file)
    #
    # with open(pkl_filename, 'rb') as file:
    #     lin_reg = pickle.load(file)

    print("Linear Reg Training Error: ", lin_reg.score(x_train, y_train))

    return lin_reg



def train_classification(x_train, y_train_factor, x_test, y_test_factor):
    """
    This function is where we tried different possible classifiers until we
    found the best one
    :param data: Preprocessed data that includes the ArrDelay and
    DelayFactor columns
    :return: a trained classifier that predicts the delay factor
    """

    # Take the indices of the rows with flights that arrived on time
    on_time = np.where(x_train["ArrDelay"] <= 0)
    x_train.drop(columns="DelayFactor", inplace=True)


    classifier_list = [DecisionTreeClassifier(), SVC(kernel='linear')]

    all_predictions = pd.DataFrame({"True_y": y_test_factor})
    all_scores = []
    for c in classifier_list:
        print("in classifiers loop")
        classifier = OneVsRestClassifier(c).fit(x_train, y_train_factor)
        score = classifier.score(x_test.drop(columns=["DelayFactor"]), y_test_factor)
        all_scores.append(score)
        y_predict = classifier.predict(x_test.drop(columns=["DelayFactor"]))
        y_predict[on_time] = np.nan
        all_predictions[c.__class__.__name__] = y_predict

    print(all_predictions)
    print(all_scores)


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("{} score: ".format(model), model.score(x_test, y_test))
    print("{} mean square error: ".format(model), mean_squared_error(y_test,
                                                                     y_pred))


def split_data():
    train_data = pd.read_csv("train_data.csv")
    all_train, test = train_test_split(train_data, test_size=0.25)
    test.to_csv("test_split.csv", index=False)

    train, validation = train_test_split(all_train, test_size=0.25)
    validation.to_csv("train_validation.csv")
    train.to_csv("train_split.csv", index=False)


def main():
    all_model_data = pd.read_csv("train_data.csv") # this is all the data
    # used just to get the correct  columns
    required_cols= all_data_pre_processing(all_model_data)

    train_split = pd.read_csv("train_split.csv")
    x_train = pre_process(train_split,required_cols)
    y_train_delay = x_train["ArrDelay"]
    y_train_factor = x_train["DelayFactor"]

    test_split = pd.read_csv("train_validation.csv")
    x_test = pre_process(test_split,required_cols)

    y_test_factor = x_test["DelayFactor"]

    lin_reg = train_regression(x_train.drop(columns=["ArrDelay", "DelayFactor"]),
                                       y_train_delay)

    classifier = train_classification(x_train, y_train_factor, x_test,
                                     y_test_factor)


if __name__ == '__main__':
    #split_data()

    main()

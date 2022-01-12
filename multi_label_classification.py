import pandas as pd
import numpy as np
from sklearn.datasets import make_multilabel_classification
import time
import merge_classification
import pre_processing as pp

# start = time.time()
# end = time.time()
# print(end - start)


def generate_predictions_vector(predictions_binary):
    y_predict = []
    for i in range(len(predictions_binary.toarray())):
        if predictions_binary.toarray()[i][0] == 1:
            y_predict.append("CarrierDelay")
        else:
            if predictions_binary.toarray()[i][1] == 1:
                y_predict.append("LateAircraftDelay")
            else:
                if predictions_binary.toarray()[i][2] == 1:
                    y_predict.append("NASDelay")
                else:
                    if predictions_binary.toarray()[i][3] == 1:
                        y_predict.append("WeatherDelay")
                    else:
                        y_predict.append(np.nan)
    return pd.DataFrame({'DelayReason': y_predict})


# data_path = "Flight_data/train_split.csv"
# X_train = pd.read_csv(data_path)
#
# test_path = "Flight_data/test_split.csv"
# X_test = pd.read_csv(test_path)


# X_train, y_train = make_multilabel_classification(n_samples=20000, n_classes=5,
#                                                   allow_unlabeled=True,
#                                                   random_state=1)
#
# X_test, y_test = make_multilabel_classification(n_samples=20000, n_classes=5,
#                                                 allow_unlabeled=True,
#                                                 random_state=1)


# y_train = pd.get_dummies(X_train["DelayFactor"])
# X_train.drop(columns=['DelayFactor', 'Unnamed: 0', 'FlightDate',
#                        'Reporting_Airline', 'Tail_Number', 'Origin',
#                        'OriginCityName', 'OriginState', 'Dest', 'DestCityName',
#                        'DestState'], inplace=True)
#
#
# y_test = pd.get_dummies(X_test["DelayFactor"])
# X_test.drop(columns=['DelayFactor', 'Unnamed: 0', 'FlightDate',
#                        'Reporting_Airline', 'Tail_Number', 'Origin',
#                        'OriginCityName', 'OriginState', 'Dest', 'DestCityName',
#                        'DestState'], inplace=True)

all_model_data = pd.read_csv("train_data.csv") # this is all the data
# used just to get the correct  columns
required_cols= pp.all_data_pre_processing(all_model_data)

train_split = pd.read_csv("train_data.csv")
X_train = pp.pre_process(train_split,required_cols)
y_train = X_train["DelayFactor"]

test_split = pd.read_csv("train_validation.csv")
X_test = pp.pre_process(test_split,required_cols)

y_test = X_test["DelayFactor"]



from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

print("-----------------------------")
start = time.time()

# using binary relevance
from skmultilearn.problem_transform import BinaryRelevance


# initialize binary relevance multi-label classifier
# with a gaussian naive bayes base classifier
classifier_binary = BinaryRelevance(GaussianNB())

# train
classifier_binary.fit(X_train, y_train)

# predict
predictions_binary = classifier_binary.predict(X_test)
y_predict_binary = generate_predictions_vector(predictions_binary)

print("binary relevance: ")
print("accuracy: ", accuracy_score(y_test,predictions_binary))

end = time.time()
print("time: ", end - start)
print("-----------------------------")

start = time.time()

# using Label Powerset
from skmultilearn.problem_transform import LabelPowerset

# initialize Label Powerset multi-label classifier
# with a gaussian naive bayes base classifier
classifier_powerset = LabelPowerset(GaussianNB())

# train
classifier_powerset.fit(X_train, y_train)

# predict
predictions_powerset= classifier_powerset.predict(X_test)
y_predict_powerset = generate_predictions_vector(predictions_binary)

print("Label Powerset: ")
print("accuracy: ", accuracy_score(y_test,predictions_powerset))

end = time.time()
print("time: ", end - start)
print("-----------------------------")

start = time.time()

# using classifier chains
from skmultilearn.problem_transform import ClassifierChain

# initialize classifier chains multi-label classifier
# with a gaussian naive bayes base classifier
classifier_chains = ClassifierChain(GaussianNB())

# train
classifier_chains.fit(X_train, y_train)

# predict
predictions_chains = classifier_chains.predict(X_test)
y_predict_chains = generate_predictions_vector(predictions_binary)

print("classifier chains: ")
print("accuracy: ", accuracy_score(y_test,predictions_chains))

end = time.time()
print("time: ", end - start)
print("-----------------------------")



# from skmultilearn.adapt import MLkNN
# from scipy.sparse import csr_matrix, lil_matrix
#
# start = time.time()
#
# # using Adapted Algorithm
# classifier_new = MLkNN(k=10)
#
# # Note that this classifier can throw up errors when handling sparse matrices.
# x_train = lil_matrix(X_train).toarray()
# y_train = lil_matrix(y_train).toarray()
# x_test = lil_matrix(X_test).toarray()
#
# # train
# classifier_new.fit(x_train, y_train)
#
# # predict
# predictions_adapted = classifier_new.predict(x_test)
# y_predict_adapted = generate_predictions_vector(predictions_adapted)
#
# # accuracy
# print("Adapted Algorithm: ")
# print("accuracy: ", accuracy_score(y_test,predictions_adapted))
#
# end = time.time()
# print("time: ", end - start)
# print("-----------------------------")

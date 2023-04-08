from DataParser import DataParser
from DataPreparation import DataTokenization
from DetectionEngine import DetectionEngine

import numpy as np


# Data organiser is used to organise data into CSV files and create training and test datasets.
# Original dataset used is CSIC 2010 dataset. This dataset comes as txt file with unlabeled data.
# Since this system implements Multilayer Perceptron, data needs to be labeled to enable supervised learning.
def data_organiser():
    # Read in and label the data
    normal_traffic = DataParser('datasets/normalTrafficTraining.txt', 'normal')
    anomalous_traffic = DataParser('datasets/anomalousTrafficTest.txt', 'anomalous')
    # Convert the data into CSV
    normal_traffic.txt_cvs_parse()
    anomalous_traffic.txt_cvs_parse()
    # Split data in half
    normal_traffic.splitter()
    anomalous_traffic.splitter()
    # Create and shuffle training data
    DataParser.combine('normal_1st_half.csv', 'anomalous_2nd_half.csv', 'training_set.csv')
    DataParser.shuffle('training_set.csv', 'training_data_shuffled.csv')
    # Create and shuffle the test set
    DataParser.combine('normal_2nd_half.csv', 'anomalous_1st_half.csv', 'test_set.csv')
    DataParser.shuffle('test_set.csv', 'test_set_shuffled.csv')


def accuracy(predictions, targets):
    correct = 0
    false_positive = 0
    false_negative = 0
    for pred, targ in zip(predictions, targets):
        if pred == targ:
            correct += 1
        elif pred == 0 and targ == 1:
            false_negative += 1
        elif pred == 1 and targ == 0:
            false_positive += 1

    accuracy = correct / len(predictions) * 100
    false_n = false_negative / len(predictions) * 100
    false_p = false_positive / len(predictions) * 100

    print(f"Accuracy: {accuracy:.2f}% | False negatives: {false_n:.2f}% | False positives: {false_p:.2f}%")


def main():
    first_fold = np.array(DataTokenization('training_data_shuffled.csv').tokenize_data())
    first_fold_targets = np.array(DataTokenization('training_data_shuffled.csv').tokenize_labels())
    second_fold = np.array(DataTokenization('test_set_shuffled.csv').tokenize_data())
    second_fold_targets = np.array(DataTokenization('test_set_shuffled.csv').tokenize_labels())
    # print(training_data)
    # print(targets)
    # neural = NeuralNetwork(11, 16, 16, 1, 0.01, 3)
    # neural.training(training_data, targets)
    first_fold_targets = first_fold_targets.reshape(-1, 1)
    second_fold_targets = second_fold_targets.reshape(-1, 1)
    mlp = DetectionEngine(11, 64, 32, 1, 0.001)
    mlp2 = DetectionEngine(11, 64, 32, 1, 0.001)
    mlp.train(first_fold, first_fold_targets, 420)
    predictions = mlp.predict(second_fold)
    accuracy(predictions, second_fold_targets)
    # mlp2.train(second_fold, second_fold_targets, 420)
    # predictions = mlp2.predict(first_fold)
    # accuracy(predictions, first_fold_targets)


if __name__ == "__main__":
    # data_organiser()
    main()

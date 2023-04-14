from DataParser import DataParser
from DataPreparation import DataTokenization
from DetectionEngine import DetectionEngine


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


def main():
    # Tokenize data and labels
    first_fold = DataTokenization('training_data_shuffled.csv').tokenize_data()
    first_fold_targets = DataTokenization('training_data_shuffled.csv').tokenize_labels()
    second_fold = DataTokenization('test_set_shuffled.csv').tokenize_data()
    second_fold_targets = DataTokenization('test_set_shuffled.csv').tokenize_labels()
    # Initialise Neural Network
    mlp = DetectionEngine(11, 64, 32, 1, 0.001)
    mlp2 = DetectionEngine(11, 64, 32, 1, 0.001)
    # Train and test the model
    mlp.train(first_fold, first_fold_targets, 512, 128)
    mlp.predict(second_fold, second_fold_targets)
    # mlp2.train(second_fold, second_fold_targets, 420, 128)
    # predictions = mlp2.predict(first_fold)
    # accuracy(predictions, first_fold_targets)


if __name__ == "__main__":
    # data_organiser()
    main()

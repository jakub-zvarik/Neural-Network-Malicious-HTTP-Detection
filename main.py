from DataParser import DataParser
from DataTokenizer import DataTokenizer
from DetectionEngine import DetectionEngine

# Main file of the Feedforward Neural Network program trained to detect malicious HTTP traffic
# Architecture:
# DataParser - parse data from .txt to CSV
# DataTokenizer - tokenizing data for the neural network
# Maths - mathematical operations for the neural network
# DetectionEngine - neural network


# Method is not used, since all the datasets are parsed and ready in the datasets folder
# Data organiser is used to organise data into CSV files and create training and test datasets
# Original dataset used is CSIC 2010 dataset. This dataset comes as txt file with unlabeled data
# Since this system implements Multilayer Perceptron, data needs to be labeled to enable supervised learning
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
    DataParser.combine('datasets/normal_1st_half.csv', 'datasets/anomalous_2nd_half.csv', 'datasets/training_set.csv')
    DataParser.shuffle('datasets/training_set.csv', 'datasets/training_data_shuffled.csv')
    # Create and shuffle the test set
    DataParser.combine('datasets/normal_2nd_half.csv', 'datasets/anomalous_1st_half.csv', 'datasets/test_set.csv')
    DataParser.shuffle('datasets/test_set.csv', 'datasets/test_set_shuffled.csv')


# Main method
# Tokenize the HTTP requests (forming input for the neural network)
# Tokenize the targets (targets are labels for supervised learning)
# Create two instances of neural network, train and test them
def main():
    # Parameters for the neural network
    # Numbers of neurons in hidden layers
    NEURONS_HIDDEN1 = 64
    NEURONS_HIDDEN2 = 32
    # Others
    LEARNING_RATE = 0.001
    EPOCHS = 512
    BATCH_SIZE = 128
    # Tokenize data and labels
    first_fold = DataTokenizer('datasets/training_data_shuffled.csv').tokenize_data()
    first_fold_targets = DataTokenizer('datasets/training_data_shuffled.csv').tokenize_labels()
    second_fold = DataTokenizer('datasets/test_set_shuffled.csv').tokenize_data()
    second_fold_targets = DataTokenizer('datasets/test_set_shuffled.csv').tokenize_labels()
    # Initialise Neural Network
    detection1 = DetectionEngine(NEURONS_HIDDEN1, NEURONS_HIDDEN2, LEARNING_RATE)
    detection2 = DetectionEngine(NEURONS_HIDDEN1, NEURONS_HIDDEN2, LEARNING_RATE)
    # Train and test the model
    # Train on first fold, test on second
    print("FIRST FOLD TRAINING:")
    detection1.train(first_fold, first_fold_targets, EPOCHS, BATCH_SIZE)
    print("FIRST FOLD TEST:")
    detection1.predict(second_fold, second_fold_targets)
    # Second fold
    print("SECOND FOLD TRAINING:")
    detection2.train(second_fold, second_fold_targets, EPOCHS, BATCH_SIZE)
    print("SECOND FOLD TEST:")
    detection2.predict(first_fold, first_fold_targets)


if __name__ == "__main__":
    main()

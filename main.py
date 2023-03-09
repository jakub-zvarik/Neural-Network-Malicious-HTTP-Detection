from DataParser import DataParser
from DataPreparation import DataTokenization


def data_organiser():
    normal_traffic = DataParser('datasets/normalTrafficTraining.txt', 'normal')
    anomalous_traffic = DataParser('datasets/anomalousTrafficTest.txt', 'anomalous')
    normal_traffic.txt_cvs_parse()
    anomalous_traffic.txt_cvs_parse()
    normal_traffic.splitter()
    anomalous_traffic.splitter()
    DataParser.combine('normal_1st_half.csv', 'anomalous_2nd_half.csv', 'training_set.csv')
    DataParser.shuffle('training_set.csv', 'training_data_shuffled.csv')


def main():
    training_data = DataTokenization('training_data_shuffled.csv').tokenize_data()
    targets = DataTokenization('training_data_shuffled.csv').tokenize_labels()
    print(training_data)


if __name__ == "__main__":
    # data_organiser()
    main()

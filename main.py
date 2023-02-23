from DataParser import DataParser
from DataPreparation import DataPreparation


def main():
    # normal_traffic = DataParser('datasets/normalTrafficTraining.txt', 'normal')
    # anomalous_traffic = DataParser('datasets/anomalousTrafficTest.txt', 'anomalous')
    # normal_traffic.txt_cvs_parse()
    # anomalous_traffic.txt_cvs_parse()
    # normal_traffic.splitter()
    # anomalous_traffic.splitter()
    # DataParser.combine('normal_1st_half.csv', 'anomalous_2nd_half.csv', 'training_set.csv')
    # DataParser.shuffle('training_set.csv', 'shuffled2.csv')
    print(DataPreparation.num_numbers('0,0,1,2,3,4,5'))


if __name__ == "__main__":
    main()

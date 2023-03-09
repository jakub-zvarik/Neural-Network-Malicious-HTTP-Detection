import csv
import re


class DataTokenization:

    def __init__(self, dataset):
        self.data = dataset

    def tokenize_data(self):
        tokenized_data = []
        with open(self.data, 'r') as datafile:
            reader = csv.reader(datafile)
            for row in reader:
                new_array = []
                # Method
                new_array.append(self.method_token(row[0]))
                # URL
                new_array.append(self.length(row[1]))
                new_array.append((self.num_upper_cases(row[1])))
                new_array.append(self.num_lower_cases(row[1]))
                new_array.append(self.num_numbers(row[1]))
                new_array.append(self.num_special_chars(new_array[1], new_array[2], new_array[3], new_array[4]))
                # Body of request
                new_array.append(self.length(row[2]))
                new_array.append((self.num_upper_cases(row[2])))
                new_array.append(self.num_lower_cases(row[2]))
                new_array.append(self.num_numbers(row[2]))
                new_array.append(self.num_special_chars(new_array[6], new_array[7], new_array[8], new_array[9]))
                # Append to the resulting array
                tokenized_data.append(new_array)

            return tokenized_data

    def tokenize_labels(self):
        tokenized_labels = []
        with open(self.data, 'r') as datafile:
            reader = csv.reader(datafile)
            for row in reader:
                label = self.label(row[3])
                tokenized_labels.append(label)
            return tokenized_labels

    @staticmethod
    def method_token(method):
        token = 0
        if 'GET' in method:
            token = 1
        elif 'POST' in method:
            token = 2
        elif 'PUT' in method:
            token = 3
        return token

    @staticmethod
    def length(string):
        length = len(string)
        return length

    @staticmethod
    def num_upper_cases(string):
        upper_cases = len(re.findall(r'[A-Z]', string))
        return upper_cases

    @staticmethod
    def num_lower_cases(string):
        lower_cases = len(re.findall(r'[a-z]', string))
        return lower_cases

    @staticmethod
    def num_numbers(string):
        numbers = len(re.findall(r'[0-9]', string))
        return numbers

    @staticmethod
    def num_special_chars(full_length, uppers, lowers, numbers):
        special_chars = full_length - (uppers + lowers + numbers)
        return special_chars

    @staticmethod
    def label(string):
        label = 2
        if string == 'normal':
            label = 0
        elif string == 'anomalous':
            label = 1
        return label

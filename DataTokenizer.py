import csv
import re
import numpy as np


# Data Tokenizer
# Transform parts of HTTP requests into numerical values to form input for the neural network
class DataTokenizer:

    # Constructor
    def __init__(self, dataset):
        self.data = dataset

    # Tokenize parts of the HTTP requests - Method, URI, Body of request
    # Add to a 2D array, return it transformed into NumPy array
    def tokenize_data(self):
        tokenized_data = []
        # Open CSV with the dataset
        with open(self.data, 'r') as datafile:
            reader = csv.reader(datafile)
            for row in reader:
                new_array = []
                # Method
                new_array.append(self.method_token(row[0]))
                # URI
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
            # Return as Numpy array
            return np.array(tokenized_data)

    # Tokenize labels for supervised learning
    # Return as NumPy array
    def tokenize_labels(self):
        tokenized_labels = []
        with open(self.data, 'r') as datafile:
            reader = csv.reader(datafile)
            for row in reader:
                label = [self.label(row[3])]
                tokenized_labels.append(label)
            # Return as Numpy array
            return np.array(tokenized_labels)

    # Tokenize method
    # Only GET, POST, PUT are in the dataset
    def method_token(self, method):
        token = 0
        if 'GET' in method:
            token = 1
        elif 'POST' in method:
            token = 2
        elif 'PUT' in method:
            token = 3
        return float(token)

    # Return length of a string
    def length(self, string):
        length = len(string)
        return float(length)

    # Return number of upper cases in a string
    def num_upper_cases(self, string):
        upper_cases = len(re.findall(r'[A-Z]', string))
        return float(upper_cases)

    # Return number of lower cases in a string
    def num_lower_cases(self, string):
        lower_cases = len(re.findall(r'[a-z]', string))
        return float(lower_cases)

    # Return number of numbers in a string
    def num_numbers(self, string):
        numbers = len(re.findall(r'[0-9]', string))
        return float(numbers)

    # Return number of special characters in a string
    def num_special_chars(self, full_length, uppers, lowers, numbers):
        special_chars = full_length - (uppers + lowers + numbers)
        return float(special_chars)

    # Transform labels in the dataset to a numerical value
    def label(self, string):
        label = 2
        if string == 'normal':
            label = 0.0
        elif string == 'anomalous':
            label = 1.0
        return label

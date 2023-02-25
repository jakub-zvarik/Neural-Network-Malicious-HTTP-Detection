import csv
import re
import random


# Data Parser
# Parameters: file name and label
# Use: Preprocess datasets from txt to csv file, fold data, combine and shuffle training set
class DataParser:

    # Constructor
    def __init__(self, file_name, label):
        self.file_name = file_name
        self.label = label

    # Convert dataset from txt to CSV file
    def txt_cvs_parse(self):
        with open(self.file_name, 'r') as input_file, open(self.label + '.csv', 'w', newline='') as output_file:
            writer = csv.writer(output_file)
            while True:
                line = input_file.readline()
                if not line:
                    break
                match = re.match(r'(.*) (http://.*) (HTTP/\d\.\d)', line)
                if match:
                    method = match.group(1)
                    url = match.group(2)
                    headers = ''
                    body = ''
                    while True:
                        line = input_file.readline()
                        if line == '\n':
                            break
                        headers += line
                    if 'Content-Length' in headers:
                        content_length = int(re.search(r'Content-Length: (\d+)', headers).group(1))
                        body = input_file.read(content_length)

                    writer.writerow([method, url, body, self.label])

    # Split CSV file into 2 CSVs
    def splitter(self):
        with open(self.label + '.csv', 'r') as splitfile:
            reader = csv.reader(splitfile)
            data = list(reader)

        length = len(data)
        half = length // 2
        if length % 2 == 0:
            first_half = data[:half]
            second_half = data[half:]
        else:
            first_half = data[:half + 1]
            second_half = data[half + 1:]

        with open(self.label + '_1st_half.csv', 'w', newline='') as new_csv:
            writer = csv.writer(new_csv)
            writer.writerows(first_half)
        with open(self.label + '_2nd_half.csv', 'w', newline='') as new_csv:
            writer = csv.writer(new_csv)
            writer.writerows(second_half)

    # Merge two CSV files into one
    @staticmethod
    def combine(first_file, second_file, output_file):
        with open(first_file, 'r') as file1:
            reader = csv.reader(file1)
            first_part = list(reader)
        with open(second_file, 'r') as file2:
            reader = csv.reader(file2)
            second_part = list(reader)

        combined = first_part + second_part
        with open(output_file, 'w', newline='') as combined_output:
            writer = csv.writer(combined_output)
            writer.writerows(combined)

    # Shuffle rows in CSV file randomly
    @staticmethod
    def shuffle(filename, output_file):
        with open(filename, 'r') as file:
            reader = csv.reader(file)
            rows = list(reader)

        random.shuffle(rows)
        with open(output_file, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(rows)

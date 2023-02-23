import csv
import re


class DataPreparation:

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
    def num_special_chars(string, uppers, lowers, numbers):
        full_length = len(string)
        normal_chars = uppers + lowers + numbers
        special_chars = full_length - normal_chars
        return special_chars

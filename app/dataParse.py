import os
import csv


def getCsvFiles(file_path):
    files = os.listdir(file_path)

    csv_files = [f for f in files if f.endswith('.csv')]

    return len(csv_files)


def load(file_path, file_index):
    files = os.listdir(file_path)

    csv_files = [f for f in files if f.endswith('.csv')]

    csv_files.sort()

    if file_index < 0 or file_index >= len(csv_files):
        return None

    selected_file = csv_files[file_index]

    file_full_path = os.path.join(file_path, selected_file)

    with open(file_full_path, 'r') as file:
        csv_reader = csv.reader(file)
        csv_data = [row for row in csv_reader]

    return csv_data


def parse(dataset, data_index, stream_length):
    lonk = len(dataset) / 6
    if data_index == lonk - (1+stream_length):
        return None
    return dataset[data_index: data_index+stream_length]

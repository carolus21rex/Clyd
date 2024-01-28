import os
import csv


def clean_csv_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, 'r', newline='') as file:
                reader = csv.reader(file)
                rows_to_keep = []
                for row_num, row in enumerate(reader, start=1):
                    # Check if all values in the row can be cast to float
                    try:
                        # If any value cannot be cast to float, skip the row
                        float_values = [float(value) for value in row]
                        rows_to_keep.append(row)
                    except ValueError:
                        print(f"Non-float values found in {filename}: Row {row_num}")

            # Write the filtered rows back to the CSV file
            with open(file_path, 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(rows_to_keep)


# Example usage: Clean all CSV files in the current directory
clean_csv_files('.')

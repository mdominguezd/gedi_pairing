import pandas as pd
from io import StringIO

def read_waveform(file_path):

    # Read file and separate header from data
    header_lines = []
    data_lines = []

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("#"):
                header_lines.append(line.strip())
            elif line.strip():  # Skip empty lines
                data_lines.append(line.strip())


    data_str = "\n".join(data_lines)
    df = pd.read_csv(StringIO(data_str), sep='\s+', header=None)

    # Assign optional column names
    if len(df.columns) == 10:
        df.columns = ['elevation', 'discrete_intensity', 'int_canopy' , 'int_ground', 'discrete_count', 'count_canopy', 'count_ground', 'discrete_fraction', 'fraction_canopy', 'fraction_ground']
    else:
        df.columns = ['elevation', 'discrete_intensity', 'discrete_count', 'discrete_fraction']

    return df


def read_metrics(file_path):
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Step 1: Extract header line (remove leading "#", split by comma or space)
    header_line = lines[0].strip()
    header_items = header_line[1:].split(",")  # Remove "#", split on commas

    # Clean and trim each column name
    columns = [item.strip() for item in header_items]

    # Step 2: Read the actual data line and split it
    data_line = lines[1].strip()
    data_values = data_line.split()

    # Step 3: Handle case where first column is not a number
    if not data_values[0].replace(".", "", 1).isdigit():
        data_values[0] = data_values[0].strip()  # e.g. 'BEAM.x.'
        
    # Step 4: Sanity check
    if len(data_values) != len(columns):
        columns = columns[:-1]

    # Step 5: Create DataFrame
    df = pd.DataFrame([data_values], columns=columns)

    return df
import pandas as pd
from datetime import datetime, time, timedelta

def calculate_distance_matrix(file_path):
    # Read the CSV file into a pandas DataFrame
    data = pd.read_csv(file_path)

    # Initialize an empty dictionary to store distances
    distances = {}

    # Iterate through the dataset and populate the distances dictionary
    for index, row in data.iterrows():
        id_start = row['id_start']
        id_end = row['id_end']
        distance = row['distance']

        # Populate distances dictionary with bidirectional distances
        if id_start not in distances:
            distances[id_start] = {}
        if id_end not in distances:
            distances[id_end] = {}

        distances[id_start][id_end] = distance
        distances[id_end][id_start] = distance

    # Create a list of unique IDs
    ids = sorted(list(distances.keys()))

    # Create an empty DataFrame to store the distance matrix
    distance_matrix = pd.DataFrame(0, index=ids, columns=ids)

    # Populate the distance matrix with the collected distances
    for i in ids:
        for j in ids:
            if i != j:
                distance_matrix.loc[i, j] = distances.get(i, {}).get(j, 0)

    # Set diagonal values to 0
    distance_matrix.values[[range(len(ids))] * 2] = 0

    return distance_matrix

# Assuming dataset-3.csv is in the same directory as the script
result_df = calculate_distance_matrix('D:\\work\\MapUp-Data-Assessment-F\\datasets\\dataset-3.csv')
print(result_df)

def unroll_distance_matrix(distance_matrix):
    # Get the values, row indices, and column indices from the distance matrix
    values = distance_matrix.values
    row_indices, col_indices = distance_matrix.index.values, distance_matrix.columns.values

    # Create an empty list to store combinations of IDs and distances
    unrolled_data = []

    # Generate combinations of id_start and id_end along with their distances
    for i, row_id in enumerate(row_indices):
        for j, col_id in enumerate(col_indices):
            if i != j:
                id_start, id_end = row_id, col_id
                distance = values[i, j]
                unrolled_data.append([id_start, id_end, distance])

    # Create a DataFrame from the unrolled data
    unrolled_df = pd.DataFrame(unrolled_data, columns=['id_start', 'id_end', 'distance'])

    return unrolled_df

# Assuming 'result_df' is the DataFrame from Question 1
unrolled_result_df = unroll_distance_matrix(result_df)
print("question:2")
print(unrolled_result_df)

def find_ids_within_ten_percentage_threshold(distance_df, reference_value):
    # Calculate average distance for the reference value
    avg_distance = distance_df[distance_df['id_start'] == reference_value]['distance'].mean()

    # Calculate the threshold range (10%)
    lower_threshold = avg_distance - (avg_distance * 0.1)
    upper_threshold = avg_distance + (avg_distance * 0.1)

    # Filter IDs within the threshold
    filtered_ids = distance_df[(distance_df['distance'] >= lower_threshold) & (distance_df['distance'] <= upper_threshold)]
    sorted_ids = filtered_ids['id_start'].unique()
    return sorted(sorted_ids)

# Assuming 'unrolled_result_df' is the DataFrame from Question 2
# Assuming 'reference_value' is the integer reference value from the 'id_start' column
reference_value = 123  # Replace with your chosen reference value
ids_within_threshold = find_ids_within_ten_percentage_threshold(unrolled_result_df, reference_value)
print("question:3")
print(ids_within_threshold)

def calculate_toll_rate(dataframe):
    # Define rate coefficients for each vehicle type
    rate_coefficients = {
        'moto': 0.8,
        'car': 1.2,
        'rv': 1.5,
        'bus': 2.2,
        'truck': 3.6
    }

    # Calculate toll rates for each vehicle type
    for vehicle, rate in rate_coefficients.items():
        dataframe[vehicle] = dataframe['distance'] * rate

    return dataframe


# Assuming 'unrolled_result_df' is the DataFrame from Question 2
result_with_toll_rates = calculate_toll_rate(
    unrolled_result_df.copy())  # Create a copy to avoid modifying the original DataFrame
print("question:4 ")
print(result_with_toll_rates)


def calculate_time_based_toll_rates(dataframe):
    # Define discount factors for different time intervals and days
    weekday_discounts = {
        (time(0, 0, 0), time(10, 0, 0)): 0.8,
        (time(10, 0, 0), time(18, 0, 0)): 1.2,
        (time(18, 0, 0), time(23, 59, 59)): 0.8
    }
    weekend_discount = 0.7

    # Create an empty list to store the time-based toll rates
    time_based_toll_rates = []

    # Iterate through each unique (id_start, id_end) pair
    unique_pairs = dataframe[['id_start', 'id_end']].drop_duplicates()
    for index, row in unique_pairs.iterrows():
        id_start, id_end = row['id_start'], row['id_end']

        # Create datetime objects to cover a full 24-hour period for 7 days
        start_datetime = datetime.combine(datetime.now().date(), time(0, 0, 0))
        end_datetime = start_datetime + timedelta(days=7) - timedelta(seconds=1)

        while start_datetime <= end_datetime:
            start_day = start_datetime.strftime("%A")  # Get the day name (Monday to Sunday)
            end_day = (start_datetime + timedelta(days=1) - timedelta(seconds=1)).strftime("%A")

            start_time = start_datetime.time()
            end_time = (start_datetime + timedelta(hours=1) - timedelta(seconds=1)).time()

            # Check if it's a weekday or weekend and apply the corresponding discount factor
            if start_day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
                for interval, discount in weekday_discounts.items():
                    if interval[0] <= start_time <= interval[1]:
                        toll_rate = dataframe[(dataframe['id_start'] == id_start) &
                                              (dataframe['id_end'] == id_end)]['distance'].values[0] * discount
                        time_based_toll_rates.append(
                            [id_start, id_end, start_day, start_time, end_day, end_time, toll_rate])
            else:  # Weekend
                toll_rate = dataframe[(dataframe['id_start'] == id_start) &
                                      (dataframe['id_end'] == id_end)]['distance'].values[0] * weekend_discount
                time_based_toll_rates.append([id_start, id_end, start_day, start_time, end_day, end_time, toll_rate])

            start_datetime += timedelta(hours=1)

    # Create a DataFrame from the time-based toll rates
    time_based_toll_df = pd.DataFrame(time_based_toll_rates,
                                      columns=['id_start', 'id_end', 'start_day', 'start_time', 'end_day', 'end_time',
                                               'toll_rate'])

    return time_based_toll_df

final_result_df = calculate_time_based_toll_rates('D:\\work\\MapUp-Data-Assessment-F\\datasets\\dataset-3.csv')
print('question:5 ')
print(final_result_df)
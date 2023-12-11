import pandas as pd

# Load dataset-1.csv and dataset-2.csv
df1 = pd.read_csv("D:\\work\\MapUp-Data-Assessment-F\\datasets\\dataset-1.csv")
df2 = pd.read_csv("D:\\work\\MapUp-Data-Assessment-F\\datasets\\dataset-2.csv")

def generate_car_matrix(df):
    """
    Creates a DataFrame for id combinations.

    Args:
        df (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Matrix generated with 'car' values,
                          where 'id_1' and 'id_2' are used as indices and columns respectively.
    """
    # Logic for generating the car matrix
    # Pivot the DataFrame based on id_1 and id_2 columns with 'car' values
    pivoted = df.pivot(index='id_1', columns='id_2', values='car')

    # Set diagonal values to 0
    for i in range(min(len(pivoted), len(pivoted.columns))):
        pivoted.iloc[i, i] = 0

    return pivoted

# Example usage:
result_question_1 = generate_car_matrix(df1)
print("Question 1 Result:")
print(result_question_1)

def get_type_count(df):
    """
    Categorizes 'car' values into types and returns a dictionary of counts.

    Args:
        df (pandas.DataFrame)

    Returns:
        dict: A dictionary with car types as keys and their counts as values.
    """
    # Logic for counting car types
    # Add a new column 'car_type' based on 'car' column values
    conditions = [
        (df['car'] <= 15),
        (df['car'] > 15) & (df['car'] <= 25),
        (df['car'] > 25)
    ]
    choices = ['low', 'medium', 'high']
    df['car_type'] = pd.Series(pd.cut(df['car'], bins=[-float('inf'), 15, 25, float('inf')], labels=choices))

    # Calculate the count of occurrences for each car_type category
    type_counts = df['car_type'].value_counts().sort_index()

    # Convert the counts to a dictionary and return the result
    type_counts_dict = type_counts.to_dict()
    return type_counts_dict


# Example usage:
result_question_2 = get_type_count(df1)
print("Question 2 Result:")
print(result_question_2)
print()

def get_bus_indexes(df):
    """
    Returns the indexes where the 'bus' values are greater than twice the mean.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of indexes where 'bus' values exceed twice the mean.
    """
    # Logic for retrieving bus indexes
    mean_bus = df['bus'].mean()
    bus_indices = df[df['bus'] > 2 * mean_bus].index.sort_values().tolist()
    return bus_indices

result_question_3 = get_bus_indexes(df1)
print("Question 3 Result:")
print(result_question_3)
print()

def filter_routes(df):
    """
    Filters and returns routes with average 'truck' values greater than 7.

    Args:
        df (pandas.DataFrame)

    Returns:
        list: List of route names with average 'truck' values greater than 7.
    """
    # Logic for filtering routes
    truck_above_7 = df.groupby('route')['truck'].mean()
    filtered_routes = truck_above_7[truck_above_7 > 7].index.sort_values().tolist()
    return filtered_routes

result_question_4 = filter_routes(df1)
print("Question 4 Result:")
print(result_question_4)
print()

def multiply_matrix(matrix):
    """
    Multiplies matrix values with custom conditions.

    Args:
        matrix (pandas.DataFrame)

    Returns:
        pandas.DataFrame: Modified matrix with values multiplied based on custom conditions.
    """
    # Logic for modifying matrix values
    modified_matrix = matrix.copy()
    modified_matrix[modified_matrix > 20] *= 0.75
    modified_matrix[modified_matrix <= 20] *= 1.25
    return modified_matrix.round(1)

result_question_5 = multiply_matrix(result_question_1)
print("Question 5 Result:")
print(result_question_5)
print()

def time_check(df):
    """
    Use shared dataset-2 to verify the completeness of the data by checking whether the timestamps for each unique (`id`, `id_2`) pair cover a full 24-hour and 7 days period.

    Args:
        df (pandas.DataFrame)

    Returns:
        pd.Series: return a boolean series
    """
    df['start_time'] = pd.to_datetime(df['startDay'] + ' ' + df['startTime'], format='%A %H:%M:%S')
    df['end_time'] = pd.to_datetime(df['endDay'] + ' ' + df['endTime'], format='%A %H:%M:%S')

    completeness_check = df.groupby(['id', 'id_2']).apply(
        lambda x: (x['startDay'].dt.time.min() != pd.Timestamp('00:00:00').time()) or
                  (x['endDay'].dt.time.max() != pd.Timestamp('23:59:59').time()) or
                  (len(x['startDay'].dt.dayofweek.unique()) != 7)
    )
    return completeness_check

result_question_6 = time_check(df2)
print("Question 6 Result:")
print(result_question_6)
print()
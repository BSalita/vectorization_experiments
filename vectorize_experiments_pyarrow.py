import numpy as np
import pandas as pd
import time
import os
import random
from concurrent.futures import ProcessPoolExecutor
import pyarrow as pa
import pyarrow.compute as pc

def process_data_cpu(data):
    result = data[:, 0]
    for col in data.T[1:]:
        result = np.logical_and(result, col)
    return result

def process_data_pyarrow(data):
    # Convert the numpy array to a list of PyArrow arrays
    arrow_arrays = [pa.array(data[:, i]) for i in range(data.shape[1])]
    # Create column names
    column_names = [f'col{i}' for i in range(data.shape[1])]
    # Create a PyArrow Table from the list of arrays and column names
    data_table = pa.table(arrow_arrays, names=column_names)
    result = data_table.column(0).combine_chunks()
    for col in data_table.columns[1:]:
        result = pc.and_kleene(result, col.combine_chunks())
    # Convert the PyArrow array to a NumPy array (ensure correct conversion for boolean type)
    return np.array(result)

def create_random_data_part(num_rows, num_cols):
    # Generate a part of the random data
    return np.random.choice([True, False], size=(num_rows, num_cols))

def create_random_data(num_rows, num_cols, ncpus=8):
    # Use multiprocessing to generate data in parts
    rows_per_part = num_rows // ncpus
    with ProcessPoolExecutor(max_workers=ncpus) as executor:
        parts = list(executor.map(create_random_data_part, [rows_per_part] * ncpus, [num_cols] * ncpus))
    # Concatenate the parts into a full DataFrame
    data = np.vstack(parts)
    df = pd.DataFrame(data, columns=[f'col{i}' for i in range(num_cols)])
    return df

def select_random_columns(df, num_cols):
    # Randomly select 1 to 6 columns
    selected_count = random.randint(1, 6)
    selected_columns = random.sample(list(df.columns), selected_count)
    return df[selected_columns].to_numpy()

def test_combination(delegate, num_processes, data):
    reference_result = None
    elapsed_total_time = 0
    iterations = 5

    for i in range(iterations):
        start_time = time.perf_counter()
        results = delegate(data, num_processes)
        end_time = time.perf_counter()

        elapsed_time = end_time - start_time
        elapsed_total_time += elapsed_time

        final_result = results[0]

        if reference_result is None:
            reference_result = final_result
        else:
            if not np.array_equal(reference_result, final_result):
                print(f"Discrepancy found at iteration {i + 1}")
                break

        print(f"{i + 1}/{iterations}: Elapsed: {elapsed_time:.4f}s")

    print(f"Elapsed Total: {elapsed_total_time:.4f}s Mean: {elapsed_total_time / iterations:.4f}s")

def main(num_processes):
    num_rows = 10_000_000
    num_cols = 335
    print(f"Number of rows: {num_rows}, Number of columns: {num_cols}, Number of processes: {num_processes}")

    start_time = time.perf_counter()
    df = create_random_data(num_rows, num_cols)
    data = select_random_columns(df, num_cols)
    elapsed_time = time.perf_counter() - start_time
    print(f"DataFrame creation and selection time: {elapsed_time:.4f}s")

    configurations = [
        (False, False, lambda d, np: [process_data_cpu(d) for _ in range(np)]),
        (False, True, lambda d, np: list(ProcessPoolExecutor(max_workers=np).map(process_data_cpu, [d] * np))),
        (True, False, lambda d, np: [process_data_pyarrow(d) for _ in range(np)]),
        (True, True, lambda d, np: list(ProcessPoolExecutor(max_workers=np).map(process_data_pyarrow, [d] * np))),
    ]

    for use_accelerator, use_multiprocessing, delegate in configurations:
        print(f"\nTesting with Accelerator: {use_accelerator}, Multiprocessing: {use_multiprocessing}")
        test_combination(delegate, num_processes, data)

if __name__ == '__main__':
    main(num_processes=max(4, int(os.cpu_count() / 4)))

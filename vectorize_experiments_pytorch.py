import numpy as np
import pandas as pd
import time
import os
import random
from concurrent.futures import ProcessPoolExecutor
import torch

def process_data_torch(data, device='cpu'):
    data_torch = torch.tensor(data, device=device)
    result = data_torch[:, 0]
    for col in data_torch.T[1:]:
        result = torch.logical_and(result, col)
    return result.cpu().numpy()

def process_data_cpu(data):
    return process_data_torch(data, device='cpu')

def process_data_gpu(data):
    return process_data_torch(data, device='cuda')

def create_random_data(num_rows, num_cols, device='cpu'):
    # Generate the random data using PyTorch
    data = torch.randint(0, 2, (num_rows, num_cols), dtype=torch.bool, device=device)
    data = data.cpu().numpy()
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Number of rows: {num_rows}, Number of columns: {num_cols}, Number of processes: {num_processes}, Device: {device}")

    start_time = time.perf_counter()
    df = create_random_data(num_rows, num_cols, device=device)
    data = select_random_columns(df, num_cols)
    elapsed_time = time.perf_counter() - start_time
    print(f"DataFrame creation and selection time: {elapsed_time:.4f}s")

    configurations = [
        (False, False, lambda d, np: [process_data_cpu(d) for _ in range(np)]),
        (False, True, lambda d, np: list(ProcessPoolExecutor(max_workers=np).map(process_data_cpu, [d] * np))),
        (True, False, lambda d, np: [process_data_gpu(d) for _ in range(np)] if torch.cuda.is_available() else [process_data_cpu(d) for _ in range(np)]),
        (True, True, lambda d, np: list(ProcessPoolExecutor(max_workers=np).map(process_data_gpu, [d] * np)) if torch.cuda.is_available() else list(ProcessPoolExecutor(max_workers=np).map(process_data_cpu, [d] * np))),
    ]

    for use_accelerator, use_multiprocessing, delegate in configurations:
        using_accelerator = use_accelerator and torch.cuda.is_available()
        print(f"\nTesting with Accelerator: {use_accelerator}, Multiprocessing: {use_multiprocessing}, Using Accelerator: {using_accelerator}")
        test_combination(delegate, num_processes, data)

if __name__ == '__main__':
    main(num_processes=max(4, int(os.cpu_count() / 4)))

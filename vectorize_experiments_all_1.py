
"""
Requires 30GB of RAM to run this script. Tested on Windows 11 only.
cupy needs a one-time setup to use the GPU. The first time it is used, it will take a long time to load. Subsequent runs will be faster.

Conclusions from limited testing on a couple Windows 11 systems:
1. Overall winner is pytorch. It is the fastest on GPU and almost the winner on CPU. Windows 11 installs are manual but work well.
2. Best on GPU: pytorch
3. Best on CPU: tensorflow, although no GPU support on Windows 11, sometimes compares well against GPU implementations.
4. Sometimes polars and pyarrow perform well too.
5. Best compatibility: pytorch cpu and gpu are most compatible as they run on Windows 11. tensorflow gpu isn't supported on Windows 11.
6. Dataframe compatibility: pandas, polars
7. Worst installs (Windows 11): tensorflow cpu, jax, cupy, numba
8. Shoutouts to ibis-duckdb for being good overall.
9. Worst of the bunch: numba doesn't nothing well. cupy has fiddly installs.

Installation instructions
1. recommend that a new virtual environment (conda) is created for this script
2. pip install -U -r requirements.txt
3. pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
4. pip install cupy-cuda12x [sic]

As far as installing cudf, I haven't found a way to do so on my Windows 11 system. All of the following failed.
conda create -n rapids-24.06 -c rapidsai -c conda-forge -c nvidia rapids=24.06 python=3.11 cuda-version=12.2
pip install --extra-index-url=https://pypi.nvidia.com "cudf-cu12==24.6.*"
conda create -n bridge11 -c rapidsai -c conda-forge -c nvidia  python=3.11.8 cuda-version=12.2 
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.6.* dask-cudf-cu12==24.6.* cuml-cu12==24.6.*
pip install --extra-index-url=https://pypi.nvidia.com cudf-cu12==24.6.* dask-cudf-cu12==24.6.* cuml-cu12==24.6.* cugraph-cu12==24.6.* cuspatial-cu12==24.6.* cuproj-cu12==24.6.* cuxfilter-cu12==24.6.* cucim-cu12==24.6.* pylibraft-cu12==24.6.* raft-dask-cu12==24.6.* cuvs-cu12==24.6.*
"""

import cupy as cp
import dask.array as da
import dask.dataframe as dd
import ibis
import ibis.backends.dask
import ibis.backends.duckdb
import ibis.backends.pandas
import ibis.backends.polars
import jax
import jax.numpy as jnp
import numba as nb
import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.compute as pc

import tensorflow as tf
import torch
import xarray as xr
import hashlib
import time
from statistics import mean

rows = 10_000_000
cols = 335
samples = 50
column_names = [f'col_{i}' for i in range(cols)]
use_gpu = False

# Create a matrix of rows x cols with random boolean values
start_time = time.perf_counter()
array_numpy_bool = np.random.randint(0, 2, size=(rows, cols), dtype=bool)
creation_time_numpy = time.perf_counter() - start_time
print('array_numpy_bool created:', array_numpy_bool.shape, f"in {creation_time_numpy:.2f}s")

# Convert to pandas DataFrame
start_time = time.perf_counter()
df_pandas_bool = pd.DataFrame(array_numpy_bool, columns=column_names)
creation_time_pandas = time.perf_counter() - start_time
print('pandas dataFrame created:', df_pandas_bool.shape, f"in {creation_time_pandas:.2f}s")
assert df_pandas_bool.dtypes.apply(lambda x: x == bool).all(), "DataFrame columns are not all boolean"

# Convert to CuPy array
start_time = time.perf_counter()
array_cupy_cpu_bool = array_numpy_bool # not really anthing more than numpy unless cp. is called.
creation_time_cupy_cpu = time.perf_counter() - start_time
print('CuPy array created:', array_cupy_cpu_bool.shape, f"in {creation_time_cupy_cpu:.2f}s")

# Convert to Dask array
start_time = time.perf_counter()
array_dask_bool = dd.from_pandas(df_pandas_bool, npartitions=10) # accepting default number of partitions for now
creation_time_dask = time.perf_counter() - start_time
print('Dask array created:', df_pandas_bool.shape, f"in {creation_time_dask:.2f}s")

# Initialize Ibis with various backends
start_time = time.perf_counter()
ibis_backend_dask = ibis.backends.dask.Backend()
ibis_table_dask = ibis_backend_dask.connect().from_dataframe(df_pandas_bool, 'df_bool')
creation_time_ibis_dask = time.perf_counter() - start_time
print('ibis dask created:', (ibis_table_dask.count().execute(), len(ibis_table_dask.columns)), f"in {creation_time_ibis_dask:.2f}s")

start_time = time.perf_counter()
ibis_backend_duckdb = ibis.backends.duckdb.Backend()
ibis_conn_duckdb = ibis_backend_duckdb.connect()
ibis_table_duckdb = ibis_conn_duckdb.read_in_memory(df_pandas_bool, 'df_pandas_bool')
creation_time_ibis_duckdb = time.perf_counter() - start_time
print('ibis duckdb created:', (ibis_table_duckdb.count().execute(), len(ibis_table_duckdb.columns)), f"in {creation_time_ibis_duckdb:.2f}s")

start_time = time.perf_counter()
ibis_backend_pandas = ibis.backends.pandas.Backend()
ibis_conn_pandas = ibis_backend_pandas.connect()
ibis_table_pandas = ibis_conn_pandas.from_dataframe(df_pandas_bool, 'df_bool')
creation_time_ibis_pandas = time.perf_counter() - start_time
print('ibis pandas created:', (ibis_table_pandas.count().execute(), len(ibis_table_pandas.columns)), f"in {creation_time_ibis_pandas:.2f}s")

# Corrected Polars connection using the latest Polars Ibis API
start_time = time.perf_counter()
ibis_backend_polars = ibis.backends.polars.Backend()
ibis_conn_polars = ibis_backend_polars.connect()
ibis_table_polars = ibis_conn_polars.read_pandas(df_pandas_bool, table_name='df_bool')
creation_time_ibis_polars = time.perf_counter() - start_time
print('ibis polars created:', (ibis_table_polars.count().execute(), len(ibis_table_polars.columns)), f"in {creation_time_ibis_polars:.2f}s")

# Convert to JAX cpu array
start_time = time.perf_counter()
array_jax_cpu_bool = jnp.array(array_numpy_bool)
creation_time_jax_cpu = time.perf_counter() - start_time
print('JAX cpu array created:', array_jax_cpu_bool.shape, f"in {creation_time_jax_cpu:.2f}s")

# Convert to JAX gpu array
start_time = time.perf_counter()
array_jax_gpu_bool = jnp.array(array_numpy_bool)
creation_time_jax_gpu = time.perf_counter() - start_time
print('JAX gpu array created:', array_jax_gpu_bool.shape, f"in {creation_time_jax_gpu:.2f}s")

# Convert to Numba array
start_time = time.perf_counter()
array_numba_bool = array_numpy_bool
creation_time_numba = time.perf_counter() - start_time
print('numba array created:', array_numba_bool.shape, f"in {creation_time_numba:.2f}s")

# Convert to Polars DataFrame
start_time = time.perf_counter()
array_polars_bool = pl.DataFrame(array_numpy_bool)
creation_time_polars = time.perf_counter() - start_time
print('Polars DataFrame created:', array_polars_bool.shape, f"in {creation_time_polars:.2f}s")

# Convert to PyArrow array
start_time = time.perf_counter()
array_pyarrow_bool = pa.Table.from_pandas(df_pandas_bool)
creation_time_pyarrow = time.perf_counter() - start_time
print('PyArrow array created:', array_pyarrow_bool.shape, f"in {creation_time_pyarrow:.2f}s")

# Convert to PyTorch tensor
start_time = time.perf_counter()
array_torch_cpu_bool = torch.from_numpy(array_numpy_bool)
creation_time_pytorch_cpu = time.perf_counter() - start_time
print('PyTorch cpu created:', array_torch_cpu_bool.shape, f"in {creation_time_pytorch_cpu:.2f}s")

# Convert to TensorFlow tensor
start_time = time.perf_counter()
array_tensorflow_cpu_bool = tf.convert_to_tensor(array_numpy_bool)
creation_time_tensorflow_cpu = time.perf_counter() - start_time
print('TensorFlow cpu created:', array_tensorflow_cpu_bool.shape, f"in {creation_time_tensorflow_cpu:.2f}s")

# Convert to xarray DataArray
start_time = time.perf_counter()
array_xarray_cpu_bool = xr.DataArray(array_numpy_bool)
creation_time_xarray = time.perf_counter() - start_time
print('xarray DataArray created:', array_xarray_cpu_bool.shape, f"in {creation_time_xarray:.2f}s")

if use_gpu:
    use_cupy_gpu = cp.cuda.runtime.getDeviceCount() > 0
    use_pytorch_gpu = torch.cuda.is_available()
    use_tensorflow_gpu = len(tf.config.list_physical_devices('GPU')) > 0
    use_jax_gpu = jax.default_backend() != 'cpu'
else:
    use_cupy_gpu = False
    use_pytorch_gpu = False
    use_tensorflow_gpu = False
    use_jax_gpu = False

methods = [
    'cupy-cpu',
    'dask',
    'ibis-dask',
    'ibis-duckdb',
    'ibis-pandas',
    'ibis-polars',
    'jax-cpu',
    'numba',
    'numpy',
    'pandas',
    'polars',
    'pyarrow',
    'pytorch-cpu',
    'tensorflow-cpu',
    'xarray',
]

if use_cupy_gpu:
    methods.append('cupy-gpu')
    methods.append('xarray-cupy')
if use_pytorch_gpu:
    methods.append('pytorch-gpu')
if use_tensorflow_gpu:
    methods.append('tensorflow-gpu')
if use_jax_gpu:
    methods.append('jax-gpu')

# Pre-select columns
selected_indices_l = [np.random.choice(cols, np.random.randint(1, 6), replace=False) for _ in range(samples)] # for each sample, randomly select between 1 and 5 columns.
print('selected_indices_l created:', len(selected_indices_l))
selected_columns_l = [[column_names[i] for i in indices] for indices in selected_indices_l] # convert indices to column names
print('selected_columns_l created:', len(selected_columns_l))

summary_l = []
timing_dict = {method: [] for method in methods}
results_list = {method: [] for method in methods}
creation_times = {
    'cupy-cpu': creation_time_cupy_cpu,
    'dask': creation_time_dask,
    'ibis-dask': creation_time_ibis_dask,
    'ibis-duckdb': creation_time_ibis_duckdb,
    'ibis-pandas': creation_time_ibis_pandas,
    'ibis-polars': creation_time_ibis_polars,
    'jax-cpu': creation_time_jax_cpu,
    'numba': creation_time_numba,
    'numpy': creation_time_numpy,
    'pandas': creation_time_pandas,
    'polars': creation_time_polars,
    'pyarrow': creation_time_pyarrow,
    'pytorch-cpu': creation_time_pytorch_cpu,
    'tensorflow-cpu': creation_time_tensorflow_cpu,
    'xarray': creation_time_xarray,
}

results_hash_dict = None
for method in methods:
 
    if method == 'cupy-gpu' and use_cupy_gpu:
        # CuPy GPU setup
        start_time = time.perf_counter()
        array_cupy_gpu_bool = cp.asarray(array_numpy_bool)
        creation_time_cupy_gpu = time.perf_counter() - start_time
        creation_times['cupy-gpu'] = creation_time_cupy_gpu
        print('CuPy gpu created:', array_cupy_gpu_bool.shape, f"in {creation_time_cupy_gpu:.2f}s")

    if method == 'jax-gpu' and use_jax_gpu:
        # JAX GPU setup
        start_time = time.perf_counter()
        array_jax_gpu_bool = jax.device_put(array_jax_cpu_bool, jax.devices('gpu')[0])
        creation_time_jax_gpu = time.perf_counter() - start_time
        creation_times['jax-gpu'] = creation_time_jax_gpu
        print('JAX gpu setup completed:', array_jax_gpu_bool.shape, f"in {creation_time_jax_gpu:.2f}s")
        
    if method == 'pytorch-gpu' and use_pytorch_gpu:
        # PyTorch GPU setup
        start_time = time.perf_counter()
        array_torch_gpu_bool = array_torch_cpu_bool.to('cuda')
        creation_time_pytorch_gpu = time.perf_counter() - start_time
        creation_times['pytorch-gpu'] = creation_time_pytorch_gpu
        print('PyTorch gpu created:', array_torch_gpu_bool.shape, f"in {creation_time_pytorch_gpu:.2f}s")

    if method == 'tensorflow-gpu' and use_tensorflow_gpu:
        # TensorFlow GPU setup
        start_time = time.perf_counter()
        array_tensorflow_gpu_bool = tf.convert_to_tensor(array_numpy_bool)
        creation_time_tensorflow_gpu = time.perf_counter() - start_time
        creation_times['tensorflow-gpu'] = creation_time_tensorflow_gpu
        print('TensorFlow gpu created:', array_tensorflow_gpu_bool.shape, f"in {creation_time_tensorflow_gpu:.2f}s")

    if method == 'xarray-cupy' and use_cupy_gpu:
        # xarray GPU setup
        start_time = time.perf_counter()
        array_xarray_cupy_bool = tf.convert_to_tensor(array_numpy_bool)
        creation_time_xarray_cupy = time.perf_counter() - start_time
        creation_times['xarray-cupy'] = creation_time_xarray_cupy
        print('xarray cupy created:', array_xarray_cupy_bool.shape, f"in {creation_time_xarray_cupy:.2f}s")

    results_hash_dict = {}

    for i, (selected_indices, selected_columns) in enumerate(zip(selected_indices_l, selected_columns_l)):
        start_time = time.perf_counter()

        if method == 'cupy-cpu':
            result = array_cupy_cpu_bool[:, selected_indices].all(axis=1)
        elif method == 'cupy-gpu':
            assert use_cupy_gpu
            result = cp.all(array_cupy_gpu_bool[:, selected_indices], axis=1)
        elif method == 'dask':
            result = array_dask_bool[selected_columns].all(axis=1).compute()
        elif method == 'ibis-dask':
            expr = ibis_table_dask[selected_columns[0]]
            for col in selected_columns[1:]:
                expr = expr & ibis_table_dask[col]
            result = expr.execute()
        elif method == 'ibis-duckdb':
            expr = ibis_table_duckdb[selected_columns[0]]
            for col in selected_columns[1:]:
                expr = expr & ibis_table_duckdb[col]
            result = expr.execute()
        elif method == 'ibis-pandas':
            expr = ibis_table_pandas[selected_columns[0]]
            for col in selected_columns[1:]:
                expr = expr & ibis_table_pandas[col]
            result = expr.execute()
        elif method == 'ibis-polars':
            expr = ibis_table_polars[selected_columns[0]]
            for col in selected_columns[1:]:
                expr = expr & ibis_table_polars[col]
            result = expr.execute()
        elif method == 'jax-cpu':
            result = jnp.all(array_jax_cpu_bool[:, selected_indices], axis=1)
        elif method == 'jax-gpu':
            assert use_jax_gpu
            result = jnp.all(array_jax_gpu_bool[:, selected_indices], axis=1)
        elif method == 'numba':
            @nb.njit(cache=True)
            def numba_and(a):
                x = a[:, 0]
                for i in range(1, a.shape[1]):
                    x = np.logical_and(x, a[:, i])
                return x
            result = numba_and(array_numpy_bool[:, selected_indices])
        elif method == 'numpy':
            result = array_numpy_bool[:, selected_indices].all(axis=1)
        elif method == 'pandas':
            result = df_pandas_bool[selected_columns].all(axis=1)
        elif method == 'polars':
            result = array_polars_bool[:, selected_indices].to_numpy().all(axis=1)
        elif method == 'pyarrow':
            expr = array_pyarrow_bool.column(selected_columns[0])
            for col in selected_columns[1:]:
                expr = pc.and_kleene(expr, array_pyarrow_bool.column(col))
            result = expr.to_numpy()
        elif method == 'pytorch-cpu':
            result = array_torch_cpu_bool[:, selected_indices].all(dim=1)
        elif method == 'pytorch-gpu':
            assert use_pytorch_gpu
            result = array_torch_gpu_bool[:, selected_indices].all(dim=1) #.cpu().numpy()
        elif method == 'tensorflow-cpu':
            result = tf.reduce_all(tf.gather(array_tensorflow_cpu_bool, selected_indices, axis=1), axis=1)
        elif method == 'tensorflow-gpu' and use_tensorflow_gpu:
            assert use_tensorflow_gpu
            result = tf.reduce_all(tf.gather(array_tensorflow_gpu_bool, selected_indices, axis=1), axis=1)
        elif method == 'xarray':
            result = array_xarray_cpu_bool[:, selected_indices].all(axis=1)
        elif method == 'xarray-cupy':
            assert use_cupy_gpu
            result = xr.DataArray(cp.asarray(array_numpy_bool))[:, selected_indices].all(axis=1)
        else:
            raise NotImplementedError(f"Method {method} is not implemented or GPU is not available")

        elapsed = time.perf_counter() - start_time
        timing_dict[method].append(elapsed)

        print(f"{method} {i}/{samples}, Time: {elapsed:.2f}s, Columns: {selected_columns}, Iteration: {i}, Shape: {result.shape}")

        # Compute hash of the result
        if isinstance(result, np.ndarray): # cupy-cpu, numba, numpy, pyarrow, polars
            result_hash = hashlib.md5(result.tobytes()).hexdigest()
        elif isinstance(result, cp.ndarray): # cupy-gpu
            result_hash = hashlib.md5(cp.asnumpy(result).tobytes()).hexdigest()
        elif isinstance(result, pd.Series): # dask, ibis-dask, ibis-duckdb, ibis-pandas, ibis-polars, pandas
            result_hash = hashlib.md5(result.to_numpy().tobytes()).hexdigest()
        elif isinstance(result, torch.Tensor): # pytorch-cpu, pytorch-gpu
            result_hash = hashlib.md5(result.cpu().numpy().tobytes()).hexdigest()
        elif isinstance(result, tf.Tensor): # tensorflow-cpu, tensorflow-gpu
            result_hash = hashlib.md5(result.numpy().tobytes()).hexdigest()
        elif isinstance(result, jnp.ndarray): # jax-cpu, jax-gpu
            result_hash = hashlib.md5(result.copy().tobytes()).hexdigest()
        elif isinstance(result, xr.DataArray): # xarray
            result_hash = hashlib.md5(result.to_numpy().tobytes()).hexdigest()
        else:
            raise NotImplementedError(f"Method {method} return type is not implemented: {type(result)}")

        if i in results_hash_dict:
            assert result_hash == results_hash_dict[i], f"Results do not match reference hash for {method} at iteration {i}"
        else:
            results_hash_dict[i] = result_hash

    # Free up GPU resources for each method
    if method == 'cupy-gpu':
        del array_cupy_gpu_bool
        cp.cuda.Stream.null.synchronize()
        cp.get_default_memory_pool().free_all_blocks()
    elif method == 'jax-gpu':
        del array_jax_gpu_bool
        jax.clear_backends()
    elif method == 'pytorch-gpu' and use_pytorch_gpu:
        del array_torch_gpu_bool
        torch.cuda.empty_cache()
    elif method == 'tensorflow-gpu' and use_tensorflow_gpu:
        del array_tensorflow_gpu_bool
        tf.keras.backend.clear_session()

# Summary of times
summary_data = []
for method in methods:
    times = timing_dict[method]
    creation_time = creation_times[method]
    sum_sample_execution_times = sum(times)
    total_time = creation_time + sum_sample_execution_times
    mean_total_time = total_time / samples
    summary_data.append({
        "Method": method,
        "Mean Total Time (s)": mean_total_time,
        "Sum of Sample Execution Times (s)": sum_sample_execution_times,
        "Creation Time (s)": creation_time,
        "Total Time (s)": total_time,
   })

# Create DataFrame and sort by mean total time
summary_df = pd.DataFrame(summary_data)
summary_df = summary_df.sort_values(by="Mean Total Time (s)")

# Print summary
print('Summary')
print(summary_df.to_string(index=False))

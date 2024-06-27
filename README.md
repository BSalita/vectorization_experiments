# Vectorization Experiments
Experiments with informal benchmarking of vectorization libs such as pandas, polars, pytorch, tensorflow and others. The programming language is python.

## Motivation
The primary motivator was to find the fastest solutions for processing columns of a large dataframe. I had outgrown pandas scalability.

## The Problem
A dataframe has a shape of 10_000_000 x 335. The need was to perform an average of 3 columnar logical AND operations over 4_000_000 times. I had to develop a vectorization mindset and select the most promising vectorization libs.

## The Experiments
First, a single python program was written to benchmark all major vectorization libs. ChatGPT 4.0o was given the task and did incredibly well. Next, the most promising libs were further benchmarked individually in individual files.

## The Conclusion
The results show, for my situation, several libs were worthy of implementing: Pytorch (CPU and GPU), Tensorflow (CPU only on Windows 11), pyarrow and possibly polars.

# Benchmark datasets

Run the evaluation commands from the repository root after configuring the model API and Sandbox Master in the [main README](../README.md).

- `--subset kernelbench --split train` selects the bundled 270-task KernelBench snapshot (100 Level 1, 100 Level 2, 50 Level 3, and 20 Level 4). The JSONL preserves the task fields from the Parquet snapshot. Use `--slice 0:1` for a first run.
- `--subset /path/to/tasks.parquet --split train` also accepts a single local Parquet file with the runner's schema. A local JSONL file is accepted the same way. Rows need `code` and either `level` plus `name`, or a unique `instance_id`. The optional `modeling` and `config` fields supply model context for Level 4.
- Obtain SOL-ExecBench from the [official dataset](https://huggingface.co/datasets/nvidia/SOL-ExecBench) under NVIDIA's dataset terms. Its upstream schema requires conversion to the KODA row format first. If you already have the converted Parquet, supply its local path as above. The 235-problem upstream dataset and the 202-task sampled evaluation reported in the model card have different scopes.

KernelBench task sources are from [ScalingIntelligence/KernelBench](https://github.com/ScalingIntelligence/KernelBench). Level 4 includes model context from [Hugging Face Transformers](https://github.com/huggingface/transformers), with source notices preserved. See the [KernelBench MIT license](licenses/KernelBench-MIT.txt) and [Transformers Apache 2.0 license](licenses/Transformers-Apache-2.0.txt).

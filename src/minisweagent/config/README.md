# ckpt140 controller configurations

- controller_h.yaml: Qwen3.6/ckpt140 on H100/H20, compile_arch 9.0.
- controller_b.yaml: Qwen3.6/ckpt140 on B200, compile_arch 10.0a.

Both use the same controller and model sampling/budgets. Fill model and sandbox endpoints at runtime. Do not commit credentials or outputs.

# Mac loading check

The first attempt used Transformers 4.57.6 with a direct MPS device map. Its caching-allocator warm-up tried to allocate a single 14.10 GiB Metal buffer and failed with `RuntimeError: Invalid buffer size: 14.10 GiB`, before any response was generated. The failed manifest is retained in outputs/hardware_initial_load_failure.json.

The loading implementation was changed before study inference: load the unquantized checkpoint as FP16 on CPU, then use PyTorch's ordinary per-parameter transfer to MPS. The controlled inference backend remains GPU; all model parameter devices and dtypes are verified after transfer. No quantization, CPU inference fallback or memory-limit override is used.

The three-response synthetic hardware test then completed with three valid explanation-first parses. Observed GPU driver memory after the final hardware response was 15.45 GiB. Loading took about 75 seconds on this first successful attempt. These synthetic responses are infrastructure checks, not e-SNLI evaluation evidence. The separate study smoke estimates throughput on frozen e-SNLI sources.

# Lab Practicum: OpenMP Multi-Core Scaling in Python

**Student:** Not provided  
**Measurement date:** October 1, 2026  
**Machine:** macOS 15.6, Apple Silicon (ARM), 10 logical CPUs; the exact chip model was not available in this environment.  
**Environment:** Python 3.9, NumPy 2.0.2, Numba 0.60.0, Matplotlib 3.9.4; isolated virtual environment.  
**Threads:** Numba maximum: 10. Results are from a single run in this environment, so normal timing variation is possible.

## Challenge 1 — Monte Carlo and Amdahl’s Law

The estimated value of π for 120,000,000 samples was approximately **3.1415–3.1418**, depending on the thread count.

| Threads | Time (s) | Speedup | Efficiency |
|---:|---:|---:|---:|
| 1 | 0.3178 | 1.00× | 100.0% |
| 2 | 0.1626 | 1.95× | 97.7% |
| 4 | 0.0970 | 3.28× | 81.9% |
| 8 | 0.0710 | 4.48× | 56.0% |
| 10 (maximum) | 0.0689 | 4.61× | 46.1% |

**A.** Baseline time `T₁ = 0.3178 s`; time with the maximum number of threads `T₁₀ = 0.0689 s`. The best measured speedup was **4.61×**.

**B.** Numba compiles the function to machine code on its first call. If compilation is included in the measurement, compile time is incorrectly counted as computation time. A small warm-up call performs compilation in advance.

**C.** Efficiency fell below 100%; scaling gains became less than linear from 4 threads onward. Contributing factors include the overhead of distributing work and synchronizing the reduction, cache and memory bandwidth limits, and differences between physical cores, SMT, and logical CPUs. This environment reports only the overall ARM CPU and logical CPU count, so the measurement does not isolate the contribution of each factor. Amdahl’s Law also limits speedup due to any sequential portion of the program.

**D.** Numba recognizes `inside_circle += 1` as a scalar reduction in `prange`: threads accumulate partial counts, which Numba then combines. This is equivalent to a sum reduction (`reduction(+:inside_circle)` in OpenMP terminology) and avoids a race on a shared counter.

## Challenge 2 — Mandelbrot

| Decomposition | Size | Threads | Time |
|---|---:|---:|---:|
| By rows | 2500 × 2500, 1000 maximum iterations | 10 | 0.8752 s |
| By columns | 2500 × 2500, 1000 maximum iterations | 10 | 0.9285 s |

**A.** The row-parallel version finished **0.0533 s** sooner (about 5.7% faster than the column version). Each thread processes more neighboring image elements, which generally makes better use of the cache and contiguous memory regions.

**B.** NumPy stores C-contiguous arrays in row-major order. In the row-parallel loop, neighboring `img[r, c]` elements are adjacent in memory. When work is divided by columns, threads write separate points with a large stride between rows; this uses cache lines less efficiently and can increase memory traffic. Both variants compute the same set of values.

**C.** The central regions of the Mandelbrot set generally require more iterations than points farther outside the set, which escape quickly. With static assignment of large blocks, a thread assigned costly rows keeps working while threads assigned easier blocks may already be idle. Dynamic scheduling gives another block to a thread as soon as it becomes available, reducing idle time at the cost of scheduling overhead. This lab compared the two decomposition axes using `parallel=True`; it did not explicitly configure a separate dynamic scheduling mode.

**D.** The generated image is saved as [mandelbrot_output.png](mandelbrot_output.png).

## Challenge 3 — Heat-Diffusion Stencil

1500 × 1500 cells, 300 steps, maximum of 10 threads.

| Precision | Time | Throughput |
|---|---:|---:|
| `float64` | 0.1218 s | 5540.45 megacells/s |
| `float32` | 0.0661 s | 10219.35 megacells/s |

**A.** `float64` result: **0.1218 s**, **5540.45 megacells/s**.

**B.** With `float32`, runtime decreased by a factor of **1.84** (about 45.8%). Each value uses half the memory, so more cells can be processed for the same cache and RAM traffic. This can reduce runtime more than would be expected from the arithmetic change alone.

**C.** At every step, the stencil reads neighboring cells and writes an output. Once memory bandwidth is saturated, additional cores cannot receive data any faster, so performance is limited by memory bandwidth rather than compute capacity. In this regime, doubling the CPU cores does not double performance.

## Deliverables

- [Mandelbrot image](mandelbrot_output.png)
- [Benchmark script](run_lab.py)
- [Numeric results in JSON](results.json)

## Reproducing the results

From the task directory, run `work/venv/bin/python outputs/run_lab.py`. The script reruns all three challenges and overwrites the result files in `outputs/`.

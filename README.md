## Build Instructions

Compile using Kokkos:

```bash
g++ -std=c++17 -O3 -fopenmp \
  -I/path/to/kokkos/include \
  -L/path/to/kokkos/lib \
  main.cpp numerical_integrator_main.cpp \
  -lkokkoscore -lkokkoscontainers -lkokkosalgorithms -lkokkossimd \
  -ldl -lpthread \
  -o integrator
```

## Sample Output

```
=== Kokkos Numerical Integration Demo ===
Hardware threads: 24
Execution space: N6Kokkos6OpenMPE

--- Small Problem Tests (50 intervals) ---
Rectangle x^2: 0.3234 (0 ms)
Trapezoidal x^2: 0.3334 (0 ms)
Simpson x^2: 0.333333 (0 ms)

--- Large Problem Serial vs Parallel Comparison ---

Problem size: 100000 intervals
Rectangle x^2:
  Serial:   0.333328 (0.3 ms)
  Parallel: 0.333328 (0.0 ms)
  Speedup:  10.4x
  Expected: 0.333333

Problem size: 1000000 intervals
Rectangle x^2:
  Serial:   0.333333 (2.7 ms)
  Parallel: 0.333333 (0.1 ms)
  Speedup:  51.4x
  Expected: 0.333333

Problem size: 10000000 intervals
Rectangle x^2:
  Serial:   0.333333 (26.4 ms)
  Parallel: 0.333333 (3.8 ms)
  Speedup:  7.0x
  Expected: 0.333333
```

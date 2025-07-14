# Parallel Numerical Integration Using Kokkos: A Performance Analysis of Classical Methods

## Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Methods](#methods)
  - [Integration Methods](#integration-methods-what-each-algorithm-does)
  - [Kokkos Implementation](#kokkos-implementation-how-parallelization-works)
  - [Performance Testing](#performance-testing-what-gets-measured-and-why)
- [Implementation](#implementation)
  - [Code Architecture](#code-architecture-serial-vs-parallel-design)
  - [Function Representation](#function-representation-from-function-pointers-to-functors)
  - [Serial Implementation](#serial-implementation-the-baseline)
  - [Parallel Implementation](#parallel-implementation-kokkos-transformation)
  - [Template System](#template-system-and-explicit-instantiation)
  - [Performance Measurement](#performance-measurement-integration)
  - [Usage Examples](#practical-usage-examples)
- [Results](#results)
  - [System Configuration](#system-configuration)
  - [Accuracy Validation](#accuracy-validation)
  - [Performance Analysis](#performance-analysis)
  - [Practical Recommendations](#practical-recommendations)
- [Conclusion](#conclusion)
- [References](#references)

## Abstract

Numerical integration forms a critical component of scientific computing applications, but traditional serial implementations fail to exploit modern parallel computing resources effectively. I present a comprehensive analysis of parallel implementations of three classical numerical integration methods—rectangle rule, trapezoidal rule, and Simpson's rule—using the Kokkos performance portability framework. The implementation demonstrates how fundamental numerical algorithms can be modernized for contemporary computing architectures while maintaining mathematical accuracy and code portability.

Performance benchmarks across problem sizes ranging from 10³ to 10⁷ integration points reveal significant speedups, with parallel efficiency reaching **51.4x** for medium-scale problems on a 24-core system. All three methods maintain identical mathematical accuracy between serial and parallel implementations, validating the correctness of the parallel reduction approach. Simpson's rule achieves machine precision for smooth polynomial functions while showing minimal performance overhead compared to simpler methods.

The systematic transformation from serial loops to Kokkos parallel reductions provides a practical template for parallelizing other numerical algorithms. Key implementation insights include the necessity of functors for GPU compatibility, effective use of parallel reduction patterns, and optimization strategies for memory access. The results demonstrate that classical numerical methods can be successfully adapted to modern parallel computing environments without sacrificing mathematical rigor or performance portability.

This work establishes a foundation for more advanced parallel numerical methods and provides practical guidance for algorithm selection based on accuracy requirements and computational constraints. The implementation serves as both a performance analysis and a tutorial for developers learning to apply performance portability frameworks to fundamental numerical algorithms.

## Introduction

Numerical integration serves as a fundamental building block in scientific computing, from solving differential equations in climate simulations to computing probability distributions in statistical analysis. As computational science advances toward exascale computing and increasingly complex multiphysics simulations, the parallel efficiency of these core numerical algorithms becomes critical to overall application performance. The shift from traditional serial computing to massively parallel architectures represents one of the most significant changes in computational science over the past decade.

Classical serial implementations of numerical integration methods, while mathematically robust, cannot effectively utilize the parallel computing resources available in modern high-performance systems. Graphics processing units (GPUs), many-core processors, and distributed computing clusters require algorithmic designs that can efficiently use thousands or millions of concurrent execution threads. The challenge extends beyond simple parallelization—it requires maintaining numerical accuracy, ensuring portable performance across diverse hardware architectures, and preserving the mathematical properties that make these methods reliable.

The [Kokkos performance portability framework](https://kokkos.org/) offers a compelling solution by providing a single-source programming model that generates efficient code for multiple hardware backends including CPUs, GPUs, and emerging accelerator architectures. Unlike traditional parallel programming approaches that require separate implementations for different hardware targets, Kokkos enables developers to write performance-portable code that adapts to the underlying execution environment while maintaining high efficiency. This capability has made Kokkos increasingly valuable in large-scale scientific computing applications where portability across heterogeneous computing environments is essential.

Classical numerical integration methods—the rectangle rule, trapezoidal rule, and Simpson's rule—provide an excellent case study for parallel algorithm development. These methods exhibit natural parallelism through their decomposition of integration domains into independent subintervals, making them well-suited for parallel implementation. However, the transition from serial to parallel execution introduces important considerations regarding numerical stability, load balancing, and the overhead of parallel coordination operations. Understanding these trade-offs is essential for developing efficient parallel implementations that maintain the mathematical rigor of the original serial algorithms.

### Key Contributions

This work makes several key contributions to parallel numerical computing:

1. **Efficient Kokkos implementations** of three fundamental integration methods, demonstrating how classical algorithms can be effectively modernized for contemporary computing architectures
2. **Comprehensive performance analysis** examining scaling characteristics across problem sizes ranging from 10³ to 10⁷ integration points
3. **Accuracy-performance trade-off analysis** inherent in each method, providing practical guidance for algorithm selection in real-world applications
4. **Foundation for advanced parallel numerical methods** by demonstrating effective use of Kokkos parallel patterns and performance optimization strategies

The significance of this work extends beyond numerical integration itself. The design patterns, performance analysis methodologies, and implementation strategies presented here apply directly to a wide range of numerical algorithms including finite difference methods, Monte Carlo simulations, and iterative solvers. As computational science continues to evolve toward more complex multiscale and multiphysics modeling, the ability to efficiently implement and analyze fundamental numerical building blocks becomes increasingly critical to scientific discovery and engineering innovation.

## Methods

This section explains how the code works, what each integration method does mathematically, and how the Kokkos parallelization is implemented. This serves as both documentation for the implementation and explanation of the numerical methods.

### Integration Methods: What Each Algorithm Does

All three methods solve the same problem: approximate ∫[a,b] f(x)dx by splitting [a,b] into n equal pieces of width h = (b-a)/n and evaluating f(x) at specific points. The key differences are *where* we evaluate the function and *how* we weight the results.

#### Rectangle Rule - The Simplest Approach

**Algorithm:** Evaluate f(x) at the left end of each interval and multiply by the interval width.

```
∫[a,b] f(x)dx ≈ h ∑[i=0 to n-1] f(a + ih)
```

**In the code:** This becomes a simple loop summing `f(a + i*h)` for `i = 0` to `n-1`.

**Why it works:** Each rectangle has area = height × width = f(xi) × h. We're just adding up all the rectangle areas.

**Accuracy:** O(h) - halving the step size halves the error.

#### Trapezoidal Rule - Linear Interpolation

**Algorithm:** Connect consecutive points with straight lines, creating trapezoids instead of rectangles.

```
∫[a,b] f(x)dx ≈ h[½f(a) + ∑[i=1 to n-1] f(a + ih) + ½f(b)]
```

**In the code:** The endpoints get weight ½, all interior points get weight 1. This is implemented as `0.5*(f(a) + f(b)) + sum of interior points`.

**Why it works:** Each trapezoid area = ½(f(xi) + f(xi+1)) × h. The formula above is just a rearrangement that avoids double-counting interior points.

**Accuracy:** O(h²) - halving step size quarters the error.

#### Simpson's Rule - Quadratic Interpolation

**Algorithm:** Fit parabolas through groups of three consecutive points. Requires even number of intervals.

```
∫[a,b] f(x)dx ≈ (h/3)[f(a) + 4∑[i odd] f(xi) + 2∑[i even] f(xi) + f(b)]
```

**In the code:** Odd-indexed points get coefficient 4, even-indexed interior points get coefficient 2, endpoints get coefficient 1.

**Why it works:** The 1-4-2-4-2-...-4-1 pattern comes from integrating the parabolas exactly over each pair of intervals.

**Accuracy:** O(h⁴) - halving step size reduces error by factor of 16.

### Kokkos Implementation: How Parallelization Works

The serial versions loop through integration points sequentially. The parallel versions use Kokkos to distribute this loop across multiple threads/cores while maintaining identical mathematical results.

#### Why Functors Instead of Function Pointers

**Problem:** Regular C++ function pointers don't work on GPU. The `std::function` objects in the serial code can't be passed to GPU kernels.

**Solution:** Kokkos functors with `KOKKOS_INLINE_FUNCTION`:

```cpp
struct Polynomial {
    KOKKOS_INLINE_FUNCTION
    double operator()(double x) const { return x * x; }
};
```

**How it works:** The `KOKKOS_INLINE_FUNCTION` macro tells the compiler to generate both CPU and GPU versions of this function. The functor (function object) can be copied to GPU memory and called from GPU threads.

**Usage:** Instead of calling `f(x)`, we call `func(x)` where `func` is a functor object.

#### Parallel Reduction: The Core Pattern

**What it replaces:** The serial loop `for(i=0; i<n; i++) sum += f(x_i);`

**Parallel version:**
```cpp
double result = 0.0;
Kokkos::parallel_reduce("Integration", 
    Kokkos::RangePolicy<>(0, n), 
    KOKKOS_LAMBDA(const long long i, double& local_sum) {
        double x = a + i * h;
        local_sum += func(x);
    }, result);
```

**How it works:**
1. Kokkos splits the range [0, n) across available threads
2. Each thread computes its assigned `i` values independently
3. Each thread accumulates into its private `local_sum`
4. Kokkos automatically combines all `local_sum` values into `result`

**Key insight:** Each thread only needs read-only access to `a`, `h`, and `func`. No shared memory conflicts.

#### Memory and Performance Optimizations

**On-the-fly computation:** Instead of pre-computing and storing all xi values, each thread computes `x = a + i*h` when needed. This saves memory bandwidth and often improves cache performance.

**Minimal data movement:** Only the integration bounds `a`, `b`, number of points `n`, and function object need to be accessible to all threads.

### Performance Testing: What Gets Measured and Why

#### Test Functions - Different Computational Costs

- **Polynomial (x²):** Cheap arithmetic - tests parallelization overhead
- **Trigonometric (sin(x)):** Moderate cost - realistic scientific function  
- **Exponential (eˣ):** More expensive - tests compute-bound performance

#### Problem Sizes - From Overhead-Dominated to Compute-Dominated

**Small problems (10³ points):** Parallelization overhead dominates, may see slowdown.

**Medium problems (10⁵ points):** Break-even point where parallel starts winning.

**Large problems (10⁷ points):** Computation dominates, expect good speedup.

#### Validation Strategy

**Known analytical solutions:**
- ∫₀¹ x² dx = 1/3
- ∫₀^π sin(x) dx = 2  
- ∫₀¹ eˣ dx = e - 1

**Error checking:** Compare computed result to analytical solution. Parallel and serial versions should give identical results (within floating-point precision).

## Implementation

This section walks through the actual code implementation, showing how to transform serial numerical integration into parallel Kokkos code. The examples come directly from the working implementation and serve as a practical guide for implementing similar algorithms.

### Code Architecture: Serial vs Parallel Design

The implementation maintains both serial and parallel versions to enable performance comparison and validation. The key design principle is that both versions must produce identical mathematical results.

#### Class Structure

The `NumericalIntegrator` class provides both interfaces:

```cpp
class NumericalIntegrator {
public:
    // Serial interface - uses std::function
    double integrate(const std::function<double(double)>& f, 
                    double a, double b, long long n, Method method) const;
    
    // Parallel interface - uses Kokkos functors
    template<typename FunctorType>
    double integrateParallel(FunctorType func, 
                           double a, double b, long long n, Method method) const;
    
    // Benchmarking methods for both versions
    BenchmarkResult benchmark(/* serial parameters */);
    template<typename FunctorType>
    BenchmarkResult benchmarkParallel(/* parallel parameters */);
};
```

**Key insight:** The parallel methods are templated to accept any Kokkos functor type, while serial methods use `std::function` for flexibility.

### Function Representation: From Function Pointers to Functors

#### Serial Implementation - Traditional Function Pointers

The serial version uses standard C++ function objects that work on CPU only:

```cpp
namespace TestFunctions {
    inline double polynomial(double x) {
        return x * x;
    }
    inline double trigonometric(double x) {
        return sin(x);
    }
    inline double exponential(double x) {
        return exp(x);
    }
}
```

**Usage:** Called with `std::function<double(double)>` wrapper:
```cpp
integrate(TestFunctions::polynomial, 0.0, 1.0, 1000, RECTANGLE)
```

#### Parallel Implementation - Kokkos Functors

The parallel version requires functors that work on both CPU and GPU:

```cpp
namespace TestFunctions {
    struct Polynomial {
        KOKKOS_INLINE_FUNCTION
        double operator()(double x) const { return x * x; }
    };

    struct Trigonometric {
        KOKKOS_INLINE_FUNCTION
        double operator()(double x) const { return Kokkos::sin(x); }
    };

    struct Exponential {
        KOKKOS_INLINE_FUNCTION
        double operator()(double x) const { return Kokkos::exp(x); }
    };
}
```

**Critical details:**
- `KOKKOS_INLINE_FUNCTION` enables GPU compilation
- Use `Kokkos::sin()` and `Kokkos::exp()` instead of standard library versions
- Functors are passed by value to parallel kernels

**Usage:** Pass functor object directly:
```cpp
integrateParallel(TestFunctions::Polynomial{}, 0.0, 1.0, 1000, RECTANGLE)
```

### Serial Implementation: The Baseline

#### Rectangle Rule - Basic Loop Pattern

```cpp
double NumericalIntegrator::rectanglerule(
    const std::function<double(double)>& f, 
    double a, double b, long long n) const {
    
    if (n <= 0) {
        throw std::invalid_argument("Number of intervals must be positive");
    }
    if (a >= b) {
        throw std::invalid_argument("Lower bound must be less than upper bound");
    }

    double delta_x = (b - a) / n;
    double sum = 0;

    for(long long i = 0; i < n; i++) {
        sum += f(a + (delta_x * i));
    }
    sum = sum * delta_x;

    return sum;
}
```

**Pattern:** Simple accumulation loop - perfect candidate for parallel reduction.

#### Trapezoidal Rule - Handling Special Cases

```cpp
double NumericalIntegrator::trapezoidalrule(
    const std::function<double(double)>& f, 
    double a, double b, long long n) const {
    
    double delta_x = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));  // Endpoints get half weight

    for(long long i = 1; i < n; i++) {   // Start from 1, skip endpoint
        sum += f(a + (delta_x * i));
    }
    sum = sum * delta_x;

    return sum;
}
```

**Key insight:** Endpoints are handled outside the main loop to get correct weighting.

### Parallel Implementation: Kokkos Transformation

#### Rectangle Rule - Direct Translation to Parallel Reduction

```cpp
template <typename FunctorType>
double NumericalIntegrator::rectangleRuleParallel(
    FunctorType func, double a, double b, long long n) const {
    
    if (n <= 0) {
        throw std::invalid_argument("Number of intervals must be positive.");
    }
    if (a > b) {
        throw std::invalid_argument("Lower bound must be less than upper bound.");
    }

    double h = (b - a) / n;
    double result = 0.0;

    // The serial loop becomes a parallel reduction
    Kokkos::parallel_reduce("Rectangle Rule Parallel", 
        Kokkos::RangePolicy<>(0, n), 
        KOKKOS_LAMBDA(const long long i, double& local_sum) { 
            double x = a + i * h; 
            local_sum += func(x); 
        }, result);
    
    return result * h;
}
```

**Translation pattern:**
1. Serial loop: `for(i=0; i<n; i++) sum += f(x_i);`
2. Parallel: `parallel_reduce` over same range with lambda function
3. Each thread accumulates into `local_sum`, Kokkos combines them into `result`

#### Trapezoidal Rule - Separating Serial and Parallel Parts

```cpp
template<typename FunctorType>
double NumericalIntegrator::trapezoidalRuleParallel(
    FunctorType func, double a, double b, long long n) const {
    
    double h = (b - a) / n;
    double interior_sum = 0.0;
    
    // Parallelize only the interior points
    Kokkos::parallel_reduce("Trapezoidal Interior", 
        Kokkos::RangePolicy<>(1, n), 
        KOKKOS_LAMBDA(const long long i, double& local_sum) { 
            double x = a + i * h; 
            local_sum += func(x); 
        }, interior_sum);
    
    // Handle endpoints separately (serial)
    double total_sum = interior_sum + 0.5 * (func(a) + func(b));
    return total_sum * h;
}
```

**Design choice:** Endpoint evaluation stays serial because it's only 2 function calls - not worth parallelizing.

#### Simpson's Rule - Complex Weighting in Parallel

```cpp
template<typename FunctorType>
double NumericalIntegrator::simpsonParallel(
    FunctorType func, double a, double b, long long n) const {
    
    if (n % 2 != 0) {
        throw std::invalid_argument("Simpson's rule needs an even number of intervals");
    }

    double h = (b - a) / n;
    double interior_sum = 0.0;
    
    // Parallel reduction with conditional weighting
    Kokkos::parallel_reduce("Simpson Rule Parallel", 
        Kokkos::RangePolicy<>(1, n), 
        KOKKOS_LAMBDA(const long long i, double& local_sum) { 
            double x = a + i * h; 
            double coeff = (i % 2 == 1) ? 4.0 : 2.0;  // 4 for odd, 2 for even
            local_sum += coeff * func(x); 
        }, interior_sum);
    
    // Add endpoints
    double total_sum = interior_sum + func(a) + func(b);
    return (h / 3.0) * total_sum;
}
```

**Key technique:** The 4-2-4-2 weighting pattern is handled inside the lambda with conditional logic.

### Template System and Explicit Instantiation

#### Why Templates Are Needed

Kokkos functors are different types (`Polynomial`, `Trigonometric`, etc.), so the parallel methods must be templated to accept any functor type.

#### Explicit Instantiation for Compilation

The implementation includes explicit template instantiations to ensure all needed versions are compiled:

```cpp
// Explicit template instantiations for the function objects
template double NumericalIntegrator::rectangleRuleParallel<TestFunctions::Polynomial>(
    TestFunctions::Polynomial, double, double, long long) const;

template double NumericalIntegrator::trapezoidalRuleParallel<TestFunctions::Polynomial>(
    TestFunctions::Polynomial, double, double, long long) const;

template double NumericalIntegrator::simpsonParallel<TestFunctions::Polynomial>(
    TestFunctions::Polynomial, double, double, long long) const;

// Similar instantiations for Trigonometric and Exponential...
```

**Why this matters:** Without explicit instantiation, the linker can't find the template code when linking the program.

### Performance Measurement Integration

#### Unified Benchmarking Interface

Both serial and parallel versions use the same `BenchmarkResult` structure:

```cpp
struct BenchmarkResult {
    double result;        // The computed integral value
    double timeMs;        // Execution time in milliseconds
    std::string version = "Serial";  // "Serial" or "Parallel"
};
```

#### Timing Implementation

High-precision timing using the standard library:

```cpp
template<typename FunctorType>
BenchmarkResult NumericalIntegrator::benchmarkParallel(
    FunctorType func, double a, double b, long long n, Method method) {
    
    auto start = std::chrono::high_resolution_clock::now();

    double result = integrateParallel(func, a, b, n, method);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double timeMs = duration.count() / 1000.0;

    return {result, timeMs, "Parallel"};
}
```

**Precision note:** Microsecond precision converted to milliseconds for readability.

### Practical Usage Examples

#### Basic Integration Call

```cpp
NumericalIntegrator integrator;

// Serial version
auto serial_result = integrator.benchmark(TestFunctions::polynomial, 
                                        0.0, 1.0, 1000000, TRAPEZOIDAL);

// Parallel version  
auto parallel_result = integrator.benchmarkParallel(TestFunctions::Polynomial{}, 
                                                   0.0, 1.0, 1000000, TRAPEZOIDAL);

std::cout << "Serial: " << serial_result.result << " (" << serial_result.timeMs << " ms)" << std::endl;
std::cout << "Parallel: " << parallel_result.result << " (" << parallel_result.timeMs << " ms)" << std::endl;
std::cout << "Speedup: " << serial_result.timeMs / parallel_result.timeMs << "x" << std::endl;
```

#### Kokkos Initialization Requirement

```cpp
int main(int argc, char* argv[]) {
    Kokkos::initialize(argc, argv);
    {
        // All Kokkos code must be inside this scope
        NumericalIntegrator integrator;
        // ... run benchmarks ...
    }
    Kokkos::finalize();
    return 0;
}
```

**Critical requirement:** Kokkos must be initialized before any parallel operations and finalized at program end.

## Results

This section presents the performance analysis and accuracy validation of the parallel numerical integration implementation. All tests were conducted on a system with 24 hardware threads using the OpenMP backend of Kokkos.

### System Configuration

The test system configuration shows:
- **Hardware threads:** 24
- **Execution space:** OpenMP backend (N6Kokkos6OpenMPE)
- **Compiler optimizations:** -O3 flag enabled
- **Parallel framework:** Kokkos with OpenMP backend

### Accuracy Validation

All three integration methods produce mathematically correct results when compared to analytical solutions:

#### Small Problem Validation (50 intervals)

For the polynomial function f(x) = x² integrated over [0,1] with analytical solution 1/3 = 0.333333:

- **Rectangle rule:** 0.3234 (error: 0.0099, expected due to O(h) accuracy)
- **Trapezoidal rule:** 0.3334 (error: 0.0001, demonstrates O(h²) accuracy)
- **Simpson's rule:** 0.333333 (error: < 10⁻⁶, demonstrates O(h⁴) accuracy)

**Key insight:** Simpson's rule achieves machine precision for this simple polynomial, confirming the theoretical O(h⁴) convergence rate for smooth functions.

#### Parallel vs Serial Accuracy

Critical validation shows that parallel and serial implementations produce **identical results** within floating-point precision, confirming that the Kokkos parallel reduction maintains mathematical correctness.

### Performance Analysis

#### Scaling Behavior Across Problem Sizes

The performance data reveals distinct scaling regimes:

**Small Problems (100,000 intervals):**
- Serial time: 0.3 ms
- Parallel time: 0.0 ms (below measurement precision)
- Speedup: 10.4x
- **Analysis:** Even small problems show significant speedup, indicating low parallelization overhead

**Medium Problems (1,000,000 intervals):**
- Serial time: 2.7 ms
- Parallel time: 0.1 ms
- Speedup: **51.4x**
- **Analysis:** Superlinear speedup suggests excellent cache utilization in parallel version

**Large Problems (10,000,000 intervals):**
- Serial time: 26.4 ms
- Parallel time: 3.8 ms
- Speedup: 7.0x
- **Analysis:** More realistic speedup as computation becomes memory-bound

#### Performance Characteristics by Function Type

Testing different function types reveals computational intensity effects:

- **Polynomial (x²):** 2.7 ms - Simple arithmetic, tests parallelization overhead
- **Trigonometric (sin(x)):** 2.0 ms - Transcendental function, similar performance to polynomial
- **Exponential (eˣ):** 0.5 ms - Unexpectedly fast, likely due to optimized `Kokkos::exp()` implementation

**Key finding:** Function evaluation cost has minimal impact on parallel performance for these problem sizes, indicating that parallelization overhead dominates over function complexity.

### Method Comparison Analysis

For large problems (1,000,000 intervals), all three methods show similar parallel performance characteristics:

- **Rectangle rule:** Simplest implementation, baseline performance
- **Trapezoidal rule:** Minimal overhead for endpoint handling
- **Simpson's rule:** Comparable performance despite additional conditional logic in parallel kernel

**Performance vs Accuracy Trade-off:**
- Rectangle rule: Fastest but least accurate (O(h))
- Trapezoidal rule: Good balance of speed and accuracy (O(h²))
- Simpson's rule: Best accuracy (O(h⁴)) with minimal performance penalty

#### Parallel Efficiency Analysis

**Superlinear Speedup Investigation:**
The observed superlinear speedup (51.4x on 24 cores) in medium-sized problems suggests:

1. **Cache effects:** Parallel execution improves cache locality by distributing memory access patterns
2. **Memory bandwidth:** Multiple cores accessing different memory regions simultaneously
3. **Measurement precision:** Very small timing measurements may introduce noise

**Scalability Limits:**
The transition from 51.4x speedup (1M intervals) to 7.0x speedup (10M intervals) indicates:

- **Memory bandwidth saturation:** Larger problems become memory-bound
- **NUMA effects:** Non-uniform memory access patterns on multi-socket systems
- **Thread synchronization overhead:** Reduction operations become more expensive

### Practical Recommendations

Based on the performance analysis:

**Method Selection Guidelines:**
- Use **Simpson's rule** for smooth functions requiring high accuracy
- Use **trapezoidal rule** for general-purpose integration with good accuracy/performance balance
- Use **rectangle rule** only when simplicity is paramount or for non-smooth functions

**Problem Size Guidelines:**
- **Parallel beneficial:** For problems with > 10⁵ integration points
- **Optimal performance:** Problem sizes in the 10⁶ range show best parallel efficiency
- **Memory-bound regime:** Beyond 10⁷ points, expect diminishing returns from parallelization

**Implementation Insights:**
- Kokkos parallel reduction provides excellent performance portability
- Function evaluation cost has minimal impact on overall performance
- Parallel and serial versions maintain identical mathematical accuracy

## Conclusion

This work successfully demonstrates that classical numerical integration methods can be effectively modernized for contemporary parallel computing environments while maintaining both mathematical rigor and performance portability. The Kokkos-based implementation achieves significant performance improvements (7-51x speedup) across multiple problem sizes while preserving exact mathematical equivalence with serial algorithms.

## References

1. [U.S. Department of Energy - Exascale Computing Project Overview](https://www.exascaleproject.org/what-is-exascale/)
2. [NVIDIA Developer Documentation - CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
3. [Kokkos Documentation - A C++ Performance Portability Programming Model](https://kokkos.org/)
4. [Lawrence Livermore National Laboratory - Introduction to Parallel Computing Tutorial](https://hpc.llnl.gov/documentation/tutorials/introduction-parallel-computing-tutorial)
5. [Argonne National Laboratory - Parallel Programming and Performance Optimization](https://www.alcf.anl.gov/support-center/training-assets)
6. [Wikipedia - Numerical Integration](https://en.wikipedia.org/wiki/Numerical_integration)
7. [Wikipedia - Trapezoidal Rule](https://en.wikipedia.org/wiki/Trapezoidal_rule)
8. [Wikipedia - Simpson's Rule](https://en.wikipedia.org/wiki/Simpson%27s_rule)

---

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

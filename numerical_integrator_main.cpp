#include "headers.hpp"

double NumericalIntegrator::rectanglerule(const std::function<double(double)>& f, double a, double b, long long n) const {

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

double NumericalIntegrator::trapezoidalrule(const std::function<double(double)>& f, double a, double b, long long n) const {

    if (n <= 0) {
        throw std::invalid_argument("Number of intervals must be positive");
    }
    if (a >= b) {
        throw std::invalid_argument("Lower bound must be less than upper bound");
    }

    double delta_x = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));

    for(long long i = 1; i < n; i++) {
        sum += f(a + (delta_x * i));
    }
    sum = sum * delta_x;

    return sum;
}

double NumericalIntegrator::simpson(const std::function<double(double)>& f, double a, double b, long long n) const {

    if (n % 2 != 0) {
        throw std::invalid_argument("Simpson's rule needs an even number of intervals");
    }

    if (n <= 0) {
        throw std::invalid_argument("Number of intervals must be positive");
    }
    if (a >= b) {
        throw std::invalid_argument("Lower bound must be less than upper bound");
    }

    double h = (b - a) / n;
    double sum = f(a) + f(b);

    for (long long i = 1; i < n; i++) {
        double x = a + i * h;

        if (i % 2 == 1) {
            sum += 4 * f(x);
        } else {
            sum += 2 * f(x);
        }
    }
    return (h / 3.0) * sum;
}

double NumericalIntegrator::integrate(const std::function<double(double)>& f, double a, double b, long long n, Method method) const {

    switch (method) {
        case RECTANGLE:
            return rectanglerule(f, a, b, n);
        case TRAPEZOIDAL:
            return trapezoidalrule(f, a, b, n);
        case SIMPSON:
            return simpson(f, a, b, n);
        default:
            throw std::invalid_argument("Invalid method selected.");
    }    
}   

BenchmarkResult NumericalIntegrator::benchmark(const std::function<double(double)>& f, double a, double b, long long n, Method method) {
    auto start = std::chrono::high_resolution_clock::now();

    double result = integrate(f, a, b, n, method);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double timeMs = duration.count() / 1000.0;

    return {result, timeMs};
}

// Parallel Functions

template <typename FunctorType>
double NumericalIntegrator::rectangleRuleParallel(FunctorType func, double a, double b, long long n) const {
    if (n <= 0) {
        throw std::invalid_argument("Number of intervals must be positive.");
    }
    if (a > b) {
        throw std::invalid_argument("Lower bound must be less than upper bound.");
    }

    double h = (b - a) / n;
    double result = 0.0;

    // Using Kokkos Parallel Reduction
    // https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html#parallel-reduce
    Kokkos::parallel_reduce("Rectangle Rule Parallel", Kokkos::RangePolicy<>(0, n), KOKKOS_LAMBDA(const long long i, double& local_sum) { double x = a + i * h; local_sum += func(x); }, result );
    return result * h;
}

template<typename FunctorType>
double NumericalIntegrator::trapezoidalRuleParallel(FunctorType func, double a, double b, long long n) const {
    if (n <= 0) {
        throw std::invalid_argument("Number of intervals must be positive");
    }
    if (a >= b) {
        throw std::invalid_argument("Lower bound must be less than upper bound");
    }

    double h = (b - a) / n;
    double interior_sum = 0.0;
    
    // Using Kokkos Parallel Reduction
    // https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html#parallel-reduce
    Kokkos::parallel_reduce("Trapezoidal Interior", Kokkos::RangePolicy<>(1, n), KOKKOS_LAMBDA(const long long i, double& local_sum) { double x = a + i * h; local_sum += func(x); }, interior_sum );
    
    // Add endpoints (with 1/2 weight)
    double total_sum = interior_sum + 0.5 * (func(a) + func(b));
    return total_sum * h;
}

template<typename FunctorType>
double NumericalIntegrator::simpsonParallel(FunctorType func, double a, double b, long long n) const {
    if (n % 2 != 0) {
        throw std::invalid_argument("Simpson's rule needs an even number of intervals");
    }
    if (n <= 0) {
        throw std::invalid_argument("Number of intervals must be positive");
    }
    if (a >= b) {
        throw std::invalid_argument("Lower bound must be less than upper bound");
    }

    double h = (b - a) / n;
    double interior_sum = 0.0;
    
    // Using Kokkos Parallel Reduction
    // https://kokkos.org/kokkos-core-wiki/ProgrammingGuide/ParallelDispatch.html#parallel-reduce
    Kokkos::parallel_reduce("Simpson Rule Parallel", Kokkos::RangePolicy<>(1, n), KOKKOS_LAMBDA(const long long i, double& local_sum) { double x = a + i * h; double coeff = (i % 2 == 1) ? 4.0 : 2.0; local_sum += coeff * func(x); }, interior_sum );
    
    // Add endpoints
    double total_sum = interior_sum + func(a) + func(b);
    return (h / 3.0) * total_sum;
}

template<typename FunctorType>
double NumericalIntegrator::integrateParallel(FunctorType func, double a, double b, long long n, Method method) const {
    switch (method) {
        case RECTANGLE:
            return rectangleRuleParallel(func, a, b, n);
        case TRAPEZOIDAL:
            return trapezoidalRuleParallel(func, a, b, n);
        case SIMPSON:
            return simpsonParallel(func, a, b, n);
        default:
            throw std::invalid_argument("Invalid method selected.");
    }    
}

template<typename FunctorType>
BenchmarkResult NumericalIntegrator::benchmarkParallel(FunctorType func, double a, double b, long long n, Method method) {
    auto start = std::chrono::high_resolution_clock::now();

    double result = integrateParallel(func, a, b, n, method);

    auto end = std::chrono::high_resolution_clock::now();

    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    double timeMs = duration.count() / 1000.0;

    return {result, timeMs, "Parallel"};
}

void NumericalIntegrator::compareSerialVsParallel(double a, double b, long long n, Method method) {
    std::string methodName;
    switch (method) {
        case RECTANGLE: methodName = "Rectangle"; break;
        case TRAPEZOIDAL: methodName = "Trapezoidal"; break;
        case SIMPSON: methodName = "Simpson"; break;
    }
    
    std::cout << "\n=== " << methodName << " Method Comparison ===" << std::endl;
    std::cout << "Intervals: " << n << ", Range: [" << a << ", " << b << "]" << std::endl;
    
    // Serial benchmark
    auto serial = benchmark(TestFunctions::polynomial, a, b, n, method);
    
    // Parallel benchmark  
    auto parallel = benchmarkParallel(TestFunctions::Polynomial{}, a, b, n, method);
    
    std::cout << "Serial:   " << std::fixed << std::setprecision(8) << serial.result 
              << " (" << std::setprecision(2) << serial.timeMs << " ms)" << std::endl;
    std::cout << "Parallel: " << std::fixed << std::setprecision(8) << parallel.result 
              << " (" << std::setprecision(2) << parallel.timeMs << " ms)" << std::endl;
    std::cout << "Speedup:  " << std::setprecision(1) << serial.timeMs / parallel.timeMs << "x" << std::endl;
    std::cout << "Error difference: " << std::scientific << std::setprecision(2) 
              << std::abs(serial.result - parallel.result) << std::endl;
}

// Explicit template instantiations for the function objects
template double NumericalIntegrator::rectangleRuleParallel<TestFunctions::Polynomial>(TestFunctions::Polynomial, double, double, long long) const;
template double NumericalIntegrator::trapezoidalRuleParallel<TestFunctions::Polynomial>(TestFunctions::Polynomial, double, double, long long) const;
template double NumericalIntegrator::simpsonParallel<TestFunctions::Polynomial>(TestFunctions::Polynomial, double, double, long long) const;
template double NumericalIntegrator::integrateParallel<TestFunctions::Polynomial>(TestFunctions::Polynomial, double, double, long long, Method) const;
template BenchmarkResult NumericalIntegrator::benchmarkParallel<TestFunctions::Polynomial>(TestFunctions::Polynomial, double, double, long long, Method);

template double NumericalIntegrator::rectangleRuleParallel<TestFunctions::Trigonometric>(TestFunctions::Trigonometric, double, double, long long) const;
template double NumericalIntegrator::trapezoidalRuleParallel<TestFunctions::Trigonometric>(TestFunctions::Trigonometric, double, double, long long) const;
template double NumericalIntegrator::simpsonParallel<TestFunctions::Trigonometric>(TestFunctions::Trigonometric, double, double, long long) const;
template double NumericalIntegrator::integrateParallel<TestFunctions::Trigonometric>(TestFunctions::Trigonometric, double, double, long long, Method) const;
template BenchmarkResult NumericalIntegrator::benchmarkParallel<TestFunctions::Trigonometric>(TestFunctions::Trigonometric, double, double, long long, Method);

template double NumericalIntegrator::rectangleRuleParallel<TestFunctions::Exponential>(TestFunctions::Exponential, double, double, long long) const;
template double NumericalIntegrator::trapezoidalRuleParallel<TestFunctions::Exponential>(TestFunctions::Exponential, double, double, long long) const;
template double NumericalIntegrator::simpsonParallel<TestFunctions::Exponential>(TestFunctions::Exponential, double, double, long long) const;
template double NumericalIntegrator::integrateParallel<TestFunctions::Exponential>(TestFunctions::Exponential, double, double, long long, Method) const;
template BenchmarkResult NumericalIntegrator::benchmarkParallel<TestFunctions::Exponential>(TestFunctions::Exponential, double, double, long long, Method);
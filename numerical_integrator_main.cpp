#include "headers.hpp"

const double NumericalIntegrator::rectanglerule(const function<double(double)>& f, double a, double b, long long n) {

    if (n <= 0) {
        throw std::invalid_argument("Number of intervals must be positive");
    }
    if (a >= b) {
        throw std::invalid_argument("Lower bound must be less than upper bound");
    }

    double delta_x = (b - a) / n;
    double sum = 0;

    for(int i = 0; i < n; i++) {
        sum += f(a + (delta_x * i));
    }
    sum = sum * delta_x;

    return sum;
}

const double NumericalIntegrator::trapezoidalrule(const std::function<double(double)>& f, double a, double b, long long n) {

    if (n <= 0) {
        throw std::invalid_argument("Number of intervals must be positive");
    }
    if (a >= b) {
        throw std::invalid_argument("Lower bound must be less than upper bound");
    }

    double delta_x = (b - a) / n;
    double sum = 0.5 * (f(a) + f(b));

    for(int i = 1; i < n; i++) {
        sum += f(a + (delta_x * i));
    }
    sum = sum * delta_x;

    return sum;
}

const double NumericalIntegrator::integrate(const std::function<double(double)>& f, double a, double b, long long n, Method method) {

    if (n <= 0) {
        throw std::invalid_argument("Number of intervals must be positive");
    }
    if (a >= b) {
        throw std::invalid_argument("Lower bound must be less than upper bound");
    }

    switch (method) {
        case RECTANGLE:
            return rectanglerule(f, a, b, n);
        case TRAPEZOIDAL:
            return trapezoidalrule(f, a, b, n);
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
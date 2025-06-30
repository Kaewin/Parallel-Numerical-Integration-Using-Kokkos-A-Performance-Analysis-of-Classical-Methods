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
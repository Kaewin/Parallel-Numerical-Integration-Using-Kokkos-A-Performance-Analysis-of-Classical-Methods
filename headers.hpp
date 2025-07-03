#pragma once
#include <iostream>
#include <functional>
#include <vector>
#include <chrono>
#include <cmath>

// These three have been added on in order to run with Kokkos.
#include <thread>
#include <iomanip>
#include <Kokkos_Core.hpp>

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

enum Method {
    RECTANGLE,
    TRAPEZOIDAL,
    SIMPSON
};

struct BenchmarkResult {
    double result;
    double timeMs;
};

class NumericalIntegrator {
    public:
        double integrate(const std::function<double(double)>& f, double a, double b, long long n, Method method) const;
        BenchmarkResult benchmark(const std::function<double(double)>& f, double a, double b, long long n, Method method);
    private:
        double rectanglerule(const std::function<double(double)>& f, double a, double b, long long n) const;
        double trapezoidalrule(const std::function<double(double)>& f, double a, double b, long long n) const;
        double simpson(const std::function<double(double)>& f, double a, double b, long long n) const;
};